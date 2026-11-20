"""Dual-Track reporting, streaming CSV writers, robustness pack, and verdict gates.

Purpose:
    Computes independent [FIXED 1-LOT] and [DYNAMIC COMPOUNDING] summary
    metrics (BASE, STRESS, PROXY), executes the seeded robustness pack
    (random-entry stream 1, day-block bootstrap stream 2, Dynamic Monte Carlo
    drawdown stream 3, fold stability, cost sweep), evaluates weekly,
    design-gate, holdout, and PAPER_READY verdicts as the single source of
    truth, and atomically writes trades_fixed.csv, trades_dynamic.csv,
    summary.json, manifest.json, verify_sample.md, REPORT.md, and REPORT.html
    (pure HTML5 + <details> + 5 inline SVG charts, zero JS/CDN).

PRD Sections:
    §3.3, §4.2, §5.5, §6.1, §12.1, §12.2 (SVGs & NaN guard), §12.3

Niyam Covered:
    - Niyam 12: Deterministic seeded RNG streams & byte-identical outputs

Guarding Tests:
    T08, T14, T15, T16, T27, T32, T55, T60

PUBLIC API:
    Functions:
        compute_track_summary, compute_cost_sweep, compute_robustness_pack,
        evaluate_weekly_verdict, evaluate_design_gate_verdict,
        evaluate_holdout_verdict, evaluate_paper_ready, write_trades_csv,
        write_summary_json, write_manifest_json, write_verify_sample_md,
        write_report_md, write_report_html, generate_reports
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from contracts import ENGINE_TAG, TRADE_RECORD_COLUMNS, TradeRecord, format_decimal
from costs import ZERO, compute_trade_accounting, round_paise, to_decimal
from data_loader import DayData, DatasetResult, atomic_write_bytes, hhmm_to_minute
from engine import DualTrackResult, TrackRunResult


# ------------------------------------------------------------------------------
# NaN-Safe Numeric Helpers (AUD-016, AUD-017)
# ------------------------------------------------------------------------------
def _finite_floats(vals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _fmt_opt_float(val: float | None, digits: int = 2, suffix: str = "") -> str:
    if val is None or not math.isfinite(float(val)):
        return "NA"
    return f"{float(val):.{digits}f}{suffix}"


def _json_clean(obj: Any) -> Any:
    """Recursively convert Decimal and numpy scalars to plain JSON-safe Python types."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Mapping):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    return str(obj)


# ------------------------------------------------------------------------------
# Track Summary & Drawdown Metrics (§4.1, §4.2, §12.1)
# ------------------------------------------------------------------------------
def compute_track_summary(track_res: TrackRunResult) -> dict[str, Any]:
    """Compute exact summary metrics for one track with zero-denominator & NaN guards."""
    trades = track_res.trades
    n = len(trades)
    wins = sum(1 for t in trades if t.net_base > ZERO)
    losses = n - wins

    gross = round_paise(sum((t.gross for t in trades), ZERO))
    charges_base = round_paise(sum((t.charges_base for t in trades), ZERO))
    fill_cost_base = round_paise(sum((t.fill_cost_base for t in trades), ZERO))
    net_base = round_paise(sum((t.net_base for t in trades), ZERO))

    charges_stress = round_paise(sum((t.charges_stress for t in trades), ZERO))
    fill_cost_stress = round_paise(sum((t.fill_cost_stress for t in trades), ZERO))
    net_stress = round_paise(sum((t.net_stress for t in trades), ZERO))

    proxy_vals = [t.net_proxy for t in trades if t.net_proxy is not None]
    net_proxy: Decimal | None = (
        round_paise(sum(proxy_vals, ZERO)) if (n > 0 and len(proxy_vals) == n) else None
    )

    win_rate_pct: float | None = (100.0 * wins / n) if n > 0 else None

    gross_wins = sum((t.net_base for t in trades if t.net_base > ZERO), ZERO)
    gross_losses = sum((-t.net_base for t in trades if t.net_base <= ZERO), ZERO)
    profit_factor: float | None = None
    if n > 0 and gross_losses > ZERO:
        pf_val = float(gross_wins / gross_losses)
        profit_factor = pf_val if math.isfinite(pf_val) else None

    cost_share: float | None = None
    if gross > ZERO:
        cs_val = float((charges_base + fill_cost_base) / gross)
        cost_share = cs_val if math.isfinite(cs_val) else None

    # Peak-to-trough drawdown over cumulative equity curve
    peak = track_res.start_balance
    cur = track_res.start_balance
    max_dd_inr = ZERO
    max_dd_pct = 0.0
    for tr in trades:
        cur = tr.balance_base if track_res.track == "dynamic" else round_paise(cur + tr.net_base)
        if cur > peak:
            peak = cur
        dd_inr = round_paise(peak - cur)
        if dd_inr > max_dd_inr:
            max_dd_inr = dd_inr
        if peak > ZERO:
            dd_p = float((peak - cur) / peak) * 100.0
            if math.isfinite(dd_p) and dd_p > max_dd_pct:
                max_dd_pct = dd_p

    liq_warn_count = sum(1 for t in trades if t.liquidity_warn)

    return {
        "track": track_res.track,
        "trades_count": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate_pct, 2) if win_rate_pct is not None else None,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "gross": format_decimal(gross, 2),
        "charges_base": format_decimal(charges_base, 2),
        "fill_cost_base": format_decimal(fill_cost_base, 2),
        "net_base": format_decimal(net_base, 2),
        "charges_stress": format_decimal(charges_stress, 2),
        "fill_cost_stress": format_decimal(fill_cost_stress, 2),
        "net_stress": format_decimal(net_stress, 2),
        "net_proxy": format_decimal(net_proxy, 2) if net_proxy is not None else None,
        "cost_share": round(cost_share, 4) if cost_share is not None else None,
        "start_balance": format_decimal(track_res.start_balance, 2),
        "final_balance_base": format_decimal(track_res.final_balance_base, 2),
        "final_balance_stress": format_decimal(track_res.final_balance_stress, 2),
        "max_drawdown_inr": format_decimal(max_dd_inr, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "liquidity_warn_count": liq_warn_count,
        "cap_breach_days": list(track_res.cap_breach_days),
        "ruin_flag": track_res.ruin_flag,
        "ruin_date": track_res.ruin_date,
        "ruin_balance": format_decimal(track_res.ruin_balance, 2)
        if track_res.ruin_balance is not None
        else None,
        "decide_calls": track_res.decide_calls,
        "data_gap_skips": track_res.data_gap_skips,
        "skipped_count": len(track_res.skipped),
    }


# ------------------------------------------------------------------------------
# Cost Sweep & Robustness Pack (§4.2, AUD-022, T32)
# ------------------------------------------------------------------------------
def compute_cost_sweep(
    trades: Sequence[TradeRecord],
    config: Mapping[str, Any],
    scales: Sequence[str] = ("0.5", "1.0", "1.5", "2.0", "3.0"),
) -> dict[str, Any]:
    """Re-price a track's trade list across spread multipliers and estimate breakeven scale."""
    points: dict[str, str] = {}
    for sc_str in scales:
        sc = Decimal(sc_str)
        total_net = ZERO
        for tr in trades:
            e_min = hhmm_to_minute(tr.entry_ts)
            x_min = hhmm_to_minute(tr.exit_ts)
            bd = compute_trade_accounting(
                tr.entry_price,
                tr.exit_price,
                tr.qty,
                e_min,
                x_min,
                tr.dte == 0,
                tr.date,
                config,
                scenario="BASE",
                is_no_bar=(tr.exit_reason == "no_bar"),
                spread_scale=sc,
            )
            assert bd is not None
            total_net = round_paise(total_net + bd.net)
        points[f"{sc_str}x"] = format_decimal(total_net, 2)

    # Linear breakeven spread multiplier estimate from 1.0x and 2.0x points
    n1 = float(Decimal(points["1.0x"]))
    n2 = float(Decimal(points["2.0x"]))
    slope = n1 - n2
    breakeven: float | None = None
    if slope > 0.0 and n1 > 0.0:
        be = 1.0 + (n1 / slope)
        if math.isfinite(be):
            breakeven = round(be, 3)

    return {"net_by_spread_mult": points, "breakeven_spread_mult": breakeven}


def compute_robustness_pack(
    *,
    dual: DualTrackResult,
    days: Mapping[str, DayData],
    trading_dates: Sequence[str],
    config: Mapping[str, Any],
    is_holdout: bool = False,
    delay1_net_fixed: Decimal | None = None,
    delay2_net_fixed: Decimal | None = None,
    delay1_net_dynamic: Decimal | None = None,
    delay2_net_dynamic: Decimal | None = None,
) -> dict[str, Any]:
    """Run seeded robustness tests using exact §4.2 SeedSequence streams 1, 2, 3."""
    seed = int(config["seed"])
    random_runs = int(config["random_runs"])
    mc_runs = int(config["mc_runs"])
    n_folds = int(
        config["holdout_stability_folds"] if is_holdout else config["stability_folds"]
    )
    start_min = hhmm_to_minute(str(config["entry_window_start"]))

    fixed_trades = dual.fixed.trades
    dyn_trades = dual.dynamic.trades

    # --- Stream 1: Random-Entry Permutation Test on Fixed track (§4.2, T32) ---
    rng_rand = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    random_p: float | None = None
    actual_mean_fixed: float | None = None
    if len(fixed_trades) >= 1 and random_runs >= 1:
        actual_nets = _finite_floats([float(t.net_base) for t in fixed_trades])
        if actual_nets.size > 0:
            actual_mean_fixed = float(np.mean(actual_nets))
            # Pre-collect live candidate minutes & close-to-close forward diffs per trade
            cand_arrays: list[np.ndarray] = []
            for tr in fixed_trades:
                day = days.get(tr.date)
                if day is None or (tr.strike, tr.side) not in day.options:
                    cand_arrays.append(np.array([float(tr.net_base)], dtype=np.float64))
                    continue
                ser = day.options[(tr.strike, tr.side)]
                cutoff_key = "expiry_last_entry_time" if day.is_expiry_day else "last_entry_time"
                cutoff_m = min(hhmm_to_minute(str(config[cutoff_key])), day.grid_len)
                hold = max(1, tr.hold_min)
                live_idxs = [
                    m for m in range(start_min, cutoff_m)
                    if bool(ser.live[m]) and (m + hold) < day.grid_len and bool(ser.live[m + hold])
                ]
                if not live_idxs:
                    cand_arrays.append(np.array([float(tr.net_base)], dtype=np.float64))
                    continue
                cost_per_trade = float(tr.charges_base + tr.fill_cost_base)
                vals = np.array(
                    [
                        (float(ser.open[m + hold]) - float(ser.open[m])) * float(tr.qty) - cost_per_trade
                        for m in live_idxs
                    ],
                    dtype=np.float64,
                )
                vals = vals[np.isfinite(vals)]
                cand_arrays.append(vals if vals.size > 0 else np.array([float(tr.net_base)], dtype=np.float64))

            ge_count = 0
            for _ in range(random_runs):
                sim_sum = 0.0
                for arr in cand_arrays:
                    idx = int(rng_rand.integers(0, arr.size))
                    sim_sum += float(arr[idx])
                sim_mean = sim_sum / float(len(cand_arrays))
                if sim_mean >= actual_mean_fixed:
                    ge_count += 1
            random_p = (1.0 + ge_count) / float(random_runs + 1)

    # --- Stream 2: Day-Block Bootstrap 95% CI on Fixed net/trade (§4.2, T32) ---
    rng_boot = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    boot_ci_low: float | None = None
    boot_ci_high: float | None = None
    if len(fixed_trades) >= 1 and len(trading_dates) >= 1 and mc_runs >= 1:
        day_nets: dict[str, list[float]] = {d: [] for d in trading_dates}
        for tr in fixed_trades:
            day_nets.setdefault(tr.date, []).append(float(tr.net_base))
        day_sums = np.array([sum(day_nets[d]) for d in trading_dates], dtype=np.float64)
        day_cnts = np.array([len(day_nets[d]) for d in trading_dates], dtype=np.int64)
        n_days = len(trading_dates)
        boot_means: list[float] = []
        for _ in range(mc_runs):
            picks = rng_boot.integers(0, n_days, size=n_days)
            tot_c = int(np.sum(day_cnts[picks]))
            if tot_c == 0:
                continue  # Skip 0-trade draws (§4.2)
            boot_means.append(float(np.sum(day_sums[picks])) / float(tot_c))
        if boot_means:
            bm = _finite_floats(boot_means)
            if bm.size > 0:
                boot_ci_low = round(float(np.quantile(bm, 0.025)), 4)
                boot_ci_high = round(float(np.quantile(bm, 0.975)), 4)

    # --- Stream 3: Monte Carlo Drawdown (5/50/95%) on Dynamic track (§4.2, T32) ---
    rng_mc = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    mc_dd_pct: dict[str, float] | None = None
    if len(dyn_trades) >= 5 and mc_runs >= 1:
        dyn_nets = _finite_floats([float(t.net_base) for t in dyn_trades])
        start_bal = float(dual.dynamic.start_balance)
        if dyn_nets.size >= 5 and start_bal > 0.0:
            dds = np.empty(mc_runs, dtype=np.float64)
            for i in range(mc_runs):
                perm = rng_mc.permutation(dyn_nets)
                eq = start_bal + np.cumsum(perm)
                eq_full = np.concatenate(([start_bal], eq))
                peaks = np.maximum.accumulate(eq_full)
                with np.errstate(divide="ignore", invalid="ignore"):
                    dd_arr = np.where(peaks > 0.0, (peaks - eq_full) / peaks * 100.0, 0.0)
                dds[i] = float(np.max(dd_arr))
            dds = dds[np.isfinite(dds)]
            if dds.size > 0:
                mc_dd_pct = {
                    "p5": round(float(np.quantile(dds, 0.05)), 4),
                    "p50": round(float(np.quantile(dds, 0.50)), 4),
                    "p95": round(float(np.quantile(dds, 0.95)), 4),
                }

    # --- Fold Stability (5 DESIGN / 3 HOLDOUT) ---
    dates_sorted = sorted(trading_dates)
    folds_fixed: list[str] = []
    folds_dynamic: list[str] = []
    pos_folds = 0
    if dates_sorted and n_folds >= 1:
        fold_splits = [list(chunk) for chunk in np.array_split(dates_sorted, n_folds) if len(chunk) > 0]
        for fold_dates in fold_splits:
            fset = set(str(x) for x in fold_dates)
            fn = round_paise(sum((t.net_base for t in fixed_trades if t.date in fset), ZERO))
            dn = round_paise(sum((t.net_base for t in dyn_trades if t.date in fset), ZERO))
            folds_fixed.append(format_decimal(fn, 2))
            folds_dynamic.append(format_decimal(dn, 2))
            if fn > ZERO:
                pos_folds += 1
        positive_folds_pct: float | None = round(100.0 * pos_folds / len(fold_splits), 2)
    else:
        positive_folds_pct = None

    return {
        "random_entry_p_value": round(random_p, 6) if random_p is not None else None,
        "bootstrap_95_ci_low": boot_ci_low,
        "bootstrap_95_ci_high": boot_ci_high,
        "mc_drawdown_pct_dynamic": mc_dd_pct,
        "folds_fixed_net_base": folds_fixed,
        "folds_dynamic_net_base": folds_dynamic,
        "positive_folds_pct_fixed": positive_folds_pct,
        "delay1_net_base_fixed": format_decimal(delay1_net_fixed, 2) if delay1_net_fixed is not None else None,
        "delay2_net_base_fixed": format_decimal(delay2_net_fixed, 2) if delay2_net_fixed is not None else None,
        "delay1_net_base_dynamic": format_decimal(delay1_net_dynamic, 2) if delay1_net_dynamic is not None else None,
        "delay2_net_base_dynamic": format_decimal(delay2_net_dynamic, 2) if delay2_net_dynamic is not None else None,
        "cost_sweep_fixed": compute_cost_sweep(fixed_trades, config),
        "cost_sweep_dynamic": compute_cost_sweep(dyn_trades, config),
    }


# ------------------------------------------------------------------------------
# Single-Source Verdict Gates (§6.1, §12.1, T27, T32, T55)
# ------------------------------------------------------------------------------
def evaluate_weekly_verdict(
    fixed_summary: Mapping[str, Any],
    dynamic_summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate §6.1 R6.5 Weekly PASS / LOW_SAMPLE / FAIL gate (FAIL checked FIRST)."""
    week_min = int(config["week_min_trades"])
    dd_max_pct = float(to_decimal(config["week_dd_max_pct"]) * Decimal("100"))

    f_trades = int(fixed_summary["trades_count"])
    d_trades = int(dynamic_summary["trades_count"])
    f_base = to_decimal(fixed_summary["net_base"])
    d_base = to_decimal(dynamic_summary["net_base"])
    f_stress = to_decimal(fixed_summary["net_stress"])
    d_stress = to_decimal(dynamic_summary["net_stress"])
    d_dd = float(dynamic_summary["max_drawdown_pct"])
    d_ruin = bool(dynamic_summary["ruin_flag"])

    reasons: list[str] = []

    # 1. FAIL takes precedence even if trades < week_min_trades (V5.2 Rule 3.5)
    if d_ruin:
        reasons.append("DYNAMIC_RUIN")
    if d_dd > dd_max_pct:
        reasons.append(f"DYNAMIC_DD_BREACH ({d_dd:.2f}% > {dd_max_pct:.2f}%)")
    if reasons:
        return {"outcome": "FAIL", "reasons": reasons}

    # 2. LOW_SAMPLE if either track has < week_min_trades
    if f_trades < week_min or d_trades < week_min:
        return {
            "outcome": "LOW_SAMPLE",
            "reasons": [f"trades fixed={f_trades}, dynamic={d_trades} < {week_min}"],
        }

    # 3. PASS requires BASE > 0 and STRESS > 0 on BOTH tracks
    if f_base <= ZERO:
        reasons.append("FIXED_BASE_NON_POSITIVE")
    if d_base <= ZERO:
        reasons.append("DYNAMIC_BASE_NON_POSITIVE")
    if f_stress <= ZERO:
        reasons.append("FIXED_STRESS_NON_POSITIVE")
    if d_stress <= ZERO:
        reasons.append("DYNAMIC_STRESS_NON_POSITIVE")

    if reasons:
        return {"outcome": "FAIL", "reasons": reasons}
    return {"outcome": "PASS", "reasons": []}


def evaluate_design_gate_verdict(
    fixed_summary: Mapping[str, Any],
    dynamic_summary: Mapping[str, Any],
    robustness: Mapping[str, Any],
    m_variants: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate §12.1 6-gate DESIGN verdict with Bonferroni m adjustment (T27)."""
    min_t = int(config["min_trades"])
    max_p = float(config["max_p_value"])
    min_folds = float(config["min_positive_folds_pct"])
    m_eff = max(1, int(m_variants))
    p_thresh = max_p / float(m_eff)

    f_trades = int(fixed_summary["trades_count"])
    f_stress = to_decimal(fixed_summary["net_stress"])
    rand_p = robustness.get("random_entry_p_value")
    boot_lo = robustness.get("bootstrap_95_ci_low")
    pos_folds = robustness.get("positive_folds_pct_fixed")
    d1_raw = robustness.get("delay1_net_base_fixed")
    d1_val = to_decimal(d1_raw) if d1_raw is not None else None
    d_ruin = bool(dynamic_summary["ruin_flag"])

    gates = {
        "min_trades": f_trades >= min_t,
        "stress_positive": f_stress > ZERO,
        "random_p_bonferroni": (rand_p is not None and float(rand_p) <= p_thresh),
        "bootstrap_ci_positive": (boot_lo is not None and float(boot_lo) > 0.0),
        "fold_stability": (pos_folds is not None and float(pos_folds) >= min_folds),
        "delay1_positive": (d1_val is not None and d1_val > ZERO),
        "no_dynamic_ruin": not d_ruin,
    }
    failed = [k for k, ok in gates.items() if not ok]
    if not failed:
        verdict = "PASS"
    elif failed == ["min_trades"]:
        verdict = "NOT_YET"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "m_variants": m_eff,
        "bonferroni_p_threshold": round(p_thresh, 6),
        "gates": gates,
        "failed_gates": failed,
    }


def evaluate_holdout_verdict(
    fixed_summary: Mapping[str, Any],
    dynamic_summary: Mapping[str, Any],
    robustness: Mapping[str, Any],
    k_opens: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate §12.1 HOLDOUT verdict with K-adjusted p-value (T32)."""
    min_t = int(config["holdout_min_trades"])
    max_p = float(config["holdout_max_p_value"])
    k_eff = max(1, int(k_opens))
    p_thresh = max_p / float(k_eff)

    f_trades = int(fixed_summary["trades_count"])
    f_stress = to_decimal(fixed_summary["net_stress"])
    rand_p = robustness.get("random_entry_p_value")
    d1_raw = robustness.get("delay1_net_base_fixed")
    d1_val = to_decimal(d1_raw) if d1_raw is not None else None
    d_ruin = bool(dynamic_summary["ruin_flag"])

    gates = {
        "holdout_min_trades": f_trades >= min_t,
        "stress_positive": f_stress > ZERO,
        "delay1_positive": (d1_val is not None and d1_val > ZERO),
        "random_p_k_adjusted": (rand_p is not None and float(rand_p) <= p_thresh),
        "no_dynamic_ruin": not d_ruin,
    }
    failed = [k for k, ok in gates.items() if not ok]
    verdict = "PASS" if not failed else "FAIL"
    return {
        "verdict": verdict,
        "k_opens": k_eff,
        "k_adjusted_p_threshold": round(p_thresh, 6),
        "gates": gates,
        "failed_gates": failed,
    }


def evaluate_paper_ready(
    weekly_pass: bool,
    design_pass: bool,
    holdout_pass: bool,
) -> bool:
    """Award PAPER_READY iff WEEKLY_PASS, DESIGN_PASS, and HOLDOUT_PASS are all True (§12.1)."""
    return bool(weekly_pass and design_pass and holdout_pass)


# ------------------------------------------------------------------------------
# Streaming CSV, JSON & Verify Sample Writers (§3.3, §12.1, §12.3)
# ------------------------------------------------------------------------------
def write_trades_csv(dest: Path, trades: Sequence[TradeRecord]) -> None:
    """Stream TradeRecord rows sorted by (date, entry_ts, trade_no) atomically to CSV."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.parent / f".tmp.{os.getpid()}.{dest.name}"
    ordered = sorted(trades, key=lambda t: (t.date, t.entry_ts, t.trade_no))
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(TRADE_RECORD_COLUMNS), lineterminator="\n"
            )
            writer.writeheader()
            for tr in ordered:
                writer.writerow(tr.to_csv_row())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_summary_json(dest: Path, summary: Mapping[str, Any]) -> None:
    """Write canonical summary.json atomically."""
    payload = json.dumps(_json_clean(summary), sort_keys=True, indent=2) + "\n"
    atomic_write_bytes(dest, payload.encode("utf-8"))


def write_manifest_json(dest: Path, manifest: Mapping[str, Any]) -> None:
    """Write canonical manifest.json atomically."""
    payload = json.dumps(_json_clean(manifest), sort_keys=True, indent=2) + "\n"
    atomic_write_bytes(dest, payload.encode("utf-8"))


def write_verify_sample_md(
    dest: Path,
    fixed_trades: Sequence[TradeRecord],
    dynamic_trades: Sequence[TradeRecord],
    days: Mapping[str, DayData],
    seed: int,
) -> None:
    """Write deterministic seeded sample of <=10 Fixed + <=5 Dynamic trades with 1m candles."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4]))

    def _pick(trades: Sequence[TradeRecord], k: int) -> list[TradeRecord]:
        if len(trades) <= k:
            return list(trades)
        idxs = sorted(int(i) for i in rng.choice(len(trades), size=k, replace=False))
        return [trades[i] for i in idxs]

    lines: list[str] = [f"# Verification Sample ({ENGINE_TAG})", ""]
    for label, subset in (
        ("[FIXED 1-LOT]", _pick(fixed_trades, 10)),
        ("[DYNAMIC COMPOUNDING]", _pick(dynamic_trades, 5)),
    ):
        lines.append(f"## {label} Sample ({len(subset)} trades)")
        if not subset:
            lines.append("_No trades executed on this track._")
            lines.append("")
            continue
        for tr in subset:
            lines.append(
                f"### {label} Trade #{tr.trade_no} — {tr.date} {tr.entry_ts}->{tr.exit_ts} "
                f"{tr.strike}{tr.side} x{tr.lots} ({tr.exit_reason})"
            )
            lines.append(
                f"- Entry: `{tr.entry_price}` | Stop: `{tr.initial_stop}` | Exit: `{tr.exit_price}` "
                f"| Gross: `{tr.gross}` | Charges: `{tr.charges_base}` | FillCost: `{tr.fill_cost_base}` "
                f"| Net(BASE): `{tr.net_base}` | Net(STRESS): `{tr.net_stress}`"
            )
            day = days.get(tr.date)
            if day is not None and (tr.strike, tr.side) in day.options:
                ser = day.options[(tr.strike, tr.side)]
                eb = hhmm_to_minute(tr.entry_ts)
                xb = hhmm_to_minute(tr.exit_ts)
                lo_b = max(0, eb - 1)
                hi_b = min(day.grid_len - 1, xb + 1)
                lines.append("| Bar | O | H | L | C | Live |")
                lines.append("|---|---|---|---|---|---|")
                for b in range(lo_b, min(hi_b + 1, lo_b + 8)):
                    o = _fmt_opt_float(float(ser.open[b]))
                    h = _fmt_opt_float(float(ser.high[b]))
                    l = _fmt_opt_float(float(ser.low[b]))
                    c = _fmt_opt_float(float(ser.close[b]))
                    lines.append(f"| {b} | {o} | {h} | {l} | {c} | {bool(ser.live[b])} |")
            lines.append("")

    atomic_write_bytes(dest, ("\n".join(lines) + "\n").encode("utf-8"))


# ------------------------------------------------------------------------------
# Inline SVG Chart Generators (5 Charts, Zero JS/CDN; §12.2)
# ------------------------------------------------------------------------------
def _svg_waterfall(fixed_trades: Sequence[TradeRecord]) -> str:
    gross = float(sum((t.gross for t in fixed_trades), ZERO))
    fc = float(sum((t.fill_cost_base for t in fixed_trades), ZERO))
    chg = float(sum((t.charges_base for t in fixed_trades), ZERO))
    net = float(sum((t.net_base for t in fixed_trades), ZERO))
    mfe_tot = float(sum((t.mfe * Decimal(t.qty) for t in fixed_trades), ZERO))
    items = [("MFE", mfe_tot, "#64748b"), ("Gross", gross, "#2563eb"), ("FillCost", -fc, "#d97706"), ("Charges", -chg, "#dc2626"), ("Net", net, "#16a34a" if net >= 0 else "#dc2626")]
    max_abs = max([abs(v) for _, v, _ in items] + [1.0])
    bars: list[str] = []
    for i, (lbl, val, col) in enumerate(items):
        x = 30 + i * 95
        h = int(abs(val) / max_abs * 80)
        y = 100 - h if val >= 0 else 100
        bars.append(
            f'<rect x="{x}" y="{y}" width="60" height="{max(h, 2)}" fill="{col}"/>'
            f'<text x="{x + 30}" y="192" text-anchor="middle" font-size="11">{html.escape(lbl)}</text>'
            f'<text x="{x + 30}" y="{max(14, y - 4)}" text-anchor="middle" font-size="10">{val:.0f}</text>'
        )
    return (
        '<svg viewBox="0 0 520 205" width="100%" role="img" aria-label="Waterfall Chart">'
        '<line x1="15" y1="100" x2="505" y2="100" stroke="#94a3b8" stroke-width="1"/>'
        + "".join(bars)
        + "</svg>"
    )


def _svg_equity_curve(fixed_res: TrackRunResult, dyn_res: TrackRunResult) -> str:
    def _pts(res: TrackRunResult) -> list[float]:
        cur = float(res.start_balance)
        pts = [cur]
        for t in res.trades:
            cur = float(t.balance_base) if res.track == "dynamic" else cur + float(t.net_base)
            pts.append(cur)
        return pts

    pf = _pts(fixed_res)
    pd = _pts(dyn_res)
    all_v = pf + pd
    lo, hi = min(all_v), max(all_v)
    span = (hi - lo) if hi > lo else 1.0

    def _poly(pts: list[float], color: str) -> str:
        n = max(len(pts) - 1, 1)
        coords = " ".join(
            f"{20 + int(480 * i / n)},{170 - int(145 * (v - lo) / span)}"
            for i, v in enumerate(pts)
        )
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'

    return (
        '<svg viewBox="0 0 520 195" width="100%" role="img" aria-label="Dual-Track Equity Curve">'
        '<rect x="20" y="20" width="480" height="150" fill="none" stroke="#e2e8f0"/>'
        + _poly(pf, "#2563eb")
        + _poly(pd, "#16a34a")
        + '<text x="24" y="14" font-size="11" fill="#2563eb">[FIXED 1-LOT]</text>'
        + '<text x="140" y="14" font-size="11" fill="#16a34a">[DYNAMIC COMPOUNDING]</text>'
        + "</svg>"
    )


def _svg_exit_reasons(fixed_trades: Sequence[TradeRecord]) -> str:
    totals: dict[str, float] = {}
    for t in fixed_trades:
        totals[t.exit_reason] = totals.get(t.exit_reason, 0.0) + float(t.net_base)
    items = sorted(totals.items())[:8]
    if not items:
        return '<svg viewBox="0 0 520 80" width="100%"><text x="20" y="40" font-size="12">No trades</text></svg>'
    max_abs = max([abs(v) for _, v in items] + [1.0])
    rows: list[str] = []
    for i, (reason, val) in enumerate(items):
        y = 15 + i * 22
        w = int(abs(val) / max_abs * 180)
        col = "#16a34a" if val >= 0 else "#dc2626"
        x = 260 if val >= 0 else 260 - w
        rows.append(
            f'<text x="10" y="{y + 12}" font-size="11">{html.escape(reason)}</text>'
            f'<rect x="{x}" y="{y}" width="{max(w, 2)}" height="14" fill="{col}"/>'
            f'<text x="450" y="{y + 12}" font-size="10">{val:.0f}</text>'
        )
    h = 25 + len(items) * 22
    return (
        f'<svg viewBox="0 0 520 {h}" width="100%" role="img" aria-label="Exit Reason PnL">'
        f'<line x1="260" y1="5" x2="260" y2="{h - 5}" stroke="#94a3b8"/>'
        + "".join(rows)
        + "</svg>"
    )


def _svg_time_weekday_heatmap(fixed_trades: Sequence[TradeRecord]) -> str:
    days_order = ("Mon", "Tue", "Wed", "Thu", "Fri")
    hours = ("09", "10", "11", "12", "13", "14")
    grid_sum: dict[tuple[str, str], float] = {}
    grid_cnt: dict[tuple[str, str], int] = {}
    for t in fixed_trades:
        hr = t.entry_ts[:2]
        key = (t.weekday, hr)
        grid_sum[key] = grid_sum.get(key, 0.0) + float(t.net_base)
        grid_cnt[key] = grid_cnt.get(key, 0) + 1
    cells: list[str] = []
    for r, wd in enumerate(days_order):
        cells.append(f'<text x="8" y="{38 + r * 28}" font-size="11">{wd}</text>')
        for c, hr in enumerate(hours):
            x = 55 + c * 70
            y = 20 + r * 28
            cnt = grid_cnt.get((wd, hr), 0)
            val = grid_sum.get((wd, hr), 0.0)
            fill = "#f1f5f9" if cnt == 0 else ("#bbf7d0" if val >= 0 else "#fecaca")
            dash = ' stroke-dasharray="3,2"' if 0 < cnt < 20 else ""
            cells.append(
                f'<rect x="{x}" y="{y}" width="64" height="24" fill="{fill}" stroke="#64748b"{dash}/>'
                f'<text x="{x + 32}" y="{y + 15}" text-anchor="middle" font-size="9">n={cnt}</text>'
            )
    hdr = "".join(f'<text x="{87 + c * 70}" y="14" text-anchor="middle" font-size="10">{hr}:00</text>' for c, hr in enumerate(hours))
    return f'<svg viewBox="0 0 500 170" width="100%" role="img" aria-label="Time x Weekday Heatmap">{hdr}{"".join(cells)}</svg>'


def _svg_mae_mfe_scatter(fixed_trades: Sequence[TradeRecord]) -> str:
    if not fixed_trades:
        return '<svg viewBox="0 0 520 80" width="100%"><text x="20" y="40" font-size="12">No trades</text></svg>'
    maes = [float(t.mae) for t in fixed_trades]
    mfes = [float(t.mfe) for t in fixed_trades]
    mx_x = max(max(maes), 1.0)
    mx_y = max(max(mfes), 1.0)
    dots: list[str] = []
    for t in fixed_trades[:300]:
        cx = 30 + int(450 * float(t.mae) / mx_x)
        cy = 165 - int(140 * float(t.mfe) / mx_y)
        col = "#16a34a" if t.net_base > ZERO else "#dc2626"
        dots.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{col}" fill-opacity="0.75"/>')
    return (
        '<svg viewBox="0 0 520 190" width="100%" role="img" aria-label="MAE vs MFE Scatter">'
        '<rect x="30" y="20" width="450" height="145" fill="none" stroke="#94a3b8"/>'
        + "".join(dots)
        + '<text x="255" y="184" text-anchor="middle" font-size="11">MAE (pts)</text>'
        + '<text x="14" y="95" font-size="11">MFE</text>'
        + "</svg>"
    )


# ------------------------------------------------------------------------------
# REPORT.md & REPORT.html Writers (§4, §5.5, §12.1)
# ------------------------------------------------------------------------------
def _capped_rows_with_others(
    trades: Sequence[TradeRecord],
    max_rows: int,
) -> tuple[list[TradeRecord], dict[str, Any] | None]:
    """Return up to max_rows trades plus a reconciling Others aggregate row if truncated."""
    ordered = sorted(trades, key=lambda t: (t.date, t.entry_ts, t.trade_no))
    if len(ordered) <= max_rows:
        return ordered, None
    half = max(1, max_rows // 2)
    kept = ordered[:half] + ordered[-(max_rows - half) :]
    omitted = ordered[half : len(ordered) - (max_rows - half)]
    others = {
        "count": len(omitted),
        "gross": format_decimal(round_paise(sum((t.gross for t in omitted), ZERO)), 2),
        "charges_base": format_decimal(round_paise(sum((t.charges_base for t in omitted), ZERO)), 2),
        "fill_cost_base": format_decimal(round_paise(sum((t.fill_cost_base for t in omitted), ZERO)), 2),
        "net_base": format_decimal(round_paise(sum((t.net_base for t in omitted), ZERO)), 2),
        "net_stress": format_decimal(round_paise(sum((t.net_stress for t in omitted), ZERO)), 2),
    }
    return kept, others


def write_report_md(
    dest: Path,
    *,
    summary: Mapping[str, Any],
    dataset: DatasetResult | None = None,
) -> None:
    """Write REPORT.md with explicit [FIXED 1-LOT] and [DYNAMIC COMPOUNDING] headers."""
    fixed = summary["fixed"]
    dyn = summary["dynamic"]
    lines: list[str] = [f"# Backtest Report ({ENGINE_TAG})", ""]

    if dyn.get("ruin_flag"):
        lines.append(
            f"> **[DYNAMIC RUIN]** Halted on `{dyn.get('ruin_date')}` at balance `₹{dyn.get('ruin_balance')}`."
        )
        lines.append("")

    if "verdict" in summary:
        lines.append(f"**Verdict:** `{summary['verdict']}` | **PAPER_READY:** `{summary.get('paper_ready', False)}`")
        lines.append("")

    if dataset is not None:
        lines.append("## Day Taxonomy Summary")
        lines.append(
            f"- Calendar days: `{len(dataset.calendar_days)}` | Market days: `{len(dataset.market_days)}` "
            f"| Excluded: `{len(dataset.excluded_days)}` | Warm-up: `{len(dataset.warmup_days)}` "
            f"| Trading: `{len(dataset.trading_days)}` | Holdout: `{len(dataset.holdout_days)}`"
        )
        lines.append("")

    for header, trk in (("[FIXED 1-LOT]", fixed), ("[DYNAMIC COMPOUNDING]", dyn)):
        lines.append(f"## {header} Summary")
        lines.append(f"| Metric ({header}) | Value |")
        lines.append("|---|---|")
        lines.append(f"| Trades (W/L) | {trk['trades_count']} ({trk['wins']}/{trk['losses']}) |")
        lines.append(f"| Win Rate % | {_fmt_opt_float(trk['win_rate_pct'], 2, '%')} |")
        lines.append(f"| Profit Factor | {_fmt_opt_float(trk['profit_factor'], 4)} |")
        lines.append(f"| Gross (INR) | {trk['gross']} |")
        lines.append(f"| BASE Net (INR) | {trk['net_base']} |")
        lines.append(f"| STRESS Net (INR) | {trk['net_stress']} |")
        lines.append(f"| PROXY Net (INR) | {trk['net_proxy'] if trk['net_proxy'] is not None else 'NA'} |")
        lines.append(f"| Max Drawdown | {trk['max_drawdown_pct']:.2f}% (INR {trk['max_drawdown_inr']}) |")
        lines.append(f"| Start -> Final Balance (BASE) | {trk['start_balance']} -> {trk['final_balance_base']} |")
        lines.append(f"| Liquidity Warnings | {trk['liquidity_warn_count']} |")
        lines.append("")

    atomic_write_bytes(dest, ("\n".join(lines) + "\n").encode("utf-8"))


def write_report_html(
    dest: Path,
    *,
    summary: Mapping[str, Any],
    dual: DualTrackResult,
    config: Mapping[str, Any],
    dataset: DatasetResult | None = None,
) -> None:
    """Write self-contained HTML5 REPORT.html (zero JS, zero CDN, 5 inline SVGs, row cap + Others)."""
    max_rows = int(config.get("report_html_max_trade_rows", 500))
    fixed = summary["fixed"]
    dyn = summary["dynamic"]

    ruin_banner = ""
    if dyn.get("ruin_flag"):
        ruin_banner = (
            '<div style="background:#fee2e2;border:2px solid #dc2626;padding:12px;margin:10px 0;font-weight:bold;">'
            f"[DYNAMIC RUIN] Dynamic track halted on {html.escape(str(dyn.get('ruin_date')))} "
            f"at balance INR {html.escape(str(dyn.get('ruin_balance')))}."
            "</div>"
        )

    def _summary_table(label: str, trk: Mapping[str, Any]) -> str:
        rows = [
            ("Trades (W/L)", f"{trk['trades_count']} ({trk['wins']}/{trk['losses']})"),
            ("Win Rate %", _fmt_opt_float(trk["win_rate_pct"], 2, "%")),
            ("Profit Factor", _fmt_opt_float(trk["profit_factor"], 4)),
            ("Gross (INR)", str(trk["gross"])),
            ("Charges BASE (INR)", str(trk["charges_base"])),
            ("Fill Cost BASE (INR)", str(trk["fill_cost_base"])),
            ("BASE Net (INR)", str(trk["net_base"])),
            ("STRESS Net (INR)", str(trk["net_stress"])),
            ("PROXY Net (INR)", str(trk["net_proxy"]) if trk["net_proxy"] is not None else "NA"),
            ("Max Drawdown", f"{trk['max_drawdown_pct']:.2f}% (INR {trk['max_drawdown_inr']})"),
            ("Balance (Start -> Final BASE)", f"{trk['start_balance']} -> {trk['final_balance_base']}"),
        ]
        tr_html = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows
        )
        return (
            f"<section><h2>{html.escape(label)} Summary</h2>"
            f"<table><thead><tr><th>Metric ({html.escape(label)})</th><th>Value</th></tr></thead>"
            f"<tbody>{tr_html}</tbody></table></section>"
        )

    def _trades_table(label: str, trades: Sequence[TradeRecord]) -> str:
        kept, others = _capped_rows_with_others(trades, max_rows)
        body: list[str] = []
        for t in kept:
            body.append(
                f"<tr><td>{t.trade_no}</td><td>{html.escape(t.date)}</td>"
                f"<td>{html.escape(t.entry_ts)}-{html.escape(t.exit_ts)}</td>"
                f"<td>{t.strike}{html.escape(t.side)}</td><td>{t.lots}</td>"
                f"<td>{t.gross}</td><td>{t.charges_base}</td><td>{t.fill_cost_base}</td>"
                f"<td>{t.net_base}</td><td>{t.net_stress}</td><td>{html.escape(t.exit_reason)}</td></tr>"
            )
        if others is not None:
            body.append(
                f'<tr class="others-row"><td colspan="5">Others ({others["count"]} trades)</td>'
                f"<td>{others['gross']}</td><td>{others['charges_base']}</td><td>{others['fill_cost_base']}</td>"
                f"<td>{others['net_base']}</td><td>{others['net_stress']}</td><td>aggregated</td></tr>"
            )
        return (
            f"<details><summary><strong>{html.escape(label)} Trades ({len(trades)} total)</strong></summary>"
            f"<table><thead><tr><th># ({html.escape(label)})</th><th>Date</th><th>Time</th>"
            f"<th>Opt</th><th>Lots</th><th>Gross</th><th>Charges</th><th>FillCost</th>"
            f"<th>Net BASE</th><th>Net STRESS</th><th>Exit</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></details>"
        )

    tax_html = ""
    if dataset is not None:
        tax_html = (
            "<section><h2>Day Taxonomy</h2>"
            f"<p>Calendar: {len(dataset.calendar_days)} | Market: {len(dataset.market_days)} | "
            f"Excluded: {len(dataset.excluded_days)} | Warm-up: {len(dataset.warmup_days)} | "
            f"Trading: {len(dataset.trading_days)} | Holdout: {len(dataset.holdout_days)}</p></section>"
        )

    svgs = (
        "<section><h2>[FIXED 1-LOT] &amp; [DYNAMIC COMPOUNDING] Visual Charts</h2>"
        "<details open><summary>1. [FIXED 1-LOT] Waterfall &amp; MFE</summary>"
        + _svg_waterfall(dual.fixed.trades)
        + "</details>"
        "<details open><summary>2. [FIXED 1-LOT] &amp; [DYNAMIC COMPOUNDING] Equity Curve</summary>"
        + _svg_equity_curve(dual.fixed, dual.dynamic)
        + "</details>"
        "<details><summary>3. [FIXED 1-LOT] Exit Reason PnL</summary>"
        + _svg_exit_reasons(dual.fixed.trades)
        + "</details>"
        "<details><summary>4. [FIXED 1-LOT] Time x Weekday Heatmap</summary>"
        + _svg_time_weekday_heatmap(dual.fixed.trades)
        + "</details>"
        "<details><summary>5. [FIXED 1-LOT] MAE vs MFE Scatter</summary>"
        + _svg_mae_mfe_scatter(dual.fixed.trades)
        + "</details></section>"
    )

    doc = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>Backtest Report ({ENGINE_TAG})</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:16px auto;padding:0 12px;color:#0f172a}"
        "table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}"
        "th,td{border:1px solid #cbd5e1;padding:4px 6px;text-align:left}"
        "th{background:#f1f5f9}.others-row{font-weight:bold;background:#fef9c3}</style></head><body>"
        f"<h1>Backtest Report ({ENGINE_TAG})</h1>"
        + ruin_banner
        + tax_html
        + _summary_table("[FIXED 1-LOT]", fixed)
        + _summary_table("[DYNAMIC COMPOUNDING]", dyn)
        + svgs
        + _trades_table("[FIXED 1-LOT]", dual.fixed.trades)
        + _trades_table("[DYNAMIC COMPOUNDING]", dual.dynamic.trades)
        + "</body></html>\n"
    )
    atomic_write_bytes(dest, doc.encode("utf-8"))


def generate_reports(
    *,
    output_dir: Path,
    dual: DualTrackResult,
    days: Mapping[str, DayData],
    trading_dates: Sequence[str],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    dataset: DatasetResult | None = None,
    mode: str = "range",
    m_variants: int = 1,
    k_opens: int = 1,
    is_holdout: bool = False,
    weekly_pass: bool = False,
    design_pass: bool = False,
    delay1_net_fixed: Decimal | None = None,
    delay2_net_fixed: Decimal | None = None,
    delay1_net_dynamic: Decimal | None = None,
    delay2_net_dynamic: Decimal | None = None,
) -> dict[str, Any]:
    """Generate all report outputs (CSVs, summary.json, manifest.json, verify_sample.md, REPORT.md/html)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_sum = compute_track_summary(dual.fixed)
    dyn_sum = compute_track_summary(dual.dynamic)
    robustness = compute_robustness_pack(
        dual=dual,
        days=days,
        trading_dates=trading_dates,
        config=config,
        is_holdout=is_holdout,
        delay1_net_fixed=delay1_net_fixed,
        delay2_net_fixed=delay2_net_fixed,
        delay1_net_dynamic=delay1_net_dynamic,
        delay2_net_dynamic=delay2_net_dynamic,
    )

    verdict_info: dict[str, Any] = {}
    holdout_pass = False
    if mode == "weekly":
        verdict_info = evaluate_weekly_verdict(fixed_sum, dyn_sum, config)
    elif mode == "design-gate":
        verdict_info = evaluate_design_gate_verdict(
            fixed_sum, dyn_sum, robustness, m_variants, config
        )
        design_pass = verdict_info["verdict"] == "PASS"
    elif mode == "holdout":
        verdict_info = evaluate_holdout_verdict(
            fixed_sum, dyn_sum, robustness, k_opens, config
        )
        holdout_pass = verdict_info["verdict"] == "PASS"

    paper_ready = evaluate_paper_ready(weekly_pass, design_pass, holdout_pass)

    summary: dict[str, Any] = {
        "engine_tag": ENGINE_TAG,
        "mode": mode,
        "fixed": fixed_sum,
        "dynamic": dyn_sum,
        "robustness": robustness,
        "verdict_detail": verdict_info,
        "verdict": verdict_info.get("outcome", verdict_info.get("verdict", "NA")),
        "paper_ready": paper_ready,
        "paper_min_trades": int(config["paper_min_trades"]),
    }

    write_trades_csv(output_dir / "trades_fixed.csv", dual.fixed.trades)
    write_trades_csv(output_dir / "trades_dynamic.csv", dual.dynamic.trades)
    write_summary_json(output_dir / "summary.json", summary)
    write_manifest_json(output_dir / "manifest.json", manifest)
    write_verify_sample_md(
        output_dir / "verify_sample.md",
        dual.fixed.trades,
        dual.dynamic.trades,
        days,
        int(config["seed"]),
    )
    write_report_md(output_dir / "REPORT.md", summary=summary, dataset=dataset)
    write_report_html(
        output_dir / "REPORT.html",
        summary=summary,
        dual=dual,
        config=config,
        dataset=dataset,
    )
    return summary
