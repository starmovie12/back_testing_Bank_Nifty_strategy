"""Read-only Fixed-track postmortem, 12-class autopsy, BH FDR slices, and AI handoff.

Purpose:
    Runs strictly read-only postmortem analysis after report.py, anchored on
    the [FIXED 1-LOT] edge track plus a [DYNAMIC COMPOUNDING] risk/DD/RUIN
    summary panel. Computes the Decimal money waterfall, entry edge curve
    with BH FDR, 12 first-match autopsy classes, exit counterfactuals (full
    and lite), 15 slice dimensions with chunked permutation tests, 4-panel
    Bharosa, funnel + 1-lot shadow P&L, 6-point milan reconciliation,
    detectors D01-D12, DIAGNOSIS.md, diagnosis.json, capped AI_HANDOFF.md,
    and conditional DIFF.md (never on holdout). Uses RNG stream 13 only.

PRD Sections:
    §3.3, §4.2, §8, §12.2

Niyam Covered:
    - Niyam 12: Deterministic RNG stream SeedSequence([seed, 13]) & holdout guard

Guarding Tests:
    T16, T45, T46, T47, T48, T49, T50, T51, T52, T60

PUBLIC API:
    Constants:
        AUTOPSY_CLASSES, SEVERITY_RANK, HOLDOUT_ALLOWED_DETECTORS
    Functions:
        benjamini_hochberg, compute_waterfall, compute_entry_edge,
        classify_trade_autopsy, compute_autopsy, compute_counterfactuals,
        compute_slices, compute_bharosa, compute_funnel, compute_shadow_pnl,
        check_milan, run_detectors, render_ai_handoff, write_diff_md,
        run_diagnosis
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from contracts import (
    ENGINE_TAG,
    SHADOW_SKIP_CODES,
    SkipCode,
    SkippedSignal,
    TradeRecord,
    format_decimal,
)
from costs import ZERO, compute_trade_accounting, round_paise, to_decimal
from data_loader import DayData, atomic_write_bytes, hhmm_to_minute
from engine import DualTrackResult, TrackRunResult, ceil_tick, price_to_decimal

AUTOPSY_CLASSES: tuple[str, ...] = (
    "GAP_BREACH",
    "COST_KILL",
    "SHAKEOUT",
    "GAVE_BACK",
    "NEVER_WORKED",
    "TIME_DRIFT",
    "OTHER_LOSS",
    "TARGET_EARLY",
    "TARGET_CLEAN",
    "PROTECTED_WIN",
    "CLOCK_WIN",
    "OTHER_WIN",
)

SEVERITY_RANK: Mapping[str, int] = {"LAL": 0, "PEELA": 1, "HARA": 2, "INFO": 3}
HOLDOUT_ALLOWED_DETECTORS: frozenset[str] = frozenset({"D01", "D02", "D09", "D11", "D12"})

_CANCEL_SKIPS: frozenset[SkipCode] = frozenset(
    {SkipCode.SKIP_NO_PRICE, SkipCode.SKIP_STALE_CANCEL, SkipCode.SKIP_CUTOFF}
)
_RISK_SKIPS: frozenset[SkipCode] = frozenset(
    {
        SkipCode.SKIP_NO_BALANCE,
        SkipCode.SKIP_SIZE,
        SkipCode.SKIP_DAYCAP,
        SkipCode.SKIP_RISK,
        SkipCode.SKIP_BAD_STOP,
    }
)


# ------------------------------------------------------------------------------
# Statistical & JSON Helpers (AUD-016, AUD-017)
# ------------------------------------------------------------------------------
def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Compute Benjamini-Hochberg adjusted q-values preserving input order (§12.2, T50)."""
    k = len(p_values)
    if k == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda x: x[1])
    q_sorted = [0.0] * k
    running_min = 1.0
    for rank_rev in range(k, 0, -1):
        _, p_val = indexed[rank_rev - 1]
        raw_q = (float(k) / float(rank_rev)) * p_val
        running_min = min(running_min, raw_q)
        q_sorted[rank_rev - 1] = min(1.0, max(0.0, running_min))
    out = [0.0] * k
    for pos, (orig_idx, _) in enumerate(indexed):
        out[orig_idx] = q_sorted[pos]
    return out


def _json_clean(obj: Any) -> Any:
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
# 1. Decimal Money Waterfall (§12.2)
# ------------------------------------------------------------------------------
def compute_waterfall(trades: Sequence[TradeRecord]) -> dict[str, Any]:
    """Compute exact Decimal money waterfall A/B/C/D/M/G/capture/N/cost_share."""
    a_gross = round_paise(sum((t.gross for t in trades), ZERO))
    b_fill = round_paise(sum((t.fill_cost_base for t in trades), ZERO))
    c_chg = round_paise(sum((t.charges_base for t in trades), ZERO))
    d_net = round_paise(a_gross - b_fill - c_chg)

    m_mfe = round_paise(sum((t.mfe * Decimal(t.qty) for t in trades), ZERO))
    g_giveback = round_paise(m_mfe - a_gross)
    n_mae = round_paise(sum((t.mae * Decimal(t.qty) for t in trades), ZERO))

    capture: float | None = None
    if m_mfe > ZERO:
        cap_v = float(a_gross / m_mfe)
        capture = round(cap_v, 4) if math.isfinite(cap_v) else None

    cost_share: float | None = None
    if a_gross > ZERO:
        cs_v = float((b_fill + c_chg) / a_gross)
        cost_share = round(cs_v, 4) if math.isfinite(cs_v) else None

    net_stress = round_paise(sum((t.net_stress for t in trades), ZERO))
    proxy_vals = [t.net_proxy for t in trades if t.net_proxy is not None]
    net_proxy = round_paise(sum(proxy_vals, ZERO)) if (trades and len(proxy_vals) == len(trades)) else None

    return {
        "gross_A": format_decimal(a_gross, 2),
        "fill_cost_B": format_decimal(b_fill, 2),
        "charges_C": format_decimal(c_chg, 2),
        "net_base_D": format_decimal(d_net, 2),
        "net_stress": format_decimal(net_stress, 2),
        "net_proxy": format_decimal(net_proxy, 2) if net_proxy is not None else None,
        "mfe_total_M": format_decimal(m_mfe, 2),
        "giveback_G": format_decimal(g_giveback, 2),
        "capture": capture,
        "mae_total_N": format_decimal(n_mae, 2),
        "cost_share": cost_share,
    }


# ------------------------------------------------------------------------------
# 2. Entry Edge Curve with BH FDR (RNG stream 13; §12.2)
# ------------------------------------------------------------------------------
def compute_entry_edge(
    trades: Sequence[TradeRecord],
    days: Mapping[str, DayData],
    config: Mapping[str, Any],
    *,
    rng: np.random.Generator,
    lite: bool = False,
    is_holdout: bool = False,
) -> dict[str, Any]:
    """Evaluate forward return fwd(H) vs seeded random baseline across diag_horizons_min."""
    horizons = [int(h) for h in config["diag_horizons_min"]]
    n_rand = int(config["diag_lite_random_runs"] if lite else config["diag_random_runs"])
    fdr_q = float(config["diag_fdr_q"])
    start_min = hhmm_to_minute(str(config["entry_window_start"]))

    cost_units = [
        float((t.charges_base + t.fill_cost_base) / Decimal(t.qty))
        for t in trades
        if t.qty > 0
    ]
    mean_cost_unit = float(np.mean(cost_units)) if cost_units else 0.0

    rows: list[dict[str, Any]] = []
    raw_ps: list[float] = []
    tested_indices: list[int] = []

    for idx_h, h_min in enumerate(horizons):
        actual_fwds: list[float] = []
        cand_pools: list[np.ndarray] = []
        excluded = 0

        for tr in trades:
            day = days.get(tr.date)
            if day is None or (tr.strike, tr.side) not in day.options:
                excluded += 1
                continue
            ser = day.options[(tr.strike, tr.side)]
            ff_key = "expiry_force_flat_time" if day.is_expiry_day else "force_flat_time"
            ff_min = min(hhmm_to_minute(str(config[ff_key])), day.grid_len - 1)
            eb = hhmm_to_minute(tr.entry_ts)
            target_b = eb + h_min - 1
            if target_b > ff_min or target_b >= day.grid_len:
                excluded += 1
                continue
            # Last live bar <= target_b
            live_c: float | None = None
            for b in range(target_b, eb - 1, -1):
                if bool(ser.live[b]) and math.isfinite(float(ser.close[b])):
                    live_c = float(ser.close[b])
                    break
            if live_c is None:
                excluded += 1
                continue
            actual_fwds.append(live_c - float(tr.entry_price))

            # Random candidate pool on same day/strike/side
            cutoff_key = "expiry_last_entry_time" if day.is_expiry_day else "last_entry_time"
            cutoff_m = min(hhmm_to_minute(str(config[cutoff_key])), ff_min - h_min + 1)
            pool_vals: list[float] = []
            for m in range(start_min, max(start_min, cutoff_m)):
                tb = m + h_min - 1
                if tb < day.grid_len and bool(ser.live[m]) and bool(ser.live[tb]):
                    pool_vals.append(float(ser.close[tb]) - float(ser.open[m]))
            cand_pools.append(
                np.asarray(pool_vals, dtype=np.float64)
                if pool_vals
                else np.asarray([live_c - float(tr.entry_price)], dtype=np.float64)
            )

        if not actual_fwds:
            rows.append(
                {
                    "horizon_min": h_min,
                    "n": 0,
                    "excluded": excluded,
                    "fwd_mean": None,
                    "rand_p95": None,
                    "p_value": None,
                    "q_value": None,
                    "edge_stat": False,
                    "edge_net": False,
                }
            )
            continue

        act_mean = float(np.mean(actual_fwds))
        sim_means = np.empty(n_rand, dtype=np.float64)
        for r_i in range(n_rand):
            s = 0.0
            for pool in cand_pools:
                s += float(pool[int(rng.integers(0, pool.size))])
            sim_means[r_i] = s / float(len(cand_pools))

        p_val = (1.0 + int(np.sum(sim_means >= act_mean))) / float(n_rand + 1)
        rand_p95 = float(np.quantile(sim_means, 0.95))
        raw_ps.append(p_val)
        tested_indices.append(idx_h)
        rows.append(
            {
                "horizon_min": h_min,
                "n": len(actual_fwds),
                "excluded": excluded,
                "fwd_mean": round(act_mean, 4),
                "rand_p95": round(rand_p95, 4),
                "p_value": round(p_val, 6),
                "q_value": None,
                "edge_stat": False,
                "edge_net": False,
            }
        )

    q_vals = benjamini_hochberg(raw_ps)
    for pos, row_idx in enumerate(tested_indices):
        q = q_vals[pos]
        row = rows[row_idx]
        row["q_value"] = round(q, 6)
        row["edge_stat"] = bool(q <= fdr_q)
        row["edge_net"] = bool(
            row["edge_stat"] and row["fwd_mean"] is not None and float(row["fwd_mean"]) > mean_cost_unit
        )

    any_net = any(r["edge_net"] for r in rows)
    any_stat = any(r["edge_stat"] for r in rows)
    verdict = "HAAN" if any_net else ("KAMZOR" if any_stat else "NAHI")

    best_h: int | None = None
    if not is_holdout:
        stat_rows = [r for r in rows if r["edge_stat"] and r["fwd_mean"] is not None]
        if stat_rows:
            best_row = max(stat_rows, key=lambda r: float(r["fwd_mean"]) - mean_cost_unit)
            best_h = int(best_row["horizon_min"])

    return {
        "verdict": verdict,
        "best_h_star": best_h,
        "mean_cost_unit": round(mean_cost_unit, 4),
        "horizons": rows,
        "label": "in-sample",
    }


# ------------------------------------------------------------------------------
# 3. 12-Class Trade Autopsy + 5 Non-Exclusive Flags (§12.2, T48)
# ------------------------------------------------------------------------------
def _post_exit_max_high(
    tr: TradeRecord,
    days: Mapping[str, DayData],
    window_min: int,
) -> tuple[Decimal | None, bool]:
    """Scan up to window_min minutes after exit_ts for max live high. Returns (max_high, unknown_post)."""
    day = days.get(tr.date)
    if day is None or (tr.strike, tr.side) not in day.options:
        return None, True
    ser = day.options[(tr.strike, tr.side)]
    xb = hhmm_to_minute(tr.exit_ts)
    end_b = min(day.grid_len - 1, xb + window_min)
    best: float | None = None
    for b in range(xb + 1, end_b + 1):
        if bool(ser.live[b]) and math.isfinite(float(ser.high[b])):
            hv = float(ser.high[b])
            if best is None or hv > best:
                best = hv
    if best is None:
        return None, True
    return price_to_decimal(best), False


def classify_trade_autopsy(
    tr: TradeRecord,
    days: Mapping[str, DayData],
    config: Mapping[str, Any],
) -> tuple[str, list[str], bool]:
    """Classify one trade into exactly one of 12 first-match autopsy classes + flags (§12.2, T48)."""
    thresh = config["diag_thresholds"]
    rec_min = int(config["diag_recovery_min"])
    fol_min = int(config["diag_target_follow_min"])
    gaveback_r = float(thresh["gaveback_mfe_r"])
    target_fol_r = float(thresh["target_early_follow_r"])
    lot_cap = float(thresh["left_on_table_capture"])

    r_unit = tr.entry_price - tr.initial_stop
    mfe_r = float(tr.mfe / r_unit) if r_unit > ZERO else 0.0
    unknown_post = False

    # First-match 12-class decision tree (§12.2)
    if tr.net_base <= ZERO:
        if tr.exit_price < tr.initial_stop:
            cls = "GAP_BREACH"
        elif tr.gross > ZERO:
            cls = "COST_KILL"
        elif tr.exit_reason in ("stop", "trail", "breakeven"):
            post_hi, unk = _post_exit_max_high(tr, days, rec_min)
            unknown_post = unk
            if post_hi is not None and post_hi >= tr.entry_price:
                cls = "SHAKEOUT"
            elif mfe_r >= gaveback_r:
                cls = "GAVE_BACK"
            else:
                cls = "NEVER_WORKED"
        elif mfe_r >= gaveback_r:
            cls = "GAVE_BACK"
        elif tr.exit_reason in ("time", "scratch", "force_flat", "strategy_exit"):
            cls = "TIME_DRIFT"
        else:
            cls = "OTHER_LOSS"
    else:
        if tr.exit_reason == "target":
            post_hi, unk = _post_exit_max_high(tr, days, fol_min)
            unknown_post = unk
            follow_lvl = tr.exit_price + (Decimal(f"{target_fol_r:.6f}") * r_unit)
            if post_hi is not None and post_hi >= follow_lvl:
                cls = "TARGET_EARLY"
            else:
                cls = "TARGET_CLEAN"
        elif tr.exit_reason in ("trail", "breakeven", "strategy_exit"):
            cls = "PROTECTED_WIN"
        elif tr.exit_reason in ("time", "scratch", "force_flat"):
            cls = "CLOCK_WIN"
        else:
            cls = "OTHER_WIN"

    # Non-exclusive flags
    flags: list[str] = []
    if tr.net_base > ZERO and tr.mfe > ZERO:
        gross_unit = tr.exit_price - tr.entry_price
        if float(gross_unit / tr.mfe) < lot_cap:
            flags.append("LEFT_ON_TABLE")
    if tr.exit_reason in ("stop", "trail", "breakeven") and tr.hold_min <= 3:
        flags.append("FAST_STOP")
    if tr.dte == 0:
        flags.append("EXPIRY_DAY")
    if tr.gap_exit:
        flags.append("GAP_EXIT")
    if tr.exit_reason == "no_bar":
        flags.append("NO_BAR")

    return cls, flags, unknown_post


def compute_autopsy(
    trades: Sequence[TradeRecord],
    days: Mapping[str, DayData],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run 12-class autopsy across all trades and aggregate exact counts & INR totals."""
    by_class: dict[str, dict[str, Any]] = {
        c: {"count": 0, "net_base": ZERO} for c in AUTOPSY_CLASSES
    }
    flag_counts: dict[str, int] = {}
    per_trade: list[dict[str, Any]] = []
    unknown_post_count = 0

    for tr in trades:
        cls, flags, unk = classify_trade_autopsy(tr, days, config)
        if unk:
            unknown_post_count += 1
        by_class[cls]["count"] += 1
        by_class[cls]["net_base"] = round_paise(by_class[cls]["net_base"] + tr.net_base)
        for fl in flags:
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
        per_trade.append(
            {
                "trade_no": tr.trade_no,
                "date": tr.date,
                "class": cls,
                "flags": flags,
                "net_base": format_decimal(tr.net_base, 2),
            }
        )

    summary_classes = {
        c: {"count": by_class[c]["count"], "net_base": format_decimal(by_class[c]["net_base"], 2)}
        for c in AUTOPSY_CLASSES
    }
    t_n = len(trades)
    max_other = float(config["diag_grid"]["other_class_max_share"])
    other_loss_flag = (t_n > 0) and (by_class["OTHER_LOSS"]["count"] / float(t_n) > max_other)
    other_win_flag = (t_n > 0) and (by_class["OTHER_WIN"]["count"] / float(t_n) > max_other)
    return {
        "classes": summary_classes,
        "flag_counts": flag_counts,
        "unknown_post": unknown_post_count,
        "other_loss_flag": other_loss_flag,
        "other_win_flag": other_win_flag,
        "per_trade": per_trade,
    }


# ------------------------------------------------------------------------------
# 4. Exit Counterfactuals (Full & Lite; §12.2)
# ------------------------------------------------------------------------------
def _replay_cf_trade(
    tr: TradeRecord,
    day: DayData | None,
    config: Mapping[str, Any],
    *,
    hold_min: int | None = None,
    no_stop: bool = False,
    stop_mult: float | None = None,
    target_r: float | None = None,
    be_r: float | None = None,
    trail_r: float | None = None,
) -> Decimal:
    """Replay one trade's exit under counterfactual rules using §11.4 + costs.py BASE."""
    if day is None or (tr.strike, tr.side) not in day.options:
        return tr.net_base
    ser = day.options[(tr.strike, tr.side)]
    eb = hhmm_to_minute(tr.entry_ts)
    ff_key = "expiry_force_flat_time" if day.is_expiry_day else "force_flat_time"
    ff_min = min(hhmm_to_minute(str(config[ff_key])), day.grid_len - 1)
    tick_size = to_decimal(config["tick_size"])

    fp = tr.entry_price
    orig_r = fp - tr.initial_stop
    if no_stop:
        cur_stop: Decimal | None = None
    elif stop_mult is not None:
        cur_stop = ceil_tick(fp - (orig_r * Decimal(f"{stop_mult:.6f}")), tick_size)
        if cur_stop <= ZERO:
            cur_stop = tick_size
    else:
        cur_stop = tr.initial_stop

    tgt = (
        ceil_tick(fp + (orig_r * Decimal(f"{target_r:.6f}")), tick_size)
        if target_r is not None
        else tr.target_price
    )

    highest = fp
    last_live_c = fp
    last_live_l = fp
    exit_bar = ff_min
    exit_price = fp
    is_no_bar = False

    for b in range(eb, ff_min + 1):
        live_b = bool(ser.live[b])
        time_due = (hold_min is not None and (b - eb) >= hold_min) or (b >= ff_min)

        if not live_b:
            if time_due and b == ff_min:
                exit_bar = b
                exit_price = min(last_live_l, last_live_c)
                is_no_bar = True
                break
            continue

        bo = price_to_decimal(float(ser.open[b]))
        bh = price_to_decimal(float(ser.high[b]))
        bl = price_to_decimal(float(ser.low[b]))
        bc = price_to_decimal(float(ser.close[b]))

        # 2A Open
        if b > eb and cur_stop is not None and bo <= cur_stop:
            exit_bar, exit_price = b, bo
            break
        if time_due:
            exit_bar, exit_price = b, bo
            break

        # 2B Intrabar (stop-first)
        if cur_stop is not None and bl <= cur_stop:
            exit_bar, exit_price = b, cur_stop
            break
        if tgt is not None and bh >= tgt:
            exit_bar, exit_price = b, tgt
            break

        # 2C Post-bar updates
        if bh > highest:
            highest = bh
        last_live_l = bl
        last_live_c = bc

        if be_r is not None and cur_stop is not None:
            if bh >= fp + (orig_r * Decimal(f"{be_r:.6f}")) and fp > cur_stop:
                cur_stop = fp
        if trail_r is not None and cur_stop is not None:
            tr_dec = Decimal(f"{trail_r:.6f}")
            if (highest - fp) >= (orig_r * tr_dec):
                cand_s = ceil_tick(highest - (orig_r * tr_dec), tick_size)
                if cand_s > cur_stop:
                    cur_stop = cand_s

    bd = compute_trade_accounting(
        fp, exit_price, tr.qty, eb, exit_bar, day.is_expiry_day, tr.date, config,
        scenario="BASE", is_no_bar=is_no_bar,
    )
    assert bd is not None
    return bd.net


def compute_counterfactuals(
    trades: Sequence[TradeRecord],
    days: Mapping[str, DayData],
    config: Mapping[str, Any],
    *,
    h_star: int | None = None,
    lite: bool = False,
) -> dict[str, Any]:
    """Compute exit counterfactuals (always labeled 'in-sample estimate'; §12.2)."""
    grid = config["diag_grid"]
    actual_net = round_paise(sum((t.net_base for t in trades), ZERO))
    results: dict[str, dict[str, str]] = {
        "CF-ACTUAL": {
            "net_base": format_decimal(actual_net, 2),
            "delta_inr": "0.00",
        }
    }

    def _run_cf(name: str, **kwargs: Any) -> None:
        tot = round_paise(
            sum((_replay_cf_trade(t, days.get(t.date), config, **kwargs) for t in trades), ZERO)
        )
        results[name] = {
            "net_base": format_decimal(tot, 2),
            "delta_inr": format_decimal(round_paise(tot - actual_net), 2),
        }

    if lite:
        if h_star is not None:
            _run_cf(f"CF-HOLD({h_star})", hold_min=h_star)
        _run_cf("CF-STOPx1.5", stop_mult=1.5)
        _run_cf("CF-TARGETx2", target_r=2.0)
        _run_cf("CF-BE(1.0)", be_r=1.0)
    else:
        for h in grid["cf_hold_min"]:
            _run_cf(f"CF-HOLD({h})", hold_min=int(h))
        if h_star is not None:
            _run_cf(f"CF-NOSTOP-HOLD({h_star})", hold_min=h_star, no_stop=True)
        for f in grid["cf_stop_mult"]:
            _run_cf(f"CF-STOPx{f}", stop_mult=float(f))
        for g in grid["cf_target_r"]:
            _run_cf(f"CF-TARGETx{g}", target_r=float(g))
        for r in grid["cf_be_r"]:
            _run_cf(f"CF-BE({r})", be_r=float(r))
        _run_cf("CF-TRAIL-1R", trail_r=float(grid["cf_trail_r"]))

    return {"label": "in-sample estimate", "scenarios": results}


# ------------------------------------------------------------------------------
# 5. 15 Slice Dimensions & Chunked BH FDR Permutation Tests (§12.2, T50, T60)
# ------------------------------------------------------------------------------
def _bucket_trade_of_day(idx_1based: int, labels: Sequence[str], edges: Sequence[int]) -> str:
    for i, edge in enumerate(edges):
        if idx_1based <= int(edge):
            return str(labels[i])
    return str(labels[-1])


def _compute_day_labels(days: Mapping[str, DayData], config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Compute diagnosis-only day_type and gap_open labels (NEVER exposed to strategy; T50)."""
    thresh = config["diag_thresholds"]
    grid = config["diag_grid"]
    trend_eff = float(thresh["trend_eff"])
    range_eff = float(thresh["range_eff"])
    gap_pct_th = float(grid["gap_open_pct"])

    labels: dict[str, dict[str, str]] = {}
    prev_close: float | None = None
    for d_str in sorted(days.keys()):
        day = days[d_str]
        live_idx = np.flatnonzero(day.index.live)
        if live_idx.size == 0:
            labels[d_str] = {"day_type": "mixed", "gap_open": "flat"}
            continue
        day_o = float(day.index.open[int(live_idx[0])])
        day_c = float(day.index.close[int(live_idx[-1])])
        day_h = float(np.nanmax(day.index.high[live_idx]))
        day_l = float(np.nanmin(day.index.low[live_idx]))
        rng_hl = day_h - day_l
        eff = abs(day_c - day_o) / rng_hl if rng_hl > 0.0 else 0.0
        if eff >= trend_eff:
            dtype = "trend_up" if day_c >= day_o else "trend_down"
        elif eff <= range_eff:
            dtype = "range"
        else:
            dtype = "mixed"

        if prev_close is not None and prev_close > 0.0:
            gp = ((day_o - prev_close) / prev_close) * 100.0
            gopen = ">=+0.5%" if gp >= gap_pct_th else ("<=-0.5%" if gp <= -gap_pct_th else "flat")
        else:
            gopen = "flat"
        prev_close = day_c
        labels[d_str] = {"day_type": dtype, "gap_open": gopen}
    return labels


def compute_slices(
    trades: Sequence[TradeRecord],
    days: Mapping[str, DayData],
    config: Mapping[str, Any],
    *,
    rng: np.random.Generator,
    lite: bool = False,
) -> dict[str, Any]:
    """Compute all 15 slice dimensions with chunked permutation tests & joint BH FDR (§12.2)."""
    grid = config["diag_grid"]
    min_n = int(config["diag_min_n"])
    fdr_q = float(config["diag_fdr_q"])
    n_perm = int(config["diag_lite_perm_runs"] if lite else config["diag_perm_runs"])
    batch_sz = max(1, int(config["diag_perm_batch"]))

    tod_labels = list(grid["trade_of_day_buckets"])
    tod_edges = [int(x) for x in grid["trade_of_day_edges"]]
    prem_edges = [float(x) for x in grid["premium_buckets"]]
    hold_edges = [int(x) for x in grid["hold_buckets"]]
    atm_steps = int(grid["moneyness_atm_steps"])
    t_bkt_min = int(grid["entry_time_bucket_min"])

    day_labels = _compute_day_labels(days, config)

    # Precompute VIX tertiles across trades that have live VIX
    vix_vals: list[float | None] = []
    for tr in trades:
        day = days.get(tr.date)
        v_val: float | None = None
        if day is not None and day.vix is not None:
            eb = max(0, hhmm_to_minute(tr.entry_ts) - 1)
            if eb < len(day.vix) and bool(day.vix.live[eb]) and math.isfinite(float(day.vix.close[eb])):
                v_val = float(day.vix.close[eb])
        vix_vals.append(v_val)
    fin_vix = np.asarray([v for v in vix_vals if v is not None], dtype=np.float64)
    vix_q33 = float(np.quantile(fin_vix, 1.0 / 3.0)) if fin_vix.size >= 3 else None
    vix_q66 = float(np.quantile(fin_vix, 2.0 / 3.0)) if fin_vix.size >= 3 else None

    # Assign each trade to a single cell in each of the 15 dimensions
    dims: dict[str, dict[str, list[int]]] = {
        name: {}
        for name in (
            "entry_time_30m",
            "weekday",
            "dte",
            "side",
            "moneyness",
            "premium",
            "vix",
            "gap_open",
            "day_type",
            "exit_reason",
            "hold_bucket",
            "trade_of_day",
            "after_loss",
            "month",
            "tag",
        )
    }

    day_trade_counter: dict[str, int] = {}
    day_prev_outcome: dict[str, str] = {}

    for idx, tr in enumerate(trades):
        # 1. entry_time_30m
        em = hhmm_to_minute(tr.entry_ts)
        b_idx = max(0, em // t_bkt_min)
        dims["entry_time_30m"].setdefault(f"B{b_idx:02d}", []).append(idx)

        # 2. weekday
        dims["weekday"].setdefault(tr.weekday, []).append(idx)

        # 3. dte
        if tr.dte == 0:
            d_bkt = "0"
        elif tr.dte == 1:
            d_bkt = "1"
        elif tr.dte == 2:
            d_bkt = "2"
        elif tr.dte <= 5:
            d_bkt = "3-5"
        else:
            d_bkt = "6+"
        dims["dte"].setdefault(d_bkt, []).append(idx)

        # 4. side
        dims["side"].setdefault(tr.side, []).append(idx)

        # 5. moneyness vs t-1 index close
        day = days.get(tr.date)
        spot = day.ref_spot if day is not None else float(tr.strike)
        if day is not None:
            sig_m = max(0, hhmm_to_minute(tr.signal_ts) - 1)
            if sig_m < len(day.index) and bool(day.index.live[sig_m]):
                spot = float(day.index.close[sig_m])
        diff_steps = int(round((tr.strike - spot) / 100.0))
        otm_steps = diff_steps if tr.side == "CE" else -diff_steps
        if abs(otm_steps) <= atm_steps:
            mon = "ATM"
        elif otm_steps < -atm_steps:
            mon = "ITM"
        elif otm_steps <= atm_steps + 2:
            mon = "OTM1-2"
        else:
            mon = "OTM3+"
        dims["moneyness"].setdefault(mon, []).append(idx)

        # 6. premium
        ep_f = float(tr.entry_price)
        if ep_f < prem_edges[0]:
            p_bkt = f"<{int(prem_edges[0])}"
        elif ep_f < prem_edges[1]:
            p_bkt = f"{int(prem_edges[0])}-{int(prem_edges[1])}"
        elif ep_f < prem_edges[2]:
            p_bkt = f"{int(prem_edges[1])}-{int(prem_edges[2])}"
        else:
            p_bkt = f">={int(prem_edges[2])}"
        dims["premium"].setdefault(p_bkt, []).append(idx)

        # 7. vix
        vv = vix_vals[idx]
        if vv is None or vix_q33 is None or vix_q66 is None:
            v_bkt = "NA"
        elif vv <= vix_q33:
            v_bkt = "low"
        elif vv <= vix_q66:
            v_bkt = "mid"
        else:
            v_bkt = "high"
        dims["vix"].setdefault(v_bkt, []).append(idx)

        # 8 & 9. gap_open & day_type
        dlbl = day_labels.get(tr.date, {"day_type": "mixed", "gap_open": "flat"})
        dims["gap_open"].setdefault(dlbl["gap_open"], []).append(idx)
        dims["day_type"].setdefault(dlbl["day_type"], []).append(idx)

        # 10. exit_reason
        dims["exit_reason"].setdefault(tr.exit_reason, []).append(idx)

        # 11. hold_bucket
        hm = tr.hold_min
        if hm <= hold_edges[0]:
            h_bkt = f"<={hold_edges[0]}"
        elif hm <= hold_edges[1]:
            h_bkt = f"{hold_edges[0]+1}-{hold_edges[1]}"
        elif hm <= hold_edges[2]:
            h_bkt = f"{hold_edges[1]+1}-{hold_edges[2]}"
        else:
            h_bkt = f"{hold_edges[2]+1}+"
        dims["hold_bucket"].setdefault(h_bkt, []).append(idx)

        # 12. trade_of_day (config-driven buckets for 50+ trades/day; AUD-004, T60)
        tod_n = day_trade_counter.get(tr.date, 0) + 1
        day_trade_counter[tr.date] = tod_n
        dims["trade_of_day"].setdefault(_bucket_trade_of_day(tod_n, tod_labels, tod_edges), []).append(idx)

        # 13. after_loss
        prev_out = day_prev_outcome.get(tr.date)
        al_bkt = "first_of_day" if prev_out is None else ("after_loss" if prev_out == "LOSS" else "after_win")
        day_prev_outcome[tr.date] = tr.outcome
        dims["after_loss"].setdefault(al_bkt, []).append(idx)

        # 14. month
        dims["month"].setdefault(tr.date[:7], []).append(idx)

        # 15. tag
        tag_lbl = tr.tags if tr.tags else "untagged"
        dims["tag"].setdefault(tag_lbl, []).append(idx)

    # Chunked permutation test across all cells with n >= diag_min_n (AUD-011)
    nets_arr = np.asarray([float(t.net_base) for t in trades], dtype=np.float64)
    total_n = nets_arr.size
    total_sum = float(np.sum(nets_arr)) if total_n > 0 else 0.0

    out_dims: dict[str, dict[str, dict[str, Any]]] = {}
    tested_keys: list[tuple[str, str]] = []
    tested_ps: list[float] = []

    for dname, cell_map in dims.items():
        out_dims[dname] = {}
        for cname, idxs in sorted(cell_map.items()):
            cnt = len(idxs)
            cell_sum_dec = round_paise(sum((trades[i].net_base for i in idxs), ZERO))
            cell_mean = float(np.mean(nets_arr[idxs])) if cnt > 0 else 0.0
            entry: dict[str, Any] = {
                "n": cnt,
                "net_base": format_decimal(cell_sum_dec, 2),
                "mean_net": round(cell_mean, 4),
                "p_value": None,
                "q_value": None,
                "status": "low_n" if cnt < min_n else "chance ho sakta hai",
            }
            if cnt >= min_n and total_n > cnt and n_perm > 0:
                rest_mean = (total_sum - float(np.sum(nets_arr[idxs]))) / float(total_n - cnt)
                obs_stat = abs(cell_mean - rest_mean)
                ge_cnt = 0
                done = 0
                while done < n_perm:
                    cur_b = min(batch_sz, n_perm - done)
                    for _ in range(cur_b):
                        perm_idx = rng.choice(total_n, size=cnt, replace=False)
                        p_sum = float(np.sum(nets_arr[perm_idx]))
                        p_cell_m = p_sum / float(cnt)
                        p_rest_m = (total_sum - p_sum) / float(total_n - cnt)
                        if abs(p_cell_m - p_rest_m) >= obs_stat:
                            ge_cnt += 1
                    done += cur_b
                p_val = (1.0 + ge_cnt) / float(n_perm + 1)
                entry["p_value"] = round(p_val, 6)
                tested_keys.append((dname, cname))
                tested_ps.append(p_val)
            out_dims[dname][cname] = entry

    q_all = benjamini_hochberg(tested_ps)
    for pos, (dname, cname) in enumerate(tested_keys):
        qv = q_all[pos]
        out_dims[dname][cname]["q_value"] = round(qv, 6)
        out_dims[dname][cname]["status"] = "PAKKA" if qv <= fdr_q else "chance ho sakta hai"

    return {"dimensions": out_dims, "tested_cells": len(tested_keys)}


# ------------------------------------------------------------------------------
# 6. 4-Panel Bharosa, Funnel, Shadow P&L & 6-Point Milan (§12.2, T46, T49)
# ------------------------------------------------------------------------------
def compute_bharosa(
    fixed_trades: Sequence[TradeRecord],
    summary: Mapping[str, Any],
    m_variants: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute 4-panel Bharosa (Sample, Concentration, Cost/Delay, Tweak Count m)."""
    thresh = config["diag_thresholds"]
    min_t = int(config["min_trades"])
    conc_k = int(thresh["conc_top_k"])
    conc_share = float(thresh["conc_share"])
    m_warn = int(thresh["m_warn"])
    t_cnt = len(fixed_trades)

    # Panel 1: Sample
    p_sample = "HARA" if t_cnt >= min_t else ("PEELA" if t_cnt >= (min_t / 2.0) else "LAL")

    # Panel 2: Concentration
    sorted_nets = sorted((t.net_base for t in fixed_trades), reverse=True)
    net_ex_top_k = round_paise(sum(sorted_nets[conc_k:], ZERO)) if len(sorted_nets) > conc_k else ZERO
    wins = [t.net_base for t in fixed_trades if t.net_base > ZERO]
    wins_sorted = sorted(wins, reverse=True)
    top_frac = float(config["diag_grid"]["top_pct"]) / 100.0
    top5_n = max(1, int(math.ceil(top_frac * len(wins_sorted)))) if wins_sorted else 0
    tot_win = sum(wins_sorted, ZERO)
    top5_share = float(sum(wins_sorted[:top5_n], ZERO) / tot_win) if tot_win > ZERO else 1.0
    c1_ok = net_ex_top_k > ZERO
    c2_ok = top5_share <= conc_share
    p_conc = "HARA" if (c1_ok and c2_ok) else ("PEELA" if (c1_ok or c2_ok) else "LAL")

    # Panel 3: Cost/Delay
    stress_ok = to_decimal(summary["fixed"]["net_stress"]) > ZERO
    d1_raw = summary.get("robustness", {}).get("delay1_net_base_fixed")
    d1_ok = True if d1_raw is None else (to_decimal(d1_raw) > ZERO)
    p_cost = "HARA" if (stress_ok and d1_ok) else ("PEELA" if (stress_ok or d1_ok) else "LAL")

    # Panel 4: Tweak Count m
    p_m = "HARA" if m_variants <= (m_warn - 1) else ("PEELA" if m_variants == m_warn else "LAL")

    panels = {"sample": p_sample, "concentration": p_conc, "cost_delay": p_cost, "tweak_m": p_m}
    overall = min(panels.values(), key=lambda c: SEVERITY_RANK[c])
    return {
        "overall": overall,
        "panels": panels,
        "net_ex_top_k": format_decimal(net_ex_top_k, 2),
        "top5_win_share": round(top5_share, 4),
    }


def compute_funnel(fixed_res: TrackRunResult) -> dict[str, Any]:
    """Compute exact §12.2 funnel counts A = B + C and C = c1 + c2 + c3 + c4."""
    a_calls = int(fixed_res.decide_calls)
    b_notrade = int(sum(fixed_res.notrade_counts.values()))
    c1_invalid = sum(1 for s in fixed_res.skipped if s.skip_code == SkipCode.SKIP_INVALID_INTENT)
    c2_cancel = sum(1 for s in fixed_res.skipped if s.skip_code in _CANCEL_SKIPS)
    c3_risk = sum(1 for s in fixed_res.skipped if s.skip_code in _RISK_SKIPS)
    c4_fills = len(fixed_res.trades)
    c_intents = c1_invalid + c2_cancel + c3_risk + c4_fills

    by_skip = {code.value: 0 for code in SkipCode}
    for s in fixed_res.skipped:
        by_skip[s.skip_code.value] += 1

    exit_counts: dict[str, int] = {}
    for tr in fixed_res.trades:
        exit_counts[tr.exit_reason] = exit_counts.get(tr.exit_reason, 0) + 1

    return {
        "decide_calls_A": a_calls,
        "notrade_B": b_notrade,
        "intents_C": c_intents,
        "c1_invalid_intent": c1_invalid,
        "c2_cancel": c2_cancel,
        "c3_risk_skip": c3_risk,
        "c4_fills": c4_fills,
        "by_skip_code": by_skip,
        "exit_reasons": exit_counts,
        "notrade_top5": sorted(fixed_res.notrade_counts.items(), key=lambda x: (-x[1], x[0]))[:5],
    }


def compute_shadow_pnl(
    day_results_or_skipped: Sequence[Any],
    days: Mapping[str, DayData],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay SHADOW_SKIP_CODES at 1 lot BASE without risk caps (full non-holdout only; §8, §12.2)."""
    fill_wait = int(config["fill_wait_bars"])
    shadow_count = 0
    replayed_count = 0
    total_shadow_net = ZERO

    # Accept either Sequence[DayResult] (with date + skipped) or Sequence[SkippedSignal]
    pairs: list[tuple[str | None, SkippedSignal]] = []
    for item in day_results_or_skipped:
        if hasattr(item, "date") and hasattr(item, "skipped"):
            for sk in item.skipped:
                if sk.skip_code in SHADOW_SKIP_CODES:
                    pairs.append((str(item.date), sk))
        elif isinstance(item, SkippedSignal) and item.skip_code in SHADOW_SKIP_CODES:
            pairs.append((None, item))

    for d_str, sk in pairs:
        shadow_count += 1
        if d_str is None or sk.est_stop is None:
            continue
        day = days.get(d_str)
        if day is None or (sk.strike, sk.side) not in day.options:
            continue
        ser = day.options[(sk.strike, sk.side)]
        sig_m = hhmm_to_minute(sk.signal_ts)
        cutoff_key = "expiry_last_entry_time" if day.is_expiry_day else "last_entry_time"
        ff_key = "expiry_force_flat_time" if day.is_expiry_day else "force_flat_time"
        cutoff_m = min(hhmm_to_minute(str(config[cutoff_key])), day.grid_len)
        ff_m = min(hhmm_to_minute(str(config[ff_key])), day.grid_len - 1)

        # Find first live fill bar within fill_wait_bars strictly < cutoff
        eb: int | None = None
        for off in range(fill_wait + 1):
            cand = sig_m + off
            if cand >= cutoff_m:
                break
            if bool(ser.live[cand]):
                eb = cand
                break
        if eb is None:
            continue

        fp = price_to_decimal(float(ser.open[eb]))
        stop = sk.est_stop
        if stop <= ZERO or stop >= fp:
            continue

        exit_b = ff_m
        exit_p = fp
        is_nb = False
        last_l = fp
        last_c = fp
        for b in range(eb, ff_m + 1):
            if not bool(ser.live[b]):
                if b == ff_m:
                    exit_b, exit_p, is_nb = b, min(last_l, last_c), True
                    break
                continue
            bo = price_to_decimal(float(ser.open[b]))
            bl = price_to_decimal(float(ser.low[b]))
            bc = price_to_decimal(float(ser.close[b]))
            if b > eb and bo <= stop:
                exit_b, exit_p = b, bo
                break
            if b >= ff_m:
                exit_b, exit_p = b, bo
                break
            if bl <= stop:
                exit_b, exit_p = b, stop
                break
            last_l, last_c = bl, bc

        bd = compute_trade_accounting(
            fp, exit_p, day.lot_size, eb, exit_b, day.is_expiry_day, d_str, config,
            scenario="BASE", is_no_bar=is_nb,
        )
        if bd is not None:
            total_shadow_net = round_paise(total_shadow_net + bd.net)
            replayed_count += 1

    return {
        "count": shadow_count,
        "replayed": replayed_count,
        "estimated_net_base": format_decimal(total_shadow_net, 2),
        "note": "shadow-only (never enters trades CSV, summary, or verdict)",
    }


def check_milan(
    *,
    trades: Sequence[TradeRecord],
    waterfall: Mapping[str, Any],
    autopsy: Mapping[str, Any],
    funnel: Mapping[str, Any],
    counterfactuals: Mapping[str, Any] | None,
    slices: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Run 6-point milan reconciliation; any failure flags status='GADBAD' (§12.2, T46)."""
    t_cnt = len(trades)
    sum_net = round_paise(sum((t.net_base for t in trades), ZERO))

    # 1. Waterfall net == sum(net_base)
    m1 = to_decimal(waterfall["net_base_D"]) == sum_net
    # 2. Sum of 12 autopsy class INR == total net_base
    aut_net = round_paise(sum((to_decimal(v["net_base"]) for v in autopsy["classes"].values()), ZERO))
    m2 = aut_net == sum_net
    # 3. Sum of 12 autopsy class counts == T
    aut_cnt = sum(int(v["count"]) for v in autopsy["classes"].values())
    m3 = aut_cnt == t_cnt
    # 4. Funnel sums: A == B + C, C == c1+c2+c3+c4, sum(exit_reasons) == c4 == T
    m4 = (
        funnel["decide_calls_A"] == funnel["notrade_B"] + funnel["intents_C"]
        and funnel["intents_C"]
        == funnel["c1_invalid_intent"] + funnel["c2_cancel"] + funnel["c3_risk_skip"] + funnel["c4_fills"]
        and sum(funnel["exit_reasons"].values()) == funnel["c4_fills"] == t_cnt
    )
    # 5. CF-ACTUAL net == run net_base
    m5 = True
    if counterfactuals is not None:
        cf_act = to_decimal(counterfactuals["scenarios"]["CF-ACTUAL"]["net_base"])
        m5 = cf_act == sum_net
    # 6. Every slice dimension sums to T
    m6 = True
    if slices is not None:
        for dim_cells in slices["dimensions"].values():
            if sum(int(c["n"]) for c in dim_cells.values()) != t_cnt:
                m6 = False
                break

    checks = {"m1_waterfall": m1, "m2_autopsy_inr": m2, "m3_autopsy_count": m3, "m4_funnel": m4, "m5_cf_actual": m5, "m6_slices": m6}
    passed = sum(1 for ok in checks.values() if ok)
    status = "OK (6/6)" if passed == 6 else "GADBAD"
    return {"status": status, "passed": passed, "total": 6, "checks": checks}


# ------------------------------------------------------------------------------
# 7. Detectors D01-D12 with Impact Filter (§12.2, T47, T52)
# ------------------------------------------------------------------------------
def run_detectors(
    *,
    trades: Sequence[TradeRecord],
    waterfall: Mapping[str, Any],
    entry_edge: Mapping[str, Any],
    autopsy: Mapping[str, Any],
    counterfactuals: Mapping[str, Any] | None,
    slices: Mapping[str, Any] | None,
    bharosa: Mapping[str, Any],
    funnel: Mapping[str, Any],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    is_holdout: bool = False,
) -> list[dict[str, Any]]:
    """Run detectors D01-D12 filtered by min_impact_pct_capital & holdout guard."""
    th = config["diag_thresholds"]
    base_cap = float(to_decimal(config["base_capital"]))
    min_imp = (float(th["min_impact_pct_capital"]) / 100.0) * base_cap
    red_imp = (float(th["red_impact_pct_capital"]) / 100.0) * base_cap
    t_cnt = len(trades)

    findings: list[dict[str, Any]] = []

    def _emit(did: str, sev: str, impact: float, detail: str, exp: str) -> None:
        if is_holdout and did not in HOLDOUT_ALLOWED_DETECTORS:
            return
        if abs(impact) < min_imp and sev != "HARA":
            return
        findings.append(
            {
                "id": did,
                "severity": sev,
                "impact_inr": round(impact, 2),
                "detail": detail,
                "experiment": exp,
                "label": "in-sample",
            }
        )

    # D01: Cost kill / cost share
    cs = waterfall["cost_share"]
    gross_f = float(to_decimal(waterfall["gross_A"]))
    net_f = float(to_decimal(waterfall["net_base_D"]))
    tot_cost = float(to_decimal(waterfall["fill_cost_B"]) + to_decimal(waterfall["charges_C"]))
    if (cs is not None and cs >= float(th["cost_share_red"])) or (gross_f > 0 and net_f <= 0):
        _emit("D01", "LAL", -tot_cost, f"Cost share high ({cs}); costs={tot_cost:.2f}", "Reduce trade count or widen R (>=6x spread); trade nearer ATM.")
    elif cs is not None and cs >= float(th["cost_share_yellow"]):
        _emit("D01", "PEELA", -tot_cost, f"Cost share moderate ({cs:.2%})", "Restrict entries to windows where gross/trade > 2x cost/trade.")

    # D02: Entry edge verdict
    ev = entry_edge["verdict"]
    if ev == "NAHI":
        _emit("D02", "LAL", -tot_cost, "Entry edge NAHI (in-sample)", "Replace entry signal/filter in Strategy Card; exit tuning will not fix zero entry edge.")
    elif ev == "KAMZOR":
        _emit("D02", "PEELA", -tot_cost, "Entry edge KAMZOR (stat only, below cost)", "Improve entry selectivity or lower cost per unit.")
    else:
        _emit("D02", "HARA", max(net_f, min_imp), "Entry edge HAAN (in-sample)", "Preserve core entry logic.")

    # D03: Early exit vs H*
    h_star = entry_edge.get("best_h_star")
    if ev in ("HAAN", "KAMZOR") and h_star is not None and counterfactuals is not None and t_cnt > 0:
        med_hold = float(np.median([t.hold_min for t in trades]))
        cf_key = f"CF-HOLD({h_star})"
        if med_hold < h_star and cf_key in counterfactuals["scenarios"]:
            cf_delta = float(to_decimal(counterfactuals["scenarios"][cf_key]["delta_inr"]))
            cf_net = float(to_decimal(counterfactuals["scenarios"][cf_key]["net_base"]))
            if cf_delta > 0:
                sev = "LAL" if (cf_net > 0 and net_f <= 0) else "PEELA"
                _emit("D03", sev, cf_delta, f"Median hold {med_hold:.1f}m < H*={h_star}m", f"Extend hold toward H*={h_star}m.")

    # D04: Shakeout share of stop exits
    stop_exits = sum(1 for t in trades if t.exit_reason in ("stop", "trail", "breakeven"))
    sh_cnt = int(autopsy["classes"]["SHAKEOUT"]["count"])
    sh_inr = float(to_decimal(autopsy["classes"]["SHAKEOUT"]["net_base"]))
    if stop_exits > 0:
        sh_share = sh_cnt / float(stop_exits)
        if sh_share >= float(th["shakeout_share_red"]):
            _emit("D04", "LAL", sh_inr, f"SHAKEOUT share {sh_share:.1%} of stops", "Widen stop (CF-STOPx1.5) or wait for pullback before entry.")
        elif sh_share >= float(th["shakeout_share"]):
            _emit("D04", "PEELA", sh_inr, f"SHAKEOUT share {sh_share:.1%} of stops", "Widen stop (CF-STOPx1.5) while keeping risk bounded.")

    # D05: Gave back
    gb_cnt = int(autopsy["classes"]["GAVE_BACK"]["count"])
    gb_inr = float(to_decimal(autopsy["classes"]["GAVE_BACK"]["net_base"]))
    if t_cnt > 0 and (gb_cnt / float(t_cnt) >= float(th["gaveback_share"]) or gb_inr <= -red_imp):
        sev = "LAL" if gb_inr <= -red_imp else "PEELA"
        _emit("D05", sev, gb_inr, f"GAVE_BACK {gb_cnt}/{t_cnt} trades ({gb_inr:.2f})", "Enable breakeven_trigger_r 0.5-1.0 or add manage() trail.")

    # D06: Target early
    tgt_exits = sum(1 for t in trades if t.exit_reason == "target")
    te_cnt = int(autopsy["classes"]["TARGET_EARLY"]["count"])
    if tgt_exits > 0 and (te_cnt / float(tgt_exits)) >= float(th["target_early_share"]):
        cf_d = 0.0
        if counterfactuals and "CF-TARGETx2.0" in counterfactuals["scenarios"]:
            cf_d = float(to_decimal(counterfactuals["scenarios"]["CF-TARGETx2.0"]["delta_inr"]))
        elif counterfactuals and "CF-TARGETx2" in counterfactuals["scenarios"]:
            cf_d = float(to_decimal(counterfactuals["scenarios"]["CF-TARGETx2"]["delta_inr"]))
        _emit("D06", "PEELA", max(abs(cf_d), min_imp), f"TARGET_EARLY {te_cnt}/{tgt_exits} target exits", "Increase target_r toward CF-TARGETx2.")

    # D07 & D08: PAKKA losing & winning slices
    if slices is not None:
        for dname, cmap in slices["dimensions"].items():
            for cname, cell in cmap.items():
                if cell["status"] == "PAKKA":
                    c_inr = float(to_decimal(cell["net_base"]))
                    if c_inr < 0:
                        sev = "LAL" if c_inr <= -red_imp else "PEELA"
                        _emit("D07", sev, c_inr, f"PAKKA losing slice {dname}={cname} ({c_inr:.2f})", f"Filter out {dname}={cname} (m -> m+1).")
                    elif c_inr > 0:
                        _emit("D08", "HARA", c_inr, f"PAKKA winning slice {dname}={cname} ({c_inr:.2f})", "MAT CHHEDO: do not over-size; verify on Holdout.")

    # D09: Concentration fragility
    net_ex3 = float(to_decimal(bharosa["net_ex_top_k"]))
    top5_s = float(bharosa["top5_win_share"])
    if t_cnt > 0 and net_ex3 <= 0:
        _emit("D09", "LAL", max(abs(net_f), min_imp), f"Net ex-top-3 <= 0 ({net_ex3:.2f})", "Expand sample size across more weeks.")
    elif t_cnt > 0 and top5_s > float(th["conc_share"]):
        _emit("D09", "PEELA", max(abs(net_f), min_imp), f"Top-5% win share {top5_s:.1%} > 80%", "Expand sample size; avoid treating as robust.")

    # D10: Split-half decay
    if t_cnt >= 40:
        mid = t_cnt // 2
        h1 = float(sum((t.net_base for t in trades[:mid]), ZERO))
        h2 = float(sum((t.net_base for t in trades[mid:]), ZERO))
        if h1 <= 0 and h2 <= 0:
            _emit("D10", "LAL", h2, "Both halves non-positive", "Audit regime shift; require fresh forward evidence.")
        elif h1 > 0 and h2 <= 0:
            _emit("D10", "PEELA", h2, f"2nd half decayed ({h2:.2f})", "Audit recent regime shift before proceeding.")

    # D11: Cap-skips, no_bar, or cap breach days
    intents_c = funnel["intents_C"]
    skip_sh = (funnel["c3_risk_skip"] / float(intents_c)) if intents_c > 0 else 0.0
    nb_sh = (funnel["exit_reasons"].get("no_bar", 0) / float(t_cnt)) if t_cnt > 0 else 0.0
    breaches = summary["fixed"].get("cap_breach_days", []) + summary["dynamic"].get("cap_breach_days", [])
    if breaches:
        _emit("D11", "LAL", -red_imp, f"Cap breach on {len(breaches)} day(s)", "Tighten stop distance or inspect gap days (never relax risk caps).")
    elif skip_sh >= float(th["skip_share"]) or nb_sh >= float(th["nobar_share"]):
        _emit("D11", "PEELA", -min_imp, f"Risk skip share {skip_sh:.1%}, no_bar {nb_sh:.1%}", "Inspect stop sizing and illiquid bars.")

    # D12: Overall Bharosa / STRESS / delay-1 / stale cancel
    stale_sh = (funnel["by_skip_code"].get("SKIP_STALE_CANCEL", 0) / float(intents_c)) if intents_c > 0 else 0.0
    stress_f = float(to_decimal(summary["fixed"]["net_stress"]))
    if net_f > 0 and stress_f <= 0:
        _emit("D12", "LAL", stress_f, "BASE > 0 but STRESS <= 0", "Improve cost/delay resilience before advancing.")
    elif bharosa["overall"] in ("LAL", "PEELA") or stale_sh >= float(th["stale_entry_share"]):
        _emit("D12", bharosa["overall"], -min_imp if net_f >= 0 else net_f, f"Bharosa {bharosa['overall']}, stale cancel {stale_sh:.1%}", "Increase sample and keep m low.")

    findings.sort(key=lambda f: (SEVERITY_RANK[f["severity"]], -abs(float(f["impact_inr"])), f["id"]))
    return findings


# ------------------------------------------------------------------------------
# 8. AI_HANDOFF.md Renderer with Strict Line/Byte Caps & Cut Order (§12.2, T51)
# ------------------------------------------------------------------------------
def render_ai_handoff(
    *,
    diag: Mapping[str, Any],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    is_holdout: bool = False,
    diff_summary_lines: Sequence[str] = (),
) -> str:
    """Render plain-Markdown AI_HANDOFF.md strictly <= handoff_max_lines & <= handoff_max_bytes."""
    max_lines = int(config["handoff_max_lines"])
    max_bytes = int(config["handoff_max_bytes"])
    cut_order = [int(x) for x in config["diag_grid"]["handoff_cut_order"]]

    header_6 = [
        "AI ke liye (STRICT RULES):",
        "1. Sirf strategies/strat_<name>.py edit karo; engine files LOCKED hain.",
        "2. Ek baar mein sirf 1 badlav (experiment) karo.",
        "3. 'chance ho sakta hai' cells aur 'MAT CHHEDO' slices ko mat chhedo.",
        "4. Sabhi diagnosis numbers in-sample hain; overfit mat karo.",
        "5. HOLDOUT: tuning mat karo." if is_holdout else "5. Har naya variant m badhata hai; holdout par kabhi tune mat karo.",
    ]

    wf = diag["waterfall"]
    ee = diag["entry_edge"]
    bh = diag["bharosa"]
    fn = diag["funnel"]
    ml = diag["milan"]
    dyn = summary["dynamic"]
    fix = summary["fixed"]
    findings = diag["findings"]

    # Build 12 sections (plain Markdown, zero HTML/SVG)
    sections: dict[int, list[str]] = {}
    sections[1] = [
        f"## 1. Header & Milan ({ENGINE_TAG})",
        f"- Milan: {ml['status']} | Track: [FIXED 1-LOT] primary + [DYNAMIC COMPOUNDING] panel | Mode: {summary.get('mode')}",
    ]
    ruin_note = f" | [DYNAMIC RUIN] {dyn['ruin_date']} bal={dyn['ruin_balance']}" if dyn.get("ruin_flag") else ""
    sections[2] = [
        "## 2. Verdict & Dynamic Summary",
        f"- Verdict: {summary.get('verdict')} | PAPER_READY: {summary.get('paper_ready')} | [FIXED 1-LOT] T={fix['trades_count']} BASE={fix['net_base']} STRESS={fix['net_stress']}",
        f"- [DYNAMIC COMPOUNDING] T={dyn['trades_count']} BASE={dyn['net_base']} STRESS={dyn['net_stress']} DD={dyn['max_drawdown_pct']}% LiqWarn={dyn['liquidity_warn_count']}{ruin_note}",
    ]
    sections[3] = [
        "## 3. Waterfall & Capture (in-sample)",
        f"- Gross={wf['gross_A']} FillCost={wf['fill_cost_B']} Charges={wf['charges_C']} Net={wf['net_base_D']} MFE={wf['mfe_total_M']} Giveback={wf['giveback_G']} Capture={wf['capture']} CostShare={wf['cost_share']}",
    ]
    top3 = findings[:3]
    s4 = ["## 4. Top Findings (in-sample)"]
    for f in top3:
        s4.append(f"- [{f['severity']}] {f['id']} ({f['impact_inr']} INR): {f['detail']}")
    if not top3:
        s4.append("- Koi major finding nahi.")
    sections[4] = s4

    s5 = [f"## 5. Entry Edge ({ee['verdict']}, H*={ee['best_h_star']})"]
    for hrow in ee["horizons"]:
        s5.append(f"- H={hrow['horizon_min']}m n={hrow['n']} fwd={hrow['fwd_mean']} q={hrow['q_value']} net={hrow['edge_net']}")
    sections[5] = s5

    s6 = ["## 6. Top 5 Counterfactuals (in-sample estimate)"]
    if diag.get("counterfactuals") and not is_holdout:
        cfs = sorted(
            diag["counterfactuals"]["scenarios"].items(),
            key=lambda kv: -float(to_decimal(kv[1]["delta_inr"])),
        )[:5]
        for k, v in cfs:
            s6.append(f"- {k}: net={v['net_base']} delta={v['delta_inr']}")
    else:
        s6.append("- Suppressed on HOLDOUT." if is_holdout else "- NA")
    sections[6] = s6

    s7 = ["## 7. 12 Autopsy Classes"]
    for c in AUTOPSY_CLASSES:
        cv = diag["autopsy"]["classes"][c]
        if cv["count"] > 0:
            s7.append(f"- {c}: n={cv['count']} net={cv['net_base']}")
    sections[7] = s7

    s8 = ["## 8. Slices (PAKKA vs chance)"]
    if diag.get("slices") and not is_holdout:
        pakka = [
            (d, c, cell)
            for d, cm in diag["slices"]["dimensions"].items()
            for c, cell in cm.items()
            if cell["status"] == "PAKKA"
        ]
        if pakka:
            for d, c, cell in pakka[:5]:
                s8.append(f"- PAKKA {d}={c}: n={cell['n']} net={cell['net_base']} q={cell['q_value']}")
        else:
            s8.append("- koi pakka tukda nahi")
    else:
        s8.append("- Suppressed on HOLDOUT.")
    sections[8] = s8

    sections[9] = [
        "## 9. Funnel & Shadow",
        f"- A={fn['decide_calls_A']} B={fn['notrade_B']} C={fn['intents_C']} c1={fn['c1_invalid_intent']} c2={fn['c2_cancel']} c3={fn['c3_risk_skip']} c4={fn['c4_fills']}",
    ]
    sections[10] = [
        "## 10. Bharosa & Fragility",
        f"- Overall={bh['overall']} | panels={bh['panels']} | net_ex_top3={bh['net_ex_top_k']}",
    ]
    s11 = ["## 11. Worst/Best Trades & Diff Summary"]
    pt = diag.get("autopsy", {}).get("per_trade", [])
    if pt:
        by_net = sorted(pt, key=lambda x: float(to_decimal(x["net_base"])))
        w10 = [f"#{x['trade_no']}({x['class']}:{x['net_base']})" for x in by_net[:10]]
        b5 = [f"#{x['trade_no']}({x['class']}:{x['net_base']})" for x in reversed(by_net[-5:])]
        s11.append("- Worst 10: " + ", ".join(w10))
        s11.append("- Best 5: " + ", ".join(b5))
    s11.extend(list(diff_summary_lines)[:5] if diff_summary_lines else ["- No prior engine-v2.0 run."])
    sections[11] = s11

    s12 = ["## 12. Next Experiment"]
    if is_holdout:
        s12.append("- HOLDOUT: tuning mat karo.")
    elif findings:
        f0 = findings[0]
        s12.append(f"- E-{f0['id']}-1: {f0['experiment']} (in-sample impact {f0['impact_inr']} INR)")
    else:
        s12.append("- Keep monitoring on next unseen week.")
    sections[12] = s12

    active_order = list(range(1, 13))

    def _assemble(active: list[int]) -> str:
        out_lines = list(header_6) + [""]
        for idx in active:
            out_lines.extend(sections[idx])
        return "\n".join(out_lines) + "\n"

    text = _assemble(active_order)
    # Deterministic truncation per handoff_cut_order (protected sections 1,2,3,4,12 never dropped)
    for sec_id in cut_order:
        if len(text.splitlines()) <= max_lines and len(text.encode("utf-8")) <= max_bytes:
            break
        if sec_id in active_order:
            active_order.remove(sec_id)
            text = _assemble(active_order)

    # Final hard safety trim if even protected sections somehow exceed byte/line cap
    lines_final = text.splitlines()
    if len(lines_final) > max_lines:
        lines_final = lines_final[:max_lines]
        text = "\n".join(lines_final) + "\n"
    while len(text.encode("utf-8")) > max_bytes and len(lines_final) > 6:
        lines_final.pop()
        text = "\n".join(lines_final) + "\n"
    return text


# ------------------------------------------------------------------------------
# 9. Conditional DIFF.md Writer (§3.3, §12.2, T52)
# ------------------------------------------------------------------------------
def write_diff_md(
    *,
    output_dir: Path,
    base_dir: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    exp_id: str | None,
) -> list[str]:
    """Emit DIFF.md ONLY when a prior same-family engine-v2.0 run exists in trials.jsonl."""
    trials_path = base_dir / "trials.jsonl"
    if not trials_path.is_file():
        return []
    family = str(manifest.get("strategy_family", ""))
    cur_dir = output_dir.name
    prior: dict[str, Any] | None = None
    for ln in trials_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if (
            rec.get("engine_tag") == ENGINE_TAG
            and rec.get("strategy_family") == family
            and rec.get("output_dir") != cur_dir
        ):
            prior = rec
    if prior is None:
        return []

    cur_f = to_decimal(summary["fixed"]["net_base"])
    prev_f = to_decimal(prior.get("fixed_net_base", "0.00"))
    delta_f = round_paise(cur_f - prev_f)

    cur_d = to_decimal(summary["dynamic"]["net_base"])
    prev_d = to_decimal(prior.get("dynamic_net_base", "0.00"))
    delta_d = round_paise(cur_d - prev_d)

    lines = [
        f"# DIFF vs Prior Run ({ENGINE_TAG})",
        f"- Prior run: `{prior.get('output_dir')}` (hash `{str(prior.get('strategy_hash', ''))[:12]}`, m={prior.get('m')})",
        f"- Current run: `{cur_dir}` (hash `{str(manifest.get('strategy_hash', ''))[:12]}`, m={manifest.get('m')})",
        f"- [FIXED 1-LOT] BASE Net delta: `{format_decimal(delta_f, 2)}` INR (`{prev_f}` -> `{cur_f}`)",
        f"- [DYNAMIC COMPOUNDING] BASE Net delta: `{format_decimal(delta_d, 2)}` INR (`{prev_d}` -> `{cur_d}`)",
    ]
    if exp_id:
        lines.append(f"- Experiment `{exp_id}` actual delta (Fixed BASE): `{format_decimal(delta_f, 2)}` INR")
    atomic_write_bytes(output_dir / "DIFF.md", ("\n".join(lines) + "\n").encode("utf-8"))
    return lines[1:6]


# ------------------------------------------------------------------------------
# 10. Main Entrypoint Called by run.py (Fault-Isolated; §12.2)
# ------------------------------------------------------------------------------
def run_diagnosis(
    *,
    output_dir: Path,
    dual: DualTrackResult,
    days: Mapping[str, DayData],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    base_dir: Path,
    is_holdout: bool = False,
    exp_id: str | None = None,
) -> dict[str, Any]:
    """Execute complete read-only diagnosis and write DIAGNOSIS.md, diagnosis.json, AI_HANDOFF.md, (DIFF.md)."""
    seed = int(config["seed"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 13]))
    lite = str(config.get("diag_level", "full")).lower() == "lite"
    fixed_trades = dual.fixed.trades
    m_variants = int(manifest.get("m", 1))

    waterfall = compute_waterfall(fixed_trades)
    entry_edge = compute_entry_edge(
        fixed_trades, days, config, rng=rng, lite=lite, is_holdout=is_holdout
    )
    autopsy = compute_autopsy(fixed_trades, days, config)
    counterfactuals = (
        None
        if is_holdout
        else compute_counterfactuals(
            fixed_trades, days, config, h_star=entry_edge.get("best_h_star"), lite=lite
        )
    )
    slices = (
        None
        if is_holdout
        else compute_slices(fixed_trades, days, config, rng=rng, lite=lite)
    )
    bharosa = compute_bharosa(fixed_trades, summary, m_variants, config)
    funnel = compute_funnel(dual.fixed)
    shadow = (
        None
        if (is_holdout or lite)
        else compute_shadow_pnl(dual.fixed.day_results, days, config)
    )
    milan = check_milan(
        trades=fixed_trades,
        waterfall=waterfall,
        autopsy=autopsy,
        funnel=funnel,
        counterfactuals=counterfactuals,
        slices=slices,
    )
    findings = run_detectors(
        trades=fixed_trades,
        waterfall=waterfall,
        entry_edge=entry_edge,
        autopsy=autopsy,
        counterfactuals=counterfactuals,
        slices=slices,
        bharosa=bharosa,
        funnel=funnel,
        summary=summary,
        config=config,
        is_holdout=is_holdout,
    )

    diag_payload: dict[str, Any] = {
        "engine_tag": ENGINE_TAG,
        "label": "in-sample",
        "is_holdout": is_holdout,
        "diag_level": "lite" if lite else "full",
        "milan": milan,
        "waterfall": waterfall,
        "entry_edge": entry_edge,
        "autopsy": autopsy,
        "counterfactuals": counterfactuals,
        "slices": slices,
        "bharosa": bharosa,
        "funnel": funnel,
        "shadow": shadow,
        "findings": findings,
        "dynamic_panel": {
            "trades_count": summary["dynamic"]["trades_count"],
            "net_base": summary["dynamic"]["net_base"],
            "net_stress": summary["dynamic"]["net_stress"],
            "max_drawdown_pct": summary["dynamic"]["max_drawdown_pct"],
            "liquidity_warn_count": summary["dynamic"]["liquidity_warn_count"],
            "cap_breach_days": summary["dynamic"]["cap_breach_days"],
            "ruin_flag": summary["dynamic"]["ruin_flag"],
            "ruin_date": summary["dynamic"]["ruin_date"],
            "ruin_balance": summary["dynamic"]["ruin_balance"],
        },
    }

    # Conditional DIFF.md (NEVER on holdout; §3.3, T52)
    diff_lines: list[str] = []
    if not is_holdout:
        diff_lines = write_diff_md(
            output_dir=output_dir,
            base_dir=base_dir,
            manifest=manifest,
            summary=summary,
            exp_id=exp_id,
        )

    # Write diagnosis.json
    atomic_write_bytes(
        output_dir / "diagnosis.json",
        (json.dumps(_json_clean(diag_payload), sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )

    # Write DIAGNOSIS.md ("Doctor ki Parchi" + Dynamic panel)
    top_k = int(config["diag_top_k"])
    md_lines = [
        f"# DIAGNOSIS — Doctor ki Parchi ({ENGINE_TAG}, in-sample)",
        "",
    ]
    if summary["dynamic"].get("ruin_flag"):
        md_lines.append(
            f"> **[DYNAMIC RUIN]** Halted on `{summary['dynamic']['ruin_date']}` at balance `INR {summary['dynamic']['ruin_balance']}`."
        )
        md_lines.append("")
    cls_items = sorted(autopsy["classes"].items(), key=lambda kv: -float(to_decimal(kv[1]["net_base"])))
    best_cls = cls_items[0] if cls_items else ("NA", {"count": 0, "net_base": "0.00"})
    worst2 = sorted(autopsy["classes"].items(), key=lambda kv: float(to_decimal(kv[1]["net_base"])))[:2]
    worst2_str = ", ".join(f"{k}({v['net_base']} INR, n={v['count']})" for k, v in worst2)
    q6_str = "Suppressed on HOLDOUT" if is_holdout else "koi pakka tukda nahi"
    if not is_holdout and slices is not None:
        p_cells = [
            f"{d}={c}({cell['net_base']} INR)"
            for d, cm in slices["dimensions"].items()
            for c, cell in cm.items()
            if cell["status"] == "PAKKA"
        ]
        if p_cells:
            q6_str = "; ".join(p_cells[:4])
    q8_str = "HOLDOUT: tuning mat karo" if is_holdout else (
        f"E-{findings[0]['id']}-1: {findings[0]['experiment']} (in-sample {findings[0]['impact_inr']} INR, m -> {m_variants + 1})"
        if findings else "No experiment suggested."
    )
    md_lines.extend(
        [
            f"- **Milan (6-point reconciliation):** `{milan['status']}`",
            f"1. **Kamai kahan se:** `[FIXED 1-LOT]` {best_cls[0]} ({best_cls[1]['net_base']} INR, n={best_cls[1]['count']})",
            f"2. **Gawaaya kahan:** `[FIXED 1-LOT]` {worst2_str}",
            f"3. **Entry mein edge:** `{entry_edge['verdict']}` (H*={entry_edge['best_h_star']}, cost/unit={entry_edge['mean_cost_unit']})",
            f"4. **Exit ka hisaab:** Giveback `{waterfall['giveback_G']}` INR | Capture `{waterfall['capture']}`",
            f"5. **Kharcha:** FillCost `{waterfall['fill_cost_B']}` + Charges `{waterfall['charges_C']}` | CostShare `{waterfall['cost_share']}`",
            f"6. **Kab chali / kab nahi:** {q6_str}",
            f"7. **Bharosa:** `{bharosa['overall']}` ({bharosa['panels']}, T={len(fixed_trades)}, m={m_variants})",
            f"8. **Agla experiment:** {q8_str}",
            f"- **[DYNAMIC COMPOUNDING] Risk Panel:** Net BASE `{summary['dynamic']['net_base']}` | DD `{summary['dynamic']['max_drawdown_pct']}%` | LiqWarn `{summary['dynamic']['liquidity_warn_count']}` | RUIN `{summary['dynamic']['ruin_flag']}`",
            "",
            f"## [FIXED 1-LOT] Top {top_k} Findings (in-sample)",
        ]
    )
    for f in findings[:top_k]:
        md_lines.append(f"- **[{f['severity']}] {f['id']}** ({f['impact_inr']} INR): {f['detail']} — _{f['experiment']}_")
    if not findings:
        md_lines.append("- No findings exceeded impact threshold.")
    md_lines.append("")
    atomic_write_bytes(output_dir / "DIAGNOSIS.md", ("\n".join(md_lines) + "\n").encode("utf-8"))

    # Write AI_HANDOFF.md
    handoff_text = render_ai_handoff(
        diag=diag_payload,
        summary=summary,
        config=config,
        is_holdout=is_holdout,
        diff_summary_lines=diff_lines,
    )
    atomic_write_bytes(output_dir / "AI_HANDOFF.md", handoff_text.encode("utf-8"))
    return diag_payload
