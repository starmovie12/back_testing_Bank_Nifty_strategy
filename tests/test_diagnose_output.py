"""Acceptance tests for diagnose.py outputs, BH FDR, holdout guard & 60-trade scaling (PRD §13: T49-T52, T60).

Purpose:
    Validates funnel reconciliation across all 9 SkipCodes & invalid NoTrade
    reason rejection (T49), 15 slices + BH FDR exact math + label isolation
    (T50), AI_HANDOFF.md line/byte caps & deterministic truncation preserving
    protected sections (T51), DIFF.md & Holdout diagnosis suppression guard
    (T52), and 60-trade single-day scaling with bucketed trade_of_day,
    streaming CSV, and REPORT.html Others reconciliation (T60).

PRD Sections:
    §3.3, §8, §12.1, §12.2, §13, §14 (Step 7)

Guarding Tests (5):
    T49, T50, T51, T52, T60

PUBLIC API:
    test_T49_funnel_reconciliation_and_9_skip_codes,
    test_T50_slices_bh_fdr_and_label_isolation,
    test_T51_ai_handoff_contract_and_truncation,
    test_T52_diff_md_and_holdout_diagnosis_guard,
    test_T60_60_trade_day_scaling_and_bucketed_trade_of_day
"""

from __future__ import annotations

import csv
import dataclasses
import json
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

import numpy as np

import data_loader as DL
import diagnose as DIAG
import engine as ENG
import report as REP
import run as RUN
from contracts import (
    ENGINE_TAG,
    Intent,
    InvalidNoTradeReason,
    MarketState,
    NoTrade,
    SkipCode,
    SkippedSignal,
    TradeRecord,
    highest_priority_skip,
    validate_notrade_reason,
)
from conftest import (
    REPO_ROOT,
    UpdlDaySpec,
    build_updl_week,
    load_test_config,
    make_base_dir,
    make_temp_repo,
    write_strategy,
)


def _make_sample_trade(n: int, *, date: str = "2026-11-20", entry_ts: str = "09:35", net: str = "150.00") -> TradeRecord:
    return TradeRecord(
        track="fixed", trade_no=n, date=date, weekday="Fri",
        signal_ts=entry_ts, entry_ts=entry_ts, side="CE", strike=52000,
        expiry="2026-11-23", dte=1, lots_signal=1, lots=1, lot_size=30, qty=30,
        liquidity_warn=False, entry_price=Decimal("200.00"), initial_stop=Decimal("180.00"),
        stop_pct=Decimal("0.1000"), target_price=Decimal("220.00"), exit_ts="09:40",
        exit_price=Decimal("210.00"), exit_reason="time", gap_exit=False, hold_min=5,
        gross=Decimal("300.00"), charges_base=Decimal("78.00"), fill_cost_base=Decimal("72.00"),
        net_base=Decimal(net), charges_stress=Decimal("78.00"), fill_cost_stress=Decimal("144.00"),
        net_stress=Decimal(net) - Decimal("72.00"), net_proxy=None,
        balance_base=Decimal("50000.00"), balance_stress=Decimal("50000.00"),
        outcome="WIN" if Decimal(net) > 0 else "LOSS", mae=Decimal("5.00"), mfe=Decimal("15.00"),
        mfe_min=3, mae_min=1, reason="r", tags="t1",
    )


def test_T49_funnel_reconciliation_and_9_skip_codes() -> None:
    """T49: All 9 SkipCodes reconciled in funnel (A=B+C, C=c1+c2+c3+c4, exits=T); bad NoTrade raises."""
    assert len(SkipCode) == 9
    skips = tuple(
        SkippedSignal(
            signal_ts="10:00", side="CE", strike=52000,
            est_price=Decimal("200.00"), est_stop=Decimal("180.00"),
            skip_code=code,
        )
        for code in SkipCode
    )
    trades = (_make_sample_trade(1), _make_sample_trade(2))
    trk = ENG.TrackRunResult(
        track="fixed",
        start_balance=Decimal("50000.00"),
        final_balance_base=Decimal("50300.00"),
        final_balance_stress=Decimal("50156.00"),
        trades=trades,
        skipped=skips,
        day_results=(),
        notrade_counts=MappingProxyType({"wait": 10, "filter": 5}),
        decide_calls=15 + len(skips) + len(trades),
        data_gap_skips=0,
        cap_breach_days=(),
        ruin_flag=False,
        ruin_date=None,
        ruin_balance=None,
    )

    fn = DIAG.compute_funnel(trk)
    assert fn["decide_calls_A"] == fn["notrade_B"] + fn["intents_C"]
    assert fn["intents_C"] == fn["c1_invalid_intent"] + fn["c2_cancel"] + fn["c3_risk_skip"] + fn["c4_fills"]
    assert sum(fn["exit_reasons"].values()) == fn["c4_fills"] == len(trades)
    assert all(fn["by_skip_code"][c.value] == 1 for c in SkipCode)

    # Priority order check (§8)
    assert highest_priority_skip([SkipCode.SKIP_STALE_CANCEL, SkipCode.SKIP_DAYCAP, SkipCode.SKIP_NO_BALANCE]) == SkipCode.SKIP_NO_BALANCE
    assert highest_priority_skip([SkipCode.SKIP_RISK, SkipCode.SKIP_INVALID_INTENT]) == SkipCode.SKIP_INVALID_INTENT

    # Invalid NoTrade.reason raises InvalidNoTradeReason
    for bad in ("UpperCase", "bad-dash", "has space", "x" * 25):
        try:
            validate_notrade_reason(bad)
            raise AssertionError(f"Expected InvalidNoTradeReason for {bad!r}")
        except InvalidNoTradeReason:
            pass


def test_T50_slices_bh_fdr_and_label_isolation(tmp_path: Path) -> None:
    """T50: BH q matches manual calc (±1e-9); low-n excluded from test yet in milan #6; labels isolated."""
    # 1. Manual BH FDR exact check
    raw_p = [0.005, 0.020, 0.015, 0.500]
    qs = DIAG.benjamini_hochberg(raw_p)
    # Sorted: 0.005 (r1 -> 0.020), 0.015 (r2 -> 0.030 -> min with r3 = 0.0266667), 0.020 (r3 -> 0.0266667), 0.500 (r4 -> 0.500)
    assert abs(qs[0] - 0.020) < 1e-9
    assert abs(qs[2] - (4.0 / 3.0) * 0.020) < 1e-9
    assert abs(qs[1] - (4.0 / 3.0) * 0.020) < 1e-9
    assert abs(qs[3] - 0.500) < 1e-9

    # 2. Low-n cell (< diag_min_n=20) excluded from permutation test, still summed into milan #6
    dirs = make_base_dir(tmp_path)
    build_updl_week(
        dirs["data_dir"] / "week_01",
        [UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30)],
        write_csv=True,
    )
    cfg = load_test_config(REPO_ROOT, base_capital="50000", overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31", "diag_perm_runs": 40})
    ds = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    trades = [_make_sample_trade(i) for i in range(1, 11)]  # 10 trades < diag_min_n (20)
    rng = np.random.default_rng(np.random.SeedSequence([42, 13]))
    sl = DIAG.compute_slices(trades, ds.days, cfg, rng=rng, lite=True)
    assert len(sl["dimensions"]) == 15
    assert sl["tested_cells"] == 0  # all cells have n < 20
    for dim_cells in sl["dimensions"].values():
        assert sum(c["n"] for c in dim_cells.values()) == 10
        for c in dim_cells.values():
            assert c["p_value"] is None and c["status"] == "low_n"

    # 3. Diagnosis-only labels (day_type, gap_open) are strictly absent from MarketState and DayData
    ms_fields = {f.name for f in dataclasses.fields(MarketState)}
    dd_fields = {f.name for f in dataclasses.fields(DL.DayData)}
    assert "day_type" not in ms_fields and "gap_open" not in ms_fields
    assert "day_type" not in dd_fields and "gap_open" not in dd_fields


def test_T51_ai_handoff_contract_and_truncation() -> None:
    """T51: AI_HANDOFF.md <= 200 lines, <= 24576 bytes, zero HTML/SVG, protected sections intact."""
    cfg = load_test_config(REPO_ROOT, base_capital="50000", overrides={"handoff_max_lines": 35, "handoff_max_bytes": 2500})
    diag_stub = {
        "milan": {"status": "OK (6/6)"},
        "waterfall": {"gross_A": "5000.00", "fill_cost_B": "400.00", "charges_C": "600.00", "net_base_D": "4000.00", "mfe_total_M": "7000.00", "giveback_G": "2000.00", "capture": 0.71, "cost_share": 0.20},
        "entry_edge": {"verdict": "HAAN", "best_h_star": 15, "horizons": [{"horizon_min": h, "n": 50, "fwd_mean": 5.0, "q_value": 0.01, "edge_net": True} for h in (1, 3, 5, 10, 15, 30, 60)]},
        "counterfactuals": {"scenarios": {f"CF-{i}": {"net_base": "4500.00", "delta_inr": "500.00"} for i in range(10)}},
        "autopsy": {"classes": {c: {"count": 5, "net_base": "100.00"} for c in DIAG.AUTOPSY_CLASSES}},
        "slices": {"dimensions": {"weekday": {"Mon": {"n": 25, "net_base": "500.00", "q_value": 0.02, "status": "PAKKA"}}}},
        "funnel": {"decide_calls_A": 100, "notrade_B": 40, "intents_C": 60, "c1_invalid_intent": 0, "c2_cancel": 0, "c3_risk_skip": 0, "c4_fills": 60},
        "bharosa": {"overall": "HARA", "panels": {"sample": "HARA"}, "net_ex_top_k": "3000.00"},
        "findings": [
            {"id": "D01", "severity": "LAL", "impact_inr": -1000.0, "detail": "High cost", "experiment": "Reduce trades"},
            {"id": "D04", "severity": "PEELA", "impact_inr": -800.0, "detail": "Shakeout", "experiment": "Widen stop"},
            {"id": "D05", "severity": "PEELA", "impact_inr": -600.0, "detail": "Gave back", "experiment": "Add BE"},
        ],
    }
    summary_stub = {
        "mode": "weekly",
        "verdict": "PASS",
        "paper_ready": False,
        "fixed": {"trades_count": 60, "net_base": "4000.00", "net_stress": "3200.00"},
        "dynamic": {"trades_count": 60, "net_base": "8000.00", "net_stress": "6400.00", "max_drawdown_pct": 4.5, "liquidity_warn_count": 2, "ruin_flag": False, "ruin_date": None, "ruin_balance": None},
    }

    text = DIAG.render_ai_handoff(
        diag=diag_stub,
        summary=summary_stub,
        config=cfg,
        is_holdout=False,
        diff_summary_lines=[f"- diff line {i}" for i in range(5)],
    )
    assert len(text.splitlines()) <= 35
    assert len(text.encode("utf-8")) <= 2500
    assert "<svg" not in text and "<table" not in text and "<div" not in text
    # Protected sections MUST survive truncation
    assert "AI ke liye" in text
    assert "## 1. Header & Milan" in text
    assert "## 2. Verdict & Dynamic Summary" in text
    assert "## 3. Waterfall & Capture" in text
    assert "## 4. Top Findings" in text
    assert "## 12. Next Experiment" in text


def test_T52_diff_md_and_holdout_diagnosis_guard(tmp_path: Path) -> None:
    """T52: 1st run no DIFF; 2nd run with --exp emits DIFF.md; holdout suppresses tuning & stamps warning."""
    dirs = make_base_dir(tmp_path)
    build_updl_week(
        dirs["data_dir"] / "week_01",
        [UpdlDaySpec(date="2026-07-06", expiry="2026-07-28", lot_size=30)],
        write_csv=True,
    )
    cfg = load_test_config(
        REPO_ROOT,
        base_capital="50000",
        overrides={
            "data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-07-06",
            "diag_random_runs": 20, "diag_perm_runs": 30,
        },
    )
    ds = DL.load_dataset(dirs["data_dir"], cfg, allow_holdout=True, auto_trim_holdout=False, base_dir=dirs["base_dir"])
    strat = "from contracts import Intent, NoTrade\ndef decide(s):\n    return Intent(side='CE', strike=52000, lots=1, stop_points=30.0, time_stop_min=5) if s.t in (20, 60) else NoTrade('w')\n"
    dual = ENG.run_dual_track(days=ds.days, trading_dates=["2026-07-06"], strategy_source=strat, config=cfg)

    manifest = {"engine_tag": ENGINE_TAG, "strategy_family": "strat_ho", "strategy_hash": "h1", "m": 1}
    od1 = dirs["output_dir"] / "run1"
    s1 = REP.generate_reports(output_dir=od1, dual=dual, days=ds.days, trading_dates=["2026-07-06"], config=cfg, manifest=manifest, mode="range")
    DIAG.run_diagnosis(output_dir=od1, dual=dual, days=ds.days, summary=s1, manifest=manifest, config=cfg, base_dir=dirs["base_dir"], is_holdout=False)
    assert not (od1 / "DIFF.md").exists()

    # Record run1 in trials.jsonl, then run2 with --exp E-D01-1 -> DIFF.md created!
    RUN.append_trial_record(
        dirs["base_dir"],
        {"engine_tag": ENGINE_TAG, "strategy_family": "strat_ho", "strategy_hash": "h1", "m": 1, "output_dir": "run1", "fixed_net_base": s1["fixed"]["net_base"], "dynamic_net_base": s1["dynamic"]["net_base"]},
    )
    od2 = dirs["output_dir"] / "run2"
    s2 = REP.generate_reports(output_dir=od2, dual=dual, days=ds.days, trading_dates=["2026-07-06"], config=cfg, manifest=manifest, mode="range")
    DIAG.run_diagnosis(output_dir=od2, dual=dual, days=ds.days, summary=s2, manifest=manifest, config=cfg, base_dir=dirs["base_dir"], is_holdout=False, exp_id="E-D01-1")
    assert (od2 / "DIFF.md").is_file()
    assert "E-D01-1" in (od2 / "DIFF.md").read_text(encoding="utf-8")

    # Holdout run: suppresses slices, counterfactuals, shadow, D03-D08, D10, DIFF.md & stamps warning
    od_ho = dirs["output_dir"] / "run_ho"
    s_ho = REP.generate_reports(output_dir=od_ho, dual=dual, days=ds.days, trading_dates=["2026-07-06"], config=cfg, manifest=manifest, mode="holdout", is_holdout=True)
    d_ho = DIAG.run_diagnosis(output_dir=od_ho, dual=dual, days=ds.days, summary=s_ho, manifest=manifest, config=cfg, base_dir=dirs["base_dir"], is_holdout=True)
    assert d_ho["slices"] is None
    assert d_ho["counterfactuals"] is None
    assert d_ho["shadow"] is None
    assert d_ho["entry_edge"]["best_h_star"] is None
    assert all(f["id"] in DIAG.HOLDOUT_ALLOWED_DETECTORS for f in d_ho["findings"])
    assert not (od_ho / "DIFF.md").exists()
    assert "HOLDOUT: tuning mat karo" in (od_ho / "AI_HANDOFF.md").read_text(encoding="utf-8")


def test_T60_60_trade_day_scaling_and_bucketed_trade_of_day(tmp_path: Path) -> None:
    """T60: 60-trade single day -> trade_of_day buckets sum to 60 (6th+=55), streaming CSV complete, HTML capped with Others, handoff capped."""
    dirs = make_base_dir(tmp_path)
    build_updl_week(
        dirs["data_dir"] / "week_01",
        [UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30)],
        write_csv=True,
    )
    cfg = load_test_config(
        REPO_ROOT,
        base_capital="500000",  # large enough so 60 trades never hit daily cap
        overrides={
            "data_layout": "csv",
            "warmup_days": 0,
            "holdout_start": "2026-12-31",
            "random_runs": 20,
            "mc_runs": 20,
            "diag_random_runs": 20,
            "diag_perm_runs": 40,
            "report_html_max_trade_rows": 20,
        },
    )
    ds = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])

    # 60 entries spaced every 5 bars starting at bar 15 (15, 20, ..., 310 < 330 cutoff), time_stop_min=1
    entry_bars = tuple(15 + i * 5 for i in range(60))
    strat_60 = f"""
from contracts import Intent, NoTrade
def decide(state):
    if state.t in {entry_bars!r}:
        return Intent(side="CE", strike=52000, lots=1, stop_points=30.0, time_stop_min=1)
    return NoTrade("wait")
"""
    dual = ENG.run_dual_track(days=ds.days, trading_dates=["2026-11-20"], strategy_source=strat_60, config=cfg)
    assert len(dual.fixed.trades) == 60

    od = dirs["output_dir"] / "heavy60"
    manifest = {"engine_tag": ENGINE_TAG, "strategy_family": "strat_60", "strategy_hash": "h60", "m": 1}
    summary = REP.generate_reports(output_dir=od, dual=dual, days=ds.days, trading_dates=["2026-11-20"], config=cfg, manifest=manifest, mode="range")
    diag = DIAG.run_diagnosis(output_dir=od, dual=dual, days=ds.days, summary=summary, manifest=manifest, config=cfg, base_dir=dirs["base_dir"], is_holdout=False)

    # 1. trade_of_day buckets reconcile strictly to 60 (1st=1, 2nd=1, 3rd=1, 4th-5th=2, 6th+=55)
    tod = diag["slices"]["dimensions"]["trade_of_day"]
    assert sum(c["n"] for c in tod.values()) == 60
    assert tod["6th+"]["n"] == 55
    assert diag["milan"]["status"] == "OK (6/6)"

    # 2. Streaming CSV has all 60 trades
    with open(od / "trades_fixed.csv", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 60

    # 3. REPORT.html caps rows at 20 and includes reconciling Others (40 trades) row
    html_text = (od / "REPORT.html").read_text(encoding="utf-8")
    assert "Others (40 trades)" in html_text

    # 4. AI_HANDOFF.md strictly within line & byte caps
    ho = (od / "AI_HANDOFF.md").read_text(encoding="utf-8")
    assert len(ho.splitlines()) <= int(cfg["handoff_max_lines"])
    assert len(ho.encode("utf-8")) <= int(cfg["handoff_max_bytes"])
