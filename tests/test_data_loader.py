"""Acceptance tests for data_loader.py (PRD §13: T17, T25, T28, T29, T37-T41).

Purpose:
    Validates updl dual-layout loading (CSV + Parquet parity), day exclusion
    rules, stale/filled -> NaN semantics, dynamic F&O grid length, contract-
    driven expiry/DTE, time-gated get_chain(asof), prior_days isolation,
    optional VIX handling, and holdout guard audit logging.

PRD Sections:
    §2 (Niyam 1, 2, 11, 12), §10.1, §10.2, §13, §14 (Step 2)

Guarding Tests (9):
    T17, T25, T28, T29, T37, T38, T39, T40, T41

PUBLIC API:
    test_T17_prior_days_isolation, test_T25_get_chain_and_asof,
    test_T28_holdout_guard_and_auto_trim, test_T29_data_quality_exclusions,
    test_T37_vendor_stale_copy_forward_rows, test_T38_dynamic_fno_grid_length,
    test_T39_data_driven_expiry_and_dte, test_T40_csv_parquet_parity,
    test_T41_optional_vix_handling
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import data_loader as DL
from contracts import DataMissingError, HoldoutLocked, LookaheadError
from conftest import (
    HAS_PYARROW,
    REPO_ROOT,
    UpdlDaySpec,
    build_updl_week,
    load_test_config,
    make_base_dir,
)


def test_T17_prior_days_isolation(tmp_path: Path) -> None:
    """T17: prior_days strictly < today, read-only arrays, excludes bad days & holdout."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(date="2026-11-18", expiry="2026-11-23", lot_size=30, spot_open=51800.0),
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30, session_type="MUHURAT"),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, spot_open=52000.0),
        UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30, spot_open=52200.0),
        UpdlDaySpec(date="2026-11-24", expiry="2026-12-28", lot_size=35, spot_open=52400.0),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 5, "holdout_start": "2026-11-24"},
    )
    res = DL.load_dataset(dirs["data_dir"], cfg, allow_holdout=True, base_dir=dirs["base_dir"])

    prior = DL.get_prior_days(res.days, "2026-11-23", warmup_days=5, holdout_start="2026-11-24")
    # Excluded MUHURAT day (2026-11-19), today (2026-11-23), and holdout (2026-11-24) must all be absent
    assert [d.date for d in prior] == ["2026-11-18", "2026-11-20"]

    # Every array in prior_days must be read-only (ValueError on write)
    for d in prior:
        for arr in (d.index.open, d.index.close, d.index.live, d.options[(52000, "CE")].close):
            assert arr.flags.writeable is False
            try:
                arr[0] = 0.0
                raise AssertionError("Expected ValueError on write to read-only prior_days array")
            except ValueError:
                pass


def test_T25_get_chain_and_asof(tmp_path: Path) -> None:
    """T25: get_chain(t-1)==option(), get_chain(t) LookaheadError, band>cache ValueError, asof age_min."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(
            date="2026-11-20",
            expiry="2026-11-23",
            lot_size=30,
            strikes=(51500, 52000, 52500),
            stale_option_minutes=frozenset({18, 19}),
        )
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    res = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    day = res.days["2026-11-20"]

    t = 20
    view = DL.TimeGatedView(
        day,
        t=t,
        chain_band_pct=float(cfg["chain_band_pct"]),
        cache_band_pct=float(cfg["cache_band_pct"]),
    )

    # Bar 17 is live: get_chain(17) matches option(52000, "CE") at index 17
    ch17 = view.get_chain(17, asof=False)
    opt_ce = view.option(52000, "CE")
    assert ch17[(52000, "CE")]["live"] is True
    assert ch17[(52000, "CE")]["age_min"] == 0
    assert math.isclose(ch17[(52000, "CE")]["close"], float(opt_ce.close[17]), rel_tol=0.0, abs_tol=1e-9)

    # get_chain(t) must raise LookaheadError (Niyam 1)
    try:
        view.get_chain(t)
        raise AssertionError("Expected LookaheadError for get_chain(t)")
    except LookaheadError:
        pass

    # band_pct > cache_band_pct (6.0) must raise ValueError
    try:
        view.get_chain(17, band_pct=7.0)
        raise AssertionError("Expected ValueError for band_pct > cache_band_pct")
    except ValueError:
        pass

    # Bars 18 and 19 are stale -> asof=False gives NaN, asof=True at i=19 returns bar 17 with age_min=2
    ch19_raw = view.get_chain(19, asof=False)
    assert ch19_raw[(52000, "CE")]["live"] is False
    assert math.isnan(ch19_raw[(52000, "CE")]["close"])

    ch19_asof = view.get_chain(19, asof=True)
    assert ch19_asof[(52000, "CE")]["live"] is False
    assert ch19_asof[(52000, "CE")]["age_min"] == 2
    assert math.isclose(ch19_asof[(52000, "CE")]["close"], float(opt_ce.close[17]), rel_tol=0.0, abs_tol=1e-9)


def test_T28_holdout_guard_and_auto_trim(tmp_path: Path) -> None:
    """T28: Holdout guard refuses unauthorized >= holdout_start, logs refusal, auto-trims default."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(date="2026-07-03", expiry="2026-07-28", lot_size=30),
        UpdlDaySpec(date="2026-07-06", expiry="2026-07-28", lot_size=30),
        UpdlDaySpec(date="2026-07-07", expiry="2026-07-28", lot_size=30),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-07-06"},
    )

    # Default design load auto-trims holdout dates (< 2026-07-06)
    res_design = DL.load_dataset(dirs["data_dir"], cfg, allow_holdout=False, base_dir=dirs["base_dir"])
    assert list(res_design.days.keys()) == ["2026-07-03"]

    # Explicit request touching >= holdout_start without allow_holdout raises HoldoutLocked & logs
    try:
        DL.load_dataset(
            dirs["data_dir"],
            cfg,
            allow_holdout=False,
            end_date="2026-07-07",
            base_dir=dirs["base_dir"],
        )
        raise AssertionError("Expected HoldoutLocked")
    except HoldoutLocked:
        pass

    log_path = dirs["base_dir"] / "holdout_access.log"
    assert log_path.is_file()
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["action"] == "REFUSED"
    assert entries[0]["first_touched"] == "2026-07-06"

    # Authorized holdout access succeeds and appends ALLOWED log entry
    res_ho = DL.load_dataset(dirs["data_dir"], cfg, allow_holdout=True, base_dir=dirs["base_dir"])
    assert "2026-07-06" in res_ho.days and "2026-07-07" in res_ho.days
    entries2 = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries2) == 2 and entries2[1]["action"] == "ALLOWED"


def test_T29_data_quality_exclusions(tmp_path: Path) -> None:
    """T29: Every §10.2 bad-data day is excluded with exact reason and zero interpolation."""
    dirs = make_base_dir(tmp_path)
    specs_w1 = [
        UpdlDaySpec(date="2026-11-02", expiry="2026-11-23", lot_size=30),  # valid
        UpdlDaySpec(date="2026-11-03", expiry="2026-11-23", lot_size=30, duplicate_index_minutes=frozenset({5})),
        UpdlDaySpec(date="2026-11-04", expiry="2026-11-23", lot_size=30, reverse_index_pair=(10, 11)),
        UpdlDaySpec(date="2026-11-05", expiry="2026-11-23", lot_size=30, bad_ohlc_index_minutes={12: (0.0, 10.0, -1.0, 5.0)}),
        UpdlDaySpec(date="2026-11-06", expiry="2026-11-23", lot_size=30, bad_ohlc_option_minutes={15: (200.0, 190.0, 210.0, 200.0)}),
        UpdlDaySpec(date="2026-11-08", expiry="2026-11-23", lot_size=30, session_type="MUHURAT"),
        UpdlDaySpec(date="2026-11-09", expiry="2026-11-23", lot_size=30, gap_index_minutes=frozenset(range(40))),
        UpdlDaySpec(date="2026-11-10", expiry="2026-11-23", lot_size=30, omit_options=True),
        UpdlDaySpec(date="2026-11-11", expiry="2026-12-28", lot_size=None),  # missing lot_size
        UpdlDaySpec(date="2026-11-12", expiry="2026-11-23", lot_size=30, spot_open=52000.0),  # conflict with week_02
    ]
    # Second folder with conflicting spot_open for 2026-11-12
    specs_w2 = [
        UpdlDaySpec(date="2026-11-12", expiry="2026-11-23", lot_size=30, spot_open=52500.0),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs_w1, write_csv=True)
    build_updl_week(dirs["data_dir"] / "week_02", specs_w2, write_csv=True)

    cfg = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    res = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])

    assert list(res.days.keys()) == ["2026-11-02"]
    excl = {e.date: e.reason for e in res.excluded_days}
    assert excl["2026-11-03"] == "duplicate_or_unsorted_ts"
    assert excl["2026-11-04"] == "duplicate_or_unsorted_ts"
    assert excl["2026-11-05"] == "invalid_ohlc"
    assert excl["2026-11-06"] == "invalid_ohlc"
    assert excl["2026-11-08"] == "special_session"
    assert excl["2026-11-09"] == "low_coverage"
    assert excl["2026-11-10"] == "missing_options"
    # Note: 2026-11-11 falls under nearest expiry 2026-11-23 unless contracts only has 2026-12-28;
    # test missing_lot_size on a date > 2026-11-23 separately below.
    assert excl["2026-11-12"] == "data_conflict"

    # Dedicated missing_lot_size check where active expiry has lot_size=None
    dirs2 = make_base_dir(tmp_path / "ml")
    build_updl_week(
        dirs2["data_dir"] / "week_01",
        [UpdlDaySpec(date="2026-11-25", expiry="2026-12-28", lot_size=None)],
        write_csv=True,
    )
    res2 = DL.load_dataset(dirs2["data_dir"], cfg, base_dir=dirs2["base_dir"])
    assert len(res2.days) == 0
    assert res2.excluded_days[0].reason == "missing_lot_size"


def test_T37_vendor_stale_copy_forward_rows(tmp_path: Path) -> None:
    """T37: Zero-volume copy-forward rows become NaN OHLC/OI & live=False; asof=True gives prior OI."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(
            date="2026-11-20",
            expiry="2026-11-23",
            lot_size=30,
            zero_vol_option_minutes=frozenset({25, 26}),
        )
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    res = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    day = res.days["2026-11-20"]
    opt = day.options[(52000, "CE")]

    for m in (25, 26):
        assert bool(opt.live[m]) is False
        assert np.isnan(opt.open[m]) and np.isnan(opt.high[m]) and np.isnan(opt.low[m])
        assert np.isnan(opt.close[m]) and np.isnan(opt.oi[m])

    view = DL.TimeGatedView(day, t=27, chain_band_pct=2.0, cache_band_pct=6.0)
    snap = view.get_chain(26, asof=True)[(52000, "CE")]
    assert snap["live"] is False
    assert snap["age_min"] == 2
    assert math.isclose(snap["oi"], float(opt.oi[24]), rel_tol=0.0, abs_tol=1e-9)


def test_T38_dynamic_fno_grid_length(tmp_path: Path) -> None:
    """T38: F&O grid length is data-driven (15:39 vs 15:29), index NaN after 15:29, no hardcoded grid length."""
    dirs = make_base_dir(tmp_path)
    normal_len = DL.hhmm_to_minute("15:29") + 1
    extended_len = DL.hhmm_to_minute("15:39") + 1
    specs = [
        UpdlDaySpec(date="2026-07-31", expiry="2026-08-25", lot_size=30, n_fno_bars=normal_len),
        UpdlDaySpec(date="2026-08-03", expiry="2026-08-25", lot_size=30, n_fno_bars=extended_len),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    res = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])

    d_pre = res.days["2026-07-31"]
    d_post = res.days["2026-08-03"]
    assert d_pre.grid_len == normal_len
    assert d_post.grid_len == extended_len
    assert np.isnan(d_post.index.close[normal_len])
    assert bool(d_post.options[(52000, "CE")].live[extended_len - 1]) is True

    # Verify zero hardcoded grid-length literals in data_loader.py
    dl_src = (REPO_ROOT / "data_loader.py").read_text(encoding="utf-8")
    assert ("37" + "4") not in dl_src
    assert ("37" + "5") not in dl_src


def test_T39_data_driven_expiry_and_dte(tmp_path: Path) -> None:
    """T39: Monday expiry 2026-11-23 has dte=0, Fri->Mon dte=1, weekday_estimate when holidays empty."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),  # Fri
        UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30),  # Mon expiry!
        UpdlDaySpec(date="2026-11-24", expiry="2026-12-28", lot_size=35),  # Future expiry outside data
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)

    cfg_no_hol = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31", "nse_holidays": []},
    )
    res = DL.load_dataset(dirs["data_dir"], cfg_no_hol, base_dir=dirs["base_dir"])
    assert res.days["2026-11-20"].dte == 1
    assert res.days["2026-11-23"].weekday == "Mon"
    assert res.days["2026-11-23"].is_expiry_day is True
    assert res.days["2026-11-23"].dte == 0
    assert res.days["2026-11-24"].dte_source == "weekday_estimate"

    # Providing nse_holidays switches dte_source to "nse_holidays" and reduces DTE by 1
    cfg_hol = load_test_config(
        REPO_ROOT,
        overrides={
            "data_layout": "csv",
            "warmup_days": 0,
            "holdout_start": "2026-12-31",
            "nse_holidays": ["2026-12-25"],
        },
    )
    res_hol = DL.load_dataset(dirs["data_dir"], cfg_hol, base_dir=dirs["base_dir"])
    assert res_hol.days["2026-11-24"].dte_source == "nse_holidays"
    assert res_hol.days["2026-11-24"].dte == res.days["2026-11-24"].dte - 1


def test_T40_csv_parquet_parity(tmp_path: Path) -> None:
    """T40: CSV (with _from_ split CSVs) and Parquet load identical DayData."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(
            date="2026-11-20",
            expiry="2026-11-23",
            lot_size=30,
            stale_option_minutes=frozenset({15}),
            include_vix=True,
            include_futures=True,
            include_greeks=True,
        ),
        UpdlDaySpec(
            date="2026-11-23",
            expiry="2026-11-23",
            lot_size=30,
            split_csv_from_date="2026-11-23",
            include_vix=True,
            include_futures=True,
            include_greeks=True,
        ),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True, write_parquet=HAS_PYARROW)

    cfg_csv = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    res_csv = DL.load_dataset(dirs["data_dir"], cfg_csv, base_dir=dirs["base_dir"])
    assert list(res_csv.days.keys()) == ["2026-11-20", "2026-11-23"]

    if HAS_PYARROW:
        cfg_pq = load_test_config(
            REPO_ROOT,
            overrides={"data_layout": "parquet", "warmup_days": 0, "holdout_start": "2026-12-31"},
        )
        res_pq = DL.load_dataset(dirs["data_dir"], cfg_pq, base_dir=dirs["base_dir"])
        assert list(res_pq.days.keys()) == list(res_csv.days.keys())
        for d in res_csv.days:
            dc, dp = res_csv.days[d], res_pq.days[d]
            assert dc.grid_len == dp.grid_len and dc.dte == dp.dte and dc.lot_size == dp.lot_size
            assert dc.strikes == dp.strikes
            assert np.array_equal(dc.index.close, dp.index.close, equal_nan=True)
            assert np.array_equal(dc.vix.close, dp.vix.close, equal_nan=True)
            assert np.array_equal(dc.futures.close, dp.futures.close, equal_nan=True)
            for k in dc.options:
                assert np.array_equal(dc.options[k].close, dp.options[k].close, equal_nan=True)
                assert np.array_equal(dc.options[k].live, dp.options[k].live)


def test_T41_optional_vix_handling(tmp_path: Path) -> None:
    """T41: Missing vix_1m does not exclude day; vix() raises DataMissingError."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(
            date="2026-11-20",
            expiry="2026-11-23",
            lot_size=30,
            include_vix=False,
        )
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    res = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    assert "2026-11-20" in res.days
    day = res.days["2026-11-20"]
    assert day.has_vix is False
    assert day.vix is None

    view = DL.TimeGatedView(day, t=20, chain_band_pct=2.0, cache_band_pct=6.0)
    try:
        view.vix()
        raise AssertionError("Expected DataMissingError when strategy requests missing VIX")
    except DataMissingError as exc:
        assert "data_missing: vix_1m" in str(exc)
