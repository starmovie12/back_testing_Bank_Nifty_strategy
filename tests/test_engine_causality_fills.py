"""Acceptance tests for engine.py causality and fills (PRD §13: T01-T06, T09, T11).

Purpose:
    Validates zero-lookahead future poisoning and avail_ts gating (T01),
    prefix causality (T02), hand-calculated 4-trade unlimited day accounting
    to exact paise (T03), same-bar stop-first and entry-bar stop checks (T04),
    gap-open stop and gap-open target fills (T05), stale entry bar wait and
    SKIP_STALE_CANCEL (T06), fast vs reference mode paise-identity (T09), and
    random-entry cost sanity (T11).

PRD Sections:
    §2 (Niyam 1-8, 10), §11.2, §11.3, §11.4, §13, §14 (Step 5)

Guarding Tests (8):
    T01, T02, T03, T04, T05, T06, T09, T11

PUBLIC API:
    test_T01_future_poison_and_avail_ts_gate, test_T02_prefix_causality,
    test_T03_hand_calculated_day_unlimited_trades,
    test_T04_same_bar_sl_target_and_entry_bar_sl,
    test_T05_gap_open_sl_and_gap_open_target,
    test_T06_stale_entry_bar_wait_and_cancel,
    test_T09_fast_mode_equals_reference_mode, test_T11_random_entry_sanity
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

import numpy as np

import costs as CO
import data_loader as DL
import engine as ENG
from contracts import Intent, NoTrade, SkipCode
from conftest import REPO_ROOT, UpdlDaySpec, build_updl_week, load_test_config, make_base_dir


def _load_single_day(tmp_path: Path, spec: UpdlDaySpec) -> tuple[DL.DayData, dict]:
    dirs = make_base_dir(tmp_path)
    build_updl_week(dirs["data_dir"] / "week_01", [spec], write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        base_capital="50000",
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    res = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    return res.days[spec.date], cfg


def test_T01_future_poison_and_avail_ts_gate(tmp_path: Path) -> None:
    """T01: Corrupting all data >= t leaves decision at t unchanged; late avail_ts bar is NaN."""
    t_check = 30
    spec = UpdlDaySpec(
        date="2026-11-20",
        expiry="2026-11-23",
        lot_size=30,
        late_avail_option_minutes={t_check - 2: t_check + 5},
    )
    day_clean, cfg = _load_single_day(tmp_path, spec)

    # Build a future-poisoned DayData where all index & option bars >= t_check are corrupted
    def _poison_series(ser: DL.BarSeries) -> DL.BarSeries:
        o = ser.open.copy()
        h = ser.high.copy()
        l = ser.low.copy()
        c = ser.close.copy()
        o[t_check:] = 999999.0
        h[t_check:] = 999999.0
        l[t_check:] = 1.0
        c[t_check:] = 999999.0
        for a in (o, h, l, c):
            a.flags.writeable = False
        return DL.BarSeries(
            open=o, high=h, low=l, close=c,
            volume=ser.volume, oi=ser.oi, live=ser.live,
            avail_minute=ser.avail_minute, half_spread_proxy=ser.half_spread_proxy,
        )

    day_poison = DL.DayData(
        date=day_clean.date,
        weekday=day_clean.weekday,
        grid_len=day_clean.grid_len,
        expiry=day_clean.expiry,
        dte=day_clean.dte,
        dte_cal=day_clean.dte_cal,
        dte_source=day_clean.dte_source,
        lot_size=day_clean.lot_size,
        is_expiry_day=day_clean.is_expiry_day,
        ref_spot=day_clean.ref_spot,
        index=_poison_series(day_clean.index),
        vix=day_clean.vix,
        futures=day_clean.futures,
        options=MappingProxyType({k: _poison_series(v) for k, v in day_clean.options.items()}),
        stocks=day_clean.stocks,
        strikes=day_clean.strikes,
        has_vix=day_clean.has_vix,
        has_futures=day_clean.has_futures,
        has_greeks=day_clean.has_greeks,
        has_stocks=day_clean.has_stocks,
    )

    observed_clean: dict[str, object] = {}
    observed_poison: dict[str, object] = {}

    def _make_probe(bag: dict[str, object]):
        def factory():
            class S:
                def decide(self, st):
                    if st.t == t_check:
                        idx_c = float(st.view.index().close[t_check - 1])
                        opt_ser = st.view.option(52000, "CE")
                        ema_v = st.view.ind.value("ema", 9)
                        bag["idx_c"] = idx_c
                        bag["late_is_nan"] = math.isnan(float(opt_ser.close[t_check - 2]))
                        bag["opt_prev"] = float(opt_ser.close[t_check - 1])
                        bag["ema"] = ema_v
                    return NoTrade("probe")
            return S()
        return factory

    ENG.run_day(day=day_clean, prior_days=(), strategy_source=_make_probe(observed_clean), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    ENG.run_day(day=day_poison, prior_days=(), strategy_source=_make_probe(observed_poison), config=cfg, track="fixed", day_open_balance=Decimal("50000"))

    assert observed_clean == observed_poison
    assert observed_clean["late_is_nan"] is True


def test_T02_prefix_causality(tmp_path: Path) -> None:
    """T02: Truncating day at t leaves all decisions <= t identical to full day."""
    t_cut = 45
    spec = UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30)
    day_full, cfg = _load_single_day(tmp_path, spec)

    def _trunc(ser: DL.BarSeries) -> DL.BarSeries:
        return ser.slice_visible(t_cut + 1)

    day_prefix = DL.DayData(
        date=day_full.date,
        weekday=day_full.weekday,
        grid_len=t_cut + 1,
        expiry=day_full.expiry,
        dte=day_full.dte,
        dte_cal=day_full.dte_cal,
        dte_source=day_full.dte_source,
        lot_size=day_full.lot_size,
        is_expiry_day=day_full.is_expiry_day,
        ref_spot=day_full.ref_spot,
        index=_trunc(day_full.index),
        vix=None,
        futures=None,
        options=MappingProxyType({k: _trunc(v) for k, v in day_full.options.items()}),
        stocks=MappingProxyType({}),
        strikes=day_full.strikes,
        has_vix=False,
        has_futures=False,
        has_greeks=False,
        has_stocks=False,
    )

    log_full: list[tuple[int, float, float]] = []
    log_pref: list[tuple[int, float, float]] = []

    def _make_logger(out_list: list[tuple[int, float, float]]):
        def factory():
            class S:
                def decide(self, st):
                    if st.t <= t_cut:
                        out_list.append((st.t, float(st.view.index().close[st.t - 1]), st.view.ind.value("ema", 5)))
                    return NoTrade("log")
            return S()
        return factory

    ENG.run_day(day=day_full, prior_days=(), strategy_source=_make_logger(log_full), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    ENG.run_day(day=day_prefix, prior_days=(), strategy_source=_make_logger(log_pref), config=cfg, track="fixed", day_open_balance=Decimal("50000"))

    assert len(log_pref) > 0
    assert log_pref == log_full


def test_T03_hand_calculated_day_unlimited_trades(tmp_path: Path) -> None:
    """T03: Synthetic 4-trade unlimited day matches manual paise accounting on every trade."""
    overrides = {
        # Trade 1: enter bar 20 @ 200.00, stop 180.00, target 220.00 -> target hit bar 21
        (52000, "CE", 20): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 202.0},
        (52000, "CE", 21): {"open": 202.0, "high": 225.0, "low": 200.0, "close": 222.0},
        # Trade 2: enter bar 25 @ 210.00, stop 190.00, target 230.00 -> stop hit bar 26
        (52000, "CE", 25): {"open": 210.0, "high": 215.0, "low": 205.0, "close": 208.0},
        (52000, "CE", 26): {"open": 208.0, "high": 210.0, "low": 185.0, "close": 188.0},
        # Trade 3: enter bar 30 @ 195.00, stop 175.00, target 215.00 -> target hit bar 31
        (52000, "CE", 30): {"open": 195.0, "high": 200.0, "low": 190.0, "close": 198.0},
        (52000, "CE", 31): {"open": 198.0, "high": 220.0, "low": 196.0, "close": 218.0},
        # Trade 4: enter bar 35 @ 205.00, stop 185.00, target 225.00 -> target hit bar 36
        (52000, "CE", 35): {"open": 205.0, "high": 210.0, "low": 200.0, "close": 207.0},
        (52000, "CE", 36): {"open": 207.0, "high": 230.0, "low": 205.0, "close": 228.0},
    }
    spec = UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=overrides)
    day, cfg = _load_single_day(tmp_path, spec)

    def strat():
        class S:
            def decide(self, st):
                if st.t in (20, 25, 30, 35):
                    return Intent(side="CE", strike=52000, lots=1, stop_points=20.0, target_r=1.0, reason="t03")
                return NoTrade("wait")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(res.trades) == 4

    expected = [
        (20, 21, Decimal("200.00"), Decimal("220.00"), "target"),
        (25, 26, Decimal("210.00"), Decimal("190.00"), "stop"),
        (30, 31, Decimal("195.00"), Decimal("215.00"), "target"),
        (35, 36, Decimal("205.00"), Decimal("225.00"), "target"),
    ]
    for tr, (eb, xb, ep, xp, er) in zip(res.trades, expected):
        assert tr.entry_price == ep and tr.exit_price == xp and tr.exit_reason == er
        bd_b = CO.compute_trade_accounting(ep, xp, 30, eb, xb, False, "2026-11-20", cfg, scenario="BASE")
        bd_s = CO.compute_trade_accounting(ep, xp, 30, eb, xb, False, "2026-11-20", cfg, scenario="STRESS")
        assert bd_b is not None and bd_s is not None
        assert tr.gross == bd_b.gross
        assert tr.charges_base == bd_b.charges
        assert tr.fill_cost_base == bd_b.fill_cost
        assert tr.net_base == bd_b.net
        assert tr.charges_stress == bd_s.charges
        assert tr.fill_cost_stress == bd_s.fill_cost
        assert tr.net_stress == bd_s.net


def test_T04_same_bar_sl_target_and_entry_bar_sl(tmp_path: Path) -> None:
    """T04: Same-bar SL+target fills stop first (Niyam 7); entry-bar SL triggers (Niyam 8)."""
    overrides = {
        # Trade 1: bar 20 enters @ 200, bar 21 touches both stop (180) and target (220)
        (52000, "CE", 20): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
        (52000, "CE", 21): {"open": 200.0, "high": 235.0, "low": 170.0, "close": 230.0},
        # Trade 2: bar 30 enters @ 200 and dips to 175 on entry bar itself (hold_min=0)
        (52000, "CE", 30): {"open": 200.0, "high": 205.0, "low": 175.0, "close": 195.0},
    }
    spec = UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=overrides)
    day, cfg = _load_single_day(tmp_path, spec)

    def strat():
        class S:
            def decide(self, st):
                if st.t in (20, 30):
                    return Intent(side="CE", strike=52000, lots=1, stop_points=20.0, target_r=1.0)
                return NoTrade("wait")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(res.trades) == 2
    assert res.trades[0].exit_reason == "stop" and res.trades[0].exit_price == Decimal("180.00")
    assert res.trades[1].exit_reason == "stop" and res.trades[1].exit_price == Decimal("180.00") and res.trades[1].hold_min == 0


def test_T05_gap_open_sl_and_gap_open_target(tmp_path: Path) -> None:
    """T05: Gap open < stop fills at open (gap_exit=True); gap open > target fills at target."""
    overrides = {
        # Trade 1: enter bar 20 @ 200, stop=180; bar 21 gaps open @ 155
        (52000, "CE", 20): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
        (52000, "CE", 21): {"open": 155.0, "high": 160.0, "low": 150.0, "close": 158.0},
        # Trade 2: enter bar 30 @ 200, stop=180, target=220; bar 31 gaps open @ 250
        (52000, "CE", 30): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
        (52000, "CE", 31): {"open": 250.0, "high": 255.0, "low": 245.0, "close": 252.0},
    }
    spec = UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=overrides)
    day, cfg = _load_single_day(tmp_path, spec)

    def strat():
        class S:
            def decide(self, st):
                if st.t in (20, 30):
                    return Intent(side="CE", strike=52000, lots=1, stop_points=20.0, target_r=1.0)
                return NoTrade("wait")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(res.trades) == 2
    assert res.trades[0].exit_reason == "stop"
    assert res.trades[0].exit_price == Decimal("155.00")
    assert res.trades[0].gap_exit is True
    assert res.trades[1].exit_reason == "target"
    assert res.trades[1].exit_price == Decimal("220.00")
    assert res.trades[1].gap_exit is False


def test_T06_stale_entry_bar_wait_and_cancel(tmp_path: Path) -> None:
    """T06: Stale bars t, t+1, t+2 -> SKIP_STALE_CANCEL; live bar at t+2 -> fills at t+2 open."""
    spec = UpdlDaySpec(
        date="2026-11-20",
        expiry="2026-11-23",
        lot_size=30,
        stale_option_minutes=frozenset({20, 21, 22, 40, 41}),  # 20..22 all stale (3 bars); 40..41 stale, 42 live
    )
    day, cfg = _load_single_day(tmp_path, spec)

    def strat():
        class S:
            def decide(self, st):
                if st.t in (20, 40):
                    return Intent(side="CE", strike=52000, lots=1, stop_points=20.0, time_stop_min=2)
                return NoTrade("wait")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert any(sk.skip_code == SkipCode.SKIP_STALE_CANCEL and sk.signal_ts == DL.minute_to_hhmm(20) for sk in res.skipped)
    assert len(res.trades) == 1
    assert res.trades[0].signal_ts == DL.minute_to_hhmm(40)
    assert res.trades[0].entry_ts == DL.minute_to_hhmm(42)


def test_T09_fast_mode_equals_reference_mode(tmp_path: Path) -> None:
    """T09: Fast mode and reference mode produce paise-identical trades on both tracks."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30, spot_open=51900.0),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, spot_open=52000.0),
        UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30, spot_open=52100.0),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        base_capital="50000",
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    ds = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    dates = sorted(ds.days.keys())

    strat_src = """
from contracts import Intent, NoTrade
PARAMS = {"step": 25}
def decide(state):
    if state.t % PARAMS["step"] == 0:
        return Intent(side="CE", strike=52000, lots=None, stop_points=30.0, target_r=1.0, time_stop_min=10)
    return NoTrade("skip")
"""
    fast_res = ENG.run_dual_track(days=ds.days, trading_dates=dates, strategy_source=strat_src, config=cfg, mode="fast")
    ref_res = ENG.run_dual_track(days=ds.days, trading_dates=dates, strategy_source=strat_src, config=cfg, mode="reference")

    assert len(fast_res.fixed.trades) > 0
    assert fast_res.fixed.trades == ref_res.fixed.trades
    assert fast_res.dynamic.trades == ref_res.dynamic.trades
    assert fast_res.dynamic.final_balance_base == ref_res.dynamic.final_balance_base


def test_T11_random_entry_sanity(tmp_path: Path) -> None:
    """T11: Seeded random strategy on flat synthetic data has net <= 0 approx equal to -total_costs."""
    # Flat option prices (200.00 every bar) so gross == 0 and net == -(charges + fill_cost)
    flat_overrides = {
        (52000, "CE", m): {"open": 200.0, "high": 201.0, "low": 199.0, "close": 200.0}
        for m in range(DL.STANDARD_INDEX_MINUTES)
    }
    spec = UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=flat_overrides)
    day, cfg = _load_single_day(tmp_path, spec)

    rng = np.random.default_rng(42)
    chosen_mins = set(int(x) for x in rng.choice(np.arange(20, 280, 15), size=8, replace=False))

    def strat():
        class S:
            def decide(self, st):
                if st.t in chosen_mins:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=30.0, time_stop_min=5)
                return NoTrade("rand_skip")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(res.trades) == 8
    total_net = sum((t.net_base for t in res.trades), Decimal("0"))
    total_costs = sum((t.charges_base + t.fill_cost_base for t in res.trades), Decimal("0"))
    assert total_net < Decimal("0")
    assert total_net == -total_costs
