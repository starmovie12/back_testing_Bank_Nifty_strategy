"""Acceptance tests for engine.py risk shell, cutoffs, and exits (PRD §13: T18, T19, T22-T24, T26, T30).

Purpose:
    Validates 15% daily cap & 4 intraday profit-buffer scenarios (T18),
    dynamic unconstrained lot scaling (T19), bar-close timing semantics (T22),
    strict exclusive < 14:45 / < 13:00 cutoffs & force-flat (T23), stale/NaN
    exit & no_bar 2x cost fallback (T24), preset/manage exits & Intent.lots
    matrix (T26), and today_trades + manage() TRAIL contract (T30).

PRD Sections:
    §2 (Niyam 2-4, 9, 10), §5.1-§5.4, §11.2, §11.3, §11.4, §13, §14 (Step 5)

Guarding Tests (7):
    T18, T19, T22, T23, T24, T26, T30

PUBLIC API:
    test_T18_daily_cap_and_intraday_profit_buffer,
    test_T19_dynamic_lot_scaling_and_cap_enforcement,
    test_T22_bar_close_timing_semantics,
    test_T23_strict_cutoff_and_force_flat,
    test_T24_stale_nan_exit_and_no_bar_fallback,
    test_T26_exit_rules_and_intent_lots_validation,
    test_T30_today_trades_and_manage_contract
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import costs as CO
import data_loader as DL
import engine as ENG
from contracts import Intent, InvalidManageAction, ManageAction, NoTrade, SkipCode
from conftest import REPO_ROOT, UpdlDaySpec, build_updl_week, load_test_config, make_base_dir


def _load_day(tmp_path: Path, spec: UpdlDaySpec, overrides: dict | None = None) -> tuple[DL.DayData, dict]:
    dirs = make_base_dir(tmp_path)
    build_updl_week(dirs["data_dir"] / "week_01", [spec], write_csv=True)
    ov = {"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"}
    if overrides:
        ov.update(overrides)
    cfg = load_test_config(REPO_ROOT, base_capital="50000", overrides=ov)
    res = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    return res.days[spec.date], cfg


def test_T18_daily_cap_and_intraday_profit_buffer(tmp_path: Path) -> None:
    """T18: All 4 §5.3 intraday profit-buffer scenarios + SKIP_DAYCAP + gap cap_breach."""
    # Scenario 1-4 via multi-trade day:
    # Trade 1 (bar 20->21): +₹3,000+ profit -> realised_loss_today == 0
    # Trade 2 (bar 25->26): -₹1,500 loss (still net green vs open) -> realised_loss_today == 0
    # Trade 3 (bar 30->31): -₹4,000 loss (net red vs open, not from peak!) -> realised_loss_today == open - balance_now
    # Trade 4 (bar 35): huge 180-pt SL exceeds remaining daily cap -> SKIP_DAYCAP
    ov = {
        (52000, "CE", 20): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
        (52000, "CE", 21): {"open": 200.0, "high": 330.0, "low": 198.0, "close": 325.0},  # +120 pt target (+3600 gross)
        (52000, "CE", 25): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
        (52000, "CE", 26): {"open": 200.0, "high": 202.0, "low": 145.0, "close": 150.0},  # -50 pt stop (-1500 gross)
        (52000, "CE", 30): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
        (52000, "CE", 31): {"open": 200.0, "high": 202.0, "low": 55.0, "close": 60.0},    # -140 pt stop (-4200 gross)
        (52000, "CE", 34): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
        (52000, "CE", 35): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
    }
    day, cfg = _load_day(tmp_path / "buf", UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=ov))

    seen_losses: dict[int, Decimal] = {}

    def strat():
        class S:
            def decide(self, st):
                if st.t in (24, 29, 34):
                    seen_losses[st.t] = st.realised_loss_today
                if st.t == 20:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=40.0, target_r=3.0)
                if st.t == 25:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=50.0, target_r=2.0)
                if st.t == 30:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=140.0, target_r=1.0)
                if st.t == 35:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=180.0, target_r=1.0)
                return NoTrade("wait")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="dynamic", day_open_balance=Decimal("50000"))
    assert len(res.trades) == 3
    # After Trade 1 (+profit): realised_loss_today == 0
    assert seen_losses[24] == Decimal("0.00")
    # After Trade 2 (profit-then-loss, still net green): realised_loss_today == 0
    assert seen_losses[29] == Decimal("0.00")
    # After Trade 3 (peak-then-giveback, net red vs 50000): anchored to day_open_balance
    expected_loss = Decimal("50000.00") - res.trades[2].balance_base
    assert expected_loss > Decimal("0")
    assert seen_losses[34] == expected_loss
    # Trade 4 blocked by SKIP_DAYCAP
    assert any(sk.skip_code == SkipCode.SKIP_DAYCAP and sk.signal_ts == DL.minute_to_hhmm(35) for sk in res.skipped)

    # Gap-open cap breach test: gap from 300 down to 10 -> loss > 15% of 50000 (7500) -> cap_breach=True
    ov_gap = {
        (52000, "CE", 19): {"open": 300.0, "high": 305.0, "low": 295.0, "close": 300.0},
        (52000, "CE", 20): {"open": 300.0, "high": 305.0, "low": 295.0, "close": 300.0},
        (52000, "CE", 21): {"open": 10.0, "high": 12.0, "low": 8.0, "close": 10.0},
    }
    day_g, cfg_g = _load_day(tmp_path / "gap", UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=ov_gap))

    def strat_gap():
        class S:
            def decide(self, st):
                if st.t == 20:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=50.0, target_r=1.0)
                return NoTrade("wait")
        return S()

    res_g = ENG.run_day(day=day_g, prior_days=(), strategy_source=strat_gap, config=cfg_g, track="dynamic", day_open_balance=Decimal("50000"))
    assert len(res_g.trades) == 1 and res_g.trades[0].gap_exit is True
    assert res_g.cap_breach is True


def test_T19_dynamic_lot_scaling_and_cap_enforcement(tmp_path: Path) -> None:
    """T19: Dynamic track with Intent.lots=None scales integer k with balance & charges scale with k."""
    ov = {
        (52000, "CE", 19): {"open": 200.0, "high": 202.0, "low": 198.0, "close": 200.0},
        (52000, "CE", 20): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
        (52000, "CE", 21): {"open": 200.0, "high": 230.0, "low": 198.0, "close": 225.0},
    }
    day, cfg = _load_day(tmp_path, UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=ov))

    def strat():
        class S:
            def decide(self, st):
                if st.t == 20:
                    return Intent(side="CE", strike=52000, lots=None, stop_points=60.0, time_stop_min=1)
                return NoTrade("wait")
        return S()

    r50 = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="dynamic", day_open_balance=Decimal("50000"))
    r100 = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="dynamic", day_open_balance=Decimal("100000"))
    assert len(r50.trades) == 1 and len(r100.trades) == 1
    assert r50.trades[0].lots == 3
    assert r100.trades[0].lots > r50.trades[0].lots
    assert r100.trades[0].charges_base > r50.trades[0].charges_base
    assert r100.trades[0].qty == r100.trades[0].lots * 30


def test_T22_bar_close_timing_semantics(tmp_path: Path) -> None:
    """T22: 09:29 close -> 09:30 decision -> 09:30 open fill (decision_step_min 1 and 5)."""
    day, cfg1 = _load_day(tmp_path, UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30))
    cfg5 = dict(cfg1, decision_step_min=5)
    m_0930 = DL.hhmm_to_minute("09:30")

    for cfg in (cfg1, cfg5):
        seen_first_t: list[int] = []

        def strat():
            class S:
                def decide(self, st):
                    seen_first_t.append(st.t)
                    if st.t == m_0930:
                        # Bar 09:29 (index m_0930 - 1) is the last visible bar
                        assert len(st.view.index()) == m_0930
                        return Intent(side="CE", strike=52000, lots=1, stop_points=30.0, time_stop_min=2)
                    return NoTrade("wait")
            return S()

        res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
        assert seen_first_t[0] == m_0930
        assert len(res.trades) == 1
        assert res.trades[0].signal_ts == "09:30"
        assert res.trades[0].entry_ts == "09:30"


def test_T23_strict_cutoff_and_force_flat(tmp_path: Path) -> None:
    """T23: All 8 §11.2 cutoff rows (<14:45 normal, <13:00 expiry) + force-flat 15:10/13:30."""
    m_1443 = DL.hhmm_to_minute("14:43")
    m_1444 = DL.hhmm_to_minute("14:44")
    m_1445 = DL.hhmm_to_minute("14:45")
    m_1258 = DL.hhmm_to_minute("12:58")
    m_1259 = DL.hhmm_to_minute("12:59")
    m_1300 = DL.hhmm_to_minute("13:00")

    d_norm_live, cfg = _load_day(tmp_path / "nl", UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30))
    d_norm_stale, _ = _load_day(tmp_path / "ns", UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, stale_option_minutes=frozenset({m_1444})))
    d_exp_live, _ = _load_day(tmp_path / "el", UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30))
    d_exp_stale, _ = _load_day(tmp_path / "es", UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30, stale_option_minutes=frozenset({m_1259})))

    def _strat_at(target_t: int):
        def factory():
            class S:
                def decide(self, st):
                    if st.t == target_t:
                        return Intent(side="CE", strike=52000, lots=1, stop_points=50.0)
                    return NoTrade("wait")
            return S()
        return factory

    # Normal day rows 1-4
    r1 = ENG.run_day(day=d_norm_live, prior_days=(), strategy_source=_strat_at(m_1443), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(r1.trades) == 1 and r1.trades[0].entry_ts == "14:43" and r1.trades[0].exit_reason == "force_flat" and r1.trades[0].exit_ts == "15:10"

    r2 = ENG.run_day(day=d_norm_live, prior_days=(), strategy_source=_strat_at(m_1444), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(r2.trades) == 1 and r2.trades[0].entry_ts == "14:44" and r2.trades[0].exit_ts <= "15:10"

    r3 = ENG.run_day(day=d_norm_stale, prior_days=(), strategy_source=_strat_at(m_1444), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(r3.trades) == 0 and any(sk.skip_code == SkipCode.SKIP_CUTOFF for sk in r3.skipped)

    r4 = ENG.run_day(day=d_norm_live, prior_days=(), strategy_source=_strat_at(m_1445), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(r4.trades) == 0

    # Expiry day rows 5-8
    r5 = ENG.run_day(day=d_exp_live, prior_days=(), strategy_source=_strat_at(m_1258), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(r5.trades) == 1 and r5.trades[0].entry_ts == "12:58" and r5.trades[0].exit_reason == "force_flat" and r5.trades[0].exit_ts == "13:30"

    r6 = ENG.run_day(day=d_exp_live, prior_days=(), strategy_source=_strat_at(m_1259), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(r6.trades) == 1 and r6.trades[0].entry_ts == "12:59" and r6.trades[0].exit_ts <= "13:30"

    r7 = ENG.run_day(day=d_exp_stale, prior_days=(), strategy_source=_strat_at(m_1259), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(r7.trades) == 0 and any(sk.skip_code == SkipCode.SKIP_CUTOFF for sk in r7.skipped)

    r8 = ENG.run_day(day=d_exp_live, prior_days=(), strategy_source=_strat_at(m_1300), config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(r8.trades) == 0


def test_T24_stale_nan_exit_and_no_bar_fallback(tmp_path: Path) -> None:
    """T24: Stale > exit_wait_bars -> no_bar at min(last_live_low, last_live_close) with 2x exit cost."""
    # Entry bar 20 live (open=200, low=195, close=198), time_stop_min=1; bars 21, 22, 23 all stale
    ov = {
        (52000, "CE", 20): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 198.0},
    }
    day, cfg = _load_day(
        tmp_path,
        UpdlDaySpec(
            date="2026-11-20",
            expiry="2026-11-23",
            lot_size=30,
            stale_option_minutes=frozenset({21, 22, 23}),
            option_overrides=ov,
        ),
    )

    def strat():
        class S:
            def decide(self, st):
                if st.t == 20:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=40.0, time_stop_min=1)
                return NoTrade("wait")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.exit_reason == "no_bar"
    assert tr.exit_price == Decimal("195.00")  # min(last_live_low=195, last_live_close=198)
    assert tr.exit_ts == DL.minute_to_hhmm(23)  # exit_wait_bars=2 exceeded on 3rd stale bar

    # Verify 2x exit cost applied
    bd_nb = CO.compute_trade_accounting(Decimal("200.00"), Decimal("195.00"), 30, 20, 23, False, "2026-11-20", cfg, scenario="BASE", is_no_bar=True)
    assert bd_nb is not None
    assert tr.fill_cost_base == bd_nb.fill_cost
    assert tr.net_base == bd_nb.net


def test_T26_exit_rules_and_intent_lots_validation(tmp_path: Path) -> None:
    """T26: Breakeven preset exit + Intent.lots matrix (None, 2, 0, -1, True, 1.5)."""
    ov = {
        # Bar 20 enters @ 200, stop=180 (R=20), breakeven_trigger_r=1.0 -> bar 21 hits 225 arms BE=200; bar 22 dips to 198 -> breakeven exit @ 200
        (52000, "CE", 20): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 202.0},
        (52000, "CE", 21): {"open": 202.0, "high": 225.0, "low": 201.0, "close": 220.0},
        (52000, "CE", 22): {"open": 218.0, "high": 219.0, "low": 198.0, "close": 199.0},
    }
    day, cfg = _load_day(tmp_path, UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=ov))

    # 1. Breakeven + lots=2 cap (Dynamic -> 2, Fixed -> 1)
    def strat_be():
        class S:
            def decide(self, st):
                if st.t == 20:
                    return Intent(side="CE", strike=52000, lots=2, stop_points=20.0, breakeven_trigger_r=1.0)
                return NoTrade("wait")
        return S()

    r_fix = ENG.run_day(day=day, prior_days=(), strategy_source=strat_be, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    r_dyn = ENG.run_day(day=day, prior_days=(), strategy_source=strat_be, config=cfg, track="dynamic", day_open_balance=Decimal("50000"))
    assert r_fix.trades[0].lots == 1 and r_fix.trades[0].exit_reason == "breakeven" and r_fix.trades[0].exit_price == Decimal("200.00")
    assert r_dyn.trades[0].lots == 2 and r_dyn.trades[0].exit_reason == "breakeven"

    # 2. Invalid lots values -> SKIP_INVALID_INTENT
    bad_lots = [0, -1, True, False, 1.5]

    def strat_bad():
        class S:
            def decide(self, st):
                idx = st.t - 20
                if 0 <= idx < len(bad_lots):
                    return Intent(side="CE", strike=52000, lots=bad_lots[idx], stop_points=20.0)
                return NoTrade("wait")
        return S()

    r_bad = ENG.run_day(day=day, prior_days=(), strategy_source=strat_bad, config=cfg, track="dynamic", day_open_balance=Decimal("50000"))
    assert len(r_bad.trades) == 0
    invalid_skips = [sk for sk in r_bad.skipped if sk.skip_code == SkipCode.SKIP_INVALID_INTENT]
    assert len(invalid_skips) == len(bad_lots)


def test_T30_today_trades_and_manage_contract(tmp_path: Path) -> None:
    """T30: today_trades <= t-1, manage() starts e+1, TRAIL floor_tick, downward/NaN raises InvalidManageAction."""
    ov = {
        (52000, "CE", 20): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 204.0},
        (52000, "CE", 21): {"open": 204.0, "high": 215.0, "low": 203.0, "close": 214.0},
        (52000, "CE", 22): {"open": 214.0, "high": 215.0, "low": 190.0, "close": 192.0},  # hits trailed stop 195.10
        (52000, "CE", 25): {"open": 200.0, "high": 205.0, "low": 195.0, "close": 202.0},
        (52000, "CE", 26): {"open": 202.0, "high": 210.0, "low": 200.0, "close": 205.0},
    }
    day, cfg = _load_day(tmp_path, UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=ov))

    manage_bars: list[int] = []
    seen_today_counts: dict[int, int] = {}

    def strat_ok():
        class S:
            def decide(self, st):
                seen_today_counts[st.t] = len(st.today_trades)
                if st.t == 20:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=20.0, exit_mode="manage")
                if st.t == 25:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=20.0, time_stop_min=1)
                return NoTrade("wait")

            def manage(self, st, pos):
                manage_bars.append(st.t)
                # TRAIL 195.14 -> floor_tick(0.05) = 195.10
                if st.t == 21:
                    return ManageAction(action="TRAIL", new_stop=195.14)
                return ManageAction(action="HOLD")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat_ok, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(res.trades) == 2
    # manage() never called on entry bar 20; starts at e+1 = 21
    assert manage_bars[0] == 21
    # Trailed stop floored to 195.10 and hit on bar 22
    assert res.trades[0].exit_reason == "trail"
    assert res.trades[0].exit_price == Decimal("195.10")
    # today_trades reflects Trade 1 at t=25 (strictly after exit at bar 22)
    assert seen_today_counts[20] == 0
    assert seen_today_counts[25] == 1

    # Downward TRAIL and NaN TRAIL must both raise InvalidManageAction
    for bad_stop in (170.0, float("nan")):
        def strat_bad_trail(ns=bad_stop):
            class S:
                def decide(self, st):
                    if st.t == 20:
                        return Intent(side="CE", strike=52000, lots=1, stop_points=20.0, exit_mode="manage")
                    return NoTrade("wait")

                def manage(self, st, pos):
                    return ManageAction(action="TRAIL", new_stop=ns)
            return S()

        try:
            ENG.run_day(day=day, prior_days=(), strategy_source=strat_bad_trail, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
            raise AssertionError(f"Expected InvalidManageAction for new_stop={bad_stop}")
        except InvalidManageAction:
            pass
