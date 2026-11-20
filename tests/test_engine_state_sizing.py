"""Acceptance tests for engine.py state, sizing, and special sessions (PRD §13: T31, T34-T36, T54, T62).

Purpose:
    Validates zero-cooloff immediate re-entry & both cap_basis modes (T31),
    Niyam 14 per-day fresh strategy namespace & read-only PARAMS (T34),
    clock exits & manage() calls on stale bars (T35), Niyam 13 pre-size
    monotonicity k_fill <= k_signal & SKIP_NO_PRICE (T36), V5.2 §5.2 Decimal
    sizing fixtures & 100% 1-lot exception (T54), and mid-span lot-size
    revision (30 -> 35) + special Saturday/MUHURAT exclusion (T62).

PRD Sections:
    §2 (Niyam 10, 13, 14), §5.1, §5.2, §10.1, §10.2, §11.4, §13, §14 (Step 5)

Guarding Tests (6):
    T31, T34, T35, T36, T54, T62

PUBLIC API:
    test_T31_accounting_and_zero_cooloff,
    test_T34_cross_day_state_isolation_niyam14,
    test_T35_clock_exits_on_stale_bars,
    test_T36_presize_monotonicity_niyam13,
    test_T54_v52_sizing_fixtures_and_100pct_exception,
    test_T62_mid_span_lot_size_revision_and_special_session
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import costs as CO
import data_loader as DL
import engine as ENG
from contracts import Intent, ManageAction, NoTrade, SkipCode
from conftest import REPO_ROOT, UpdlDaySpec, build_updl_week, load_test_config, make_base_dir


def _load_days(tmp_path: Path, specs: list[UpdlDaySpec], overrides: dict | None = None) -> tuple[DL.DatasetResult, dict]:
    dirs = make_base_dir(tmp_path)
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    ov = {"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"}
    if overrides:
        ov.update(overrides)
    cfg = load_test_config(REPO_ROOT, base_capital="50000", overrides=ov)
    res = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    return res, cfg


def test_T31_accounting_and_zero_cooloff(tmp_path: Path) -> None:
    """T31: net = gross - charges - fill_cost; immediate re-entry on cooloff_min=0; both cap_basis modes."""
    ov = {
        # Trade 1 enters bar 20 @ 200, stopped on bar 21 @ 180 (LOSS)
        (52000, "CE", 20): {"open": 200.0, "high": 202.0, "low": 195.0, "close": 198.0},
        (52000, "CE", 21): {"open": 198.0, "high": 199.0, "low": 175.0, "close": 178.0},
        # Trade 2 signals immediately at bar 22 (next bar after loss exit at 21!)
        (52000, "CE", 22): {"open": 180.0, "high": 205.0, "low": 179.0, "close": 202.0},
        (52000, "CE", 23): {"open": 202.0, "high": 210.0, "low": 200.0, "close": 205.0},
    }
    ds, cfg = _load_days(tmp_path, [UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, option_overrides=ov)])
    day = ds.days["2026-11-20"]

    def strat():
        class S:
            def decide(self, st):
                if st.t in (20, 22):
                    return Intent(side="CE", strike=52000, lots=1, stop_points=20.0, time_stop_min=1)
                return NoTrade("wait")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert len(res.trades) == 2
    assert res.trades[0].outcome == "LOSS" and res.trades[0].exit_ts == DL.minute_to_hhmm(21)
    assert res.trades[1].entry_ts == DL.minute_to_hhmm(22)
    for tr in res.trades:
        assert tr.net_base == tr.gross - tr.charges_base - tr.fill_cost_base
        assert tr.net_stress == tr.gross - tr.charges_stress - tr.fill_cost_stress

    # Both cap_basis modes ("stop" vs "premium") produce distinct risk sizing
    m_mid = DL.hhmm_to_minute("10:00")
    cfg_stop = dict(cfg, cap_basis="stop")
    cfg_prem = dict(cfg, cap_basis="premium")
    r_stop = CO.trade_max_risk_stress(1, Decimal("202.00"), Decimal("142.00"), 30, m_mid, False, "2026-04-01", cfg_stop)
    r_prem = CO.trade_max_risk_stress(1, Decimal("202.00"), Decimal("142.00"), 30, m_mid, False, "2026-04-01", cfg_prem)
    assert r_prem > r_stop


def test_T34_cross_day_state_isolation_niyam14(tmp_path: Path) -> None:
    """T34: Fresh per-day namespace resets module/class state across days; PARAMS is read-only."""
    specs = [
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
    ]
    ds, cfg = _load_days(tmp_path, specs)

    # Strategy with module-level mutable counter: must reset to 0 on Day 2 (Niyam 14)
    strat_src = """
from contracts import Intent, NoTrade
PARAMS = {"target_t": 20}
_DAY_CALLS = 0

def decide(state):
    global _DAY_CALLS
    _DAY_CALLS += 1
    # Verify PARAMS is read-only MappingProxyType
    try:
        PARAMS["target_t"] = 99
        raise AssertionError("PARAMS was mutable!")
    except TypeError:
        pass
    if _DAY_CALLS == 1:
        return Intent(side="CE", strike=52000, lots=1, stop_points=30.0, time_stop_min=1, reason="first_call")
    return NoTrade("done_today")
"""
    res = ENG.run_track(
        days=ds.days,
        trading_dates=["2026-11-19", "2026-11-20"],
        strategy_source=strat_src,
        config=cfg,
        track="fixed",
    )
    # If _DAY_CALLS leaked across days, Day 2 would have 0 trades; fresh namespace gives 1 trade per day
    assert len(res.trades) == 2
    assert [t.date for t in res.trades] == ["2026-11-19", "2026-11-20"]


def test_T35_clock_exits_on_stale_bars(tmp_path: Path) -> None:
    """T35: 15:09 & 15:10 stale -> force-flat fires on 15:11 live; manage() called on stale bar."""
    m_1509 = DL.hhmm_to_minute("15:09")
    m_1510 = DL.hhmm_to_minute("15:10")
    m_1511 = DL.hhmm_to_minute("15:11")
    spec = UpdlDaySpec(
        date="2026-11-20",
        expiry="2026-11-23",
        lot_size=30,
        stale_option_minutes=frozenset({25, m_1509, m_1510}),  # bar 25 stale during manage(); 15:09 & 15:10 stale
    )
    ds, cfg = _load_days(tmp_path, [spec])
    day = ds.days["2026-11-20"]

    manage_called_on_stale: list[bool] = []

    def strat():
        class S:
            def decide(self, st):
                if st.t == 24:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=50.0, exit_mode="manage")
                return NoTrade("wait")

            def manage(self, st, pos):
                if st.t == 25:
                    manage_called_on_stale.append(True)
                return ManageAction(action="HOLD")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
    assert manage_called_on_stale == [True]
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "force_flat"
    assert res.trades[0].exit_ts == DL.minute_to_hhmm(m_1511)


def test_T36_presize_monotonicity_niyam13(tmp_path: Path) -> None:
    """T36: Cheaper fill never increases k (k_fill <= k_signal); expensive fill reduces k; missing t-1 -> SKIP_NO_PRICE."""
    # Case A: Cheap fill open (bar 19 close=200 -> est_price=202 -> k_signal=3; bar 20 open drops to 100 -> k_fill still <= 3!)
    # Case B: Expensive fill open (bar 29 close=100 -> est_price=101 -> k_signal=5; bar 30 open jumps to 300 -> k_fill reduced!)
    # Case C: Missing live option close <= t-1 (bars 0..14 all stale -> t=15 emits SKIP_NO_PRICE)
    ov = {
        (52000, "CE", 19): {"open": 200.0, "high": 202.0, "low": 198.0, "close": 200.0},
        (52000, "CE", 20): {"open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0},
        (52000, "CE", 21): {"open": 102.0, "high": 105.0, "low": 100.0, "close": 103.0},
        (52000, "CE", 29): {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
        (52000, "CE", 30): {"open": 300.0, "high": 305.0, "low": 298.0, "close": 302.0},
        (52000, "CE", 31): {"open": 302.0, "high": 305.0, "low": 300.0, "close": 303.0},
    }
    spec = UpdlDaySpec(
        date="2026-11-20",
        expiry="2026-11-23",
        lot_size=30,
        stale_option_minutes=frozenset(range(15)),  # bars 0..14 stale so t=15 has no prior live close
        option_overrides=ov,
    )
    ds, cfg = _load_days(tmp_path, [spec])
    day = ds.days["2026-11-20"]

    def strat():
        class S:
            def decide(self, st):
                if st.t == 15:
                    return Intent(side="CE", strike=52000, lots=None, stop_points=60.0, time_stop_min=1)
                if st.t == 20:
                    return Intent(side="CE", strike=52000, lots=None, stop_frac=0.30, time_stop_min=1)
                if st.t == 30:
                    return Intent(side="CE", strike=52000, lots=None, stop_points=30.0, time_stop_min=1)
                return NoTrade("wait")
        return S()

    res = ENG.run_day(day=day, prior_days=(), strategy_source=strat, config=cfg, track="dynamic", day_open_balance=Decimal("50000"))
    # Case C: SKIP_NO_PRICE at 09:30 (t=15)
    assert any(sk.skip_code == SkipCode.SKIP_NO_PRICE and sk.signal_ts == "09:30" for sk in res.skipped)
    assert len(res.trades) == 2
    # Case A: cheaper fill never increased lots above k_signal
    t_cheap = res.trades[0]
    assert t_cheap.lots_signal == 3 and t_cheap.lots == 3 and t_cheap.lots <= t_cheap.lots_signal
    # Case B: expensive fill strictly reduced lots below k_signal
    t_exp = res.trades[1]
    assert t_exp.lots < t_exp.lots_signal


def test_T54_v52_sizing_fixtures_and_100pct_exception() -> None:
    """T54: Exact §5.2 Decimal fixtures (k=3, 2, 2), 100% 1-lot exception, and SKIP_DAYCAP."""
    cfg = load_test_config(REPO_ROOT, base_capital="50000")
    m_mid = DL.hhmm_to_minute("10:00")

    # Fixture 1: ₹50,000, loss=0, per_trade_max_risk_pct=None -> k = 3
    k1, s1 = ENG.size_order(
        track="dynamic", price=Decimal("202.00"), stop=Decimal("142.00"), lot_size=30,
        minute_offset=m_mid, is_expiry_day=False, trade_date="2026-04-01",
        balance_now=Decimal("50000"), day_open_balance=Decimal("50000"),
        realised_loss_today=Decimal("0"), intent_lots=None, config=cfg,
    )
    assert k1 == 3 and s1 is None

    # Fixture 2: ₹50,000, loss=2000, per_trade_max_risk_pct=None -> k = 2
    k2, s2 = ENG.size_order(
        track="dynamic", price=Decimal("202.00"), stop=Decimal("142.00"), lot_size=30,
        minute_offset=m_mid, is_expiry_day=False, trade_date="2026-04-01",
        balance_now=Decimal("48000"), day_open_balance=Decimal("50000"),
        realised_loss_today=Decimal("2000"), intent_lots=None, config=cfg,
    )
    assert k2 == 2 and s2 is None

    # Fixture 3: ₹50,000, loss=0, per_trade_max_risk_pct=0.10 -> k = 2
    cfg_10 = dict(cfg, per_trade_max_risk_pct="0.10")
    k3, s3 = ENG.size_order(
        track="dynamic", price=Decimal("202.00"), stop=Decimal("142.00"), lot_size=30,
        minute_offset=m_mid, is_expiry_day=False, trade_date="2026-04-01",
        balance_now=Decimal("50000"), day_open_balance=Decimal("50000"),
        realised_loss_today=Decimal("0"), intent_lots=None, config=cfg_10,
    )
    assert k3 == 2 and s3 is None

    # Small-balance 100% 1-lot exception (85% of 6500 = 5525 < 6134.14, 100% = 6500 >= 6134.14; 20-pt SL fits 15% cap)
    k_exc, s_exc = ENG.size_order(
        track="dynamic", price=Decimal("202.00"), stop=Decimal("182.00"), lot_size=30,
        minute_offset=m_mid, is_expiry_day=False, trade_date="2026-04-01",
        balance_now=Decimal("6500"), day_open_balance=Decimal("6500"),
        realised_loss_today=Decimal("0"), intent_lots=None, config=cfg,
    )
    assert k_exc == 1 and s_exc is None

    # ₹10,000 balance, 60-pt SL: 1-lot affordable, but 15% cap (1500) < STRESS risk (2025.59) -> SKIP_DAYCAP
    k_dc, s_dc = ENG.size_order(
        track="dynamic", price=Decimal("202.00"), stop=Decimal("142.00"), lot_size=30,
        minute_offset=m_mid, is_expiry_day=False, trade_date="2026-04-01",
        balance_now=Decimal("10000"), day_open_balance=Decimal("10000"),
        realised_loss_today=Decimal("0"), intent_lots=None, config=cfg,
    )
    assert k_dc == 0 and s_dc == SkipCode.SKIP_DAYCAP


def test_T62_mid_span_lot_size_revision_and_special_session(tmp_path: Path) -> None:
    """T62: Mid-span lot_size roll (30 -> 35) applies on exact roll date; special Saturday excluded."""
    specs = [
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-21", expiry="2026-11-23", lot_size=30, session_type="SPECIAL"),
        UpdlDaySpec(date="2026-11-24", expiry="2026-12-28", lot_size=35),
    ]
    ds, cfg = _load_days(tmp_path, specs)
    assert list(ds.days.keys()) == ["2026-11-20", "2026-11-24"]
    assert len(ds.excluded_days) == 1
    assert ds.excluded_days[0].date == "2026-11-21"
    assert ds.excluded_days[0].reason == "special_session"

    def strat():
        class S:
            def decide(self, st):
                if st.t == 20:
                    return Intent(side="CE", strike=52000, lots=1, stop_points=30.0, time_stop_min=2)
                return NoTrade("wait")
        return S()

    res = ENG.run_track(
        days=ds.days,
        trading_dates=["2026-11-20", "2026-11-24"],
        strategy_source=strat,
        config=cfg,
        track="fixed",
    )
    assert len(res.trades) == 2
    assert res.trades[0].lot_size == 30 and res.trades[0].qty == 30
    assert res.trades[1].lot_size == 35 and res.trades[1].qty == 35
