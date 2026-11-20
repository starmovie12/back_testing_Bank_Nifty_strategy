"""Acceptance tests for costs.py (PRD §13: T07, T44).

Purpose:
    Verifies all 8 Upstox screenshot charge fixtures, pre/post 2026-04-01 STT
    date switch, sell floor ₹0.00, MAX-rule window spread multipliers, no_bar
    exit multiplier, strict float/bool rejection, and PROXY cost scenario with
    and without iv_greeks_1m.

PRD Sections:
    §2 (Niyam 5, 6), §10.3, §10.4, §13, §14 (Step 3)

Guarding Tests (2):
    T07, T44

PUBLIC API:
    test_T07_upstox_fixtures_stt_switch_and_window_mult,
    test_T44_proxy_scenario_with_and_without_greeks
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import costs as CO
import data_loader as DL
from contracts import format_decimal
from conftest import REPO_ROOT, UpdlDaySpec, build_updl_week, load_test_config, make_base_dir


def test_T07_upstox_fixtures_stt_switch_and_window_mult() -> None:
    """T07: 8/8 Upstox fixtures (±₹0.01), STT 0.10%->0.15% switch, sell floor ₹0, MAX window mult."""
    cfg = load_test_config(REPO_ROOT)

    # 1. Verified Upstox BUY-only fixtures @ ₹426.00 (Apr 2026+ STT 0.15%)
    buy_expected = [
        (30, Decimal("41.14")),
        (60, Decimal("46.88")),
        (90, Decimal("52.62")),
        (120, Decimal("58.36")),
    ]
    for qty, exp in buy_expected:
        got = CO.order_charges("BUY", Decimal("426.00") * Decimal(qty), cfg, "2026-04-01")
        assert abs(got - exp) <= Decimal("0.01"), f"BUY qty={qty}: got {got}, exp {exp}"
        assert got == exp

    # 2. Verified Upstox Round-Trip BUY ₹424.95 + SELL ₹426.00 fixtures
    rt_expected = [
        (30, Decimal("101.06")),
        (60, Decimal("131.30")),
        (90, Decimal("161.57")),  # engine 161.56 vs screenshot 161.57 (within ±₹0.01)
        (120, Decimal("191.82")),
    ]
    for qty, screenshot_val in rt_expected:
        got = (
            CO.order_charges("BUY", Decimal("424.95") * Decimal(qty), cfg, "2026-04-01")
            + CO.order_charges("SELL", Decimal("426.00") * Decimal(qty), cfg, "2026-04-01")
        )
        assert abs(got - screenshot_val) <= Decimal("0.01"), (
            f"RT qty={qty}: got {got}, screenshot {screenshot_val}"
        )

    # 3. STT date switch boundary: 2026-03-31 (0.10%) vs 2026-04-01 (0.15%)
    sell_pre = CO.order_charges("SELL", Decimal("100000.00"), cfg, "2026-03-31")
    sell_post = CO.order_charges("SELL", Decimal("100000.00"), cfg, "2026-04-01")
    assert sell_post - sell_pre == Decimal("50.00")
    # BUY side is unaffected by STT switch
    buy_pre = CO.order_charges("BUY", Decimal("100000.00"), cfg, "2026-03-31")
    buy_post = CO.order_charges("BUY", Decimal("100000.00"), cfg, "2026-04-01")
    assert buy_pre == buy_post

    # 4. Window multiplier MAX rule (never added or multiplied) & effective slip table (§10.4)
    m_open = DL.hhmm_to_minute("09:20")
    m_mid = DL.hhmm_to_minute("10:30")
    m_close_norm = DL.hhmm_to_minute("14:45")
    m_close_exp = DL.hhmm_to_minute("13:05")

    assert CO.window_spread_mult(m_mid, False, cfg) == Decimal("2.0")
    assert CO.window_spread_mult(m_mid, True, cfg) == Decimal("3.0")
    assert CO.window_spread_mult(m_open, True, cfg) == Decimal("4.0")
    assert CO.window_spread_mult(m_close_norm, False, cfg) == Decimal("4.0")
    assert CO.window_spread_mult(m_close_exp, True, cfg) == Decimal("4.0")

    assert CO.entry_slip_unit(m_mid, False, cfg, scenario="BASE") == Decimal("1.20")
    assert CO.entry_slip_unit(m_mid, False, cfg, scenario="STRESS") == Decimal("2.40")
    assert CO.entry_slip_unit(m_mid, True, cfg, scenario="BASE") == Decimal("1.70")
    assert CO.entry_slip_unit(m_mid, True, cfg, scenario="STRESS") == Decimal("3.40")
    assert CO.entry_slip_unit(m_open, False, cfg, scenario="BASE") == Decimal("2.20")
    assert CO.entry_slip_unit(m_open, False, cfg, scenario="STRESS") == Decimal("4.40")
    assert CO.exit_slip_unit(m_mid, False, cfg, scenario="BASE", is_no_bar=True) == Decimal("2.40")

    # 5. Sell floor ₹0.00 and zero double-counting accounting identity
    bd = CO.compute_trade_accounting(
        entry_price=Decimal("5.00"),
        exit_price=Decimal("0.40"),
        qty=30,
        entry_min=m_mid,
        exit_min=m_mid,
        is_expiry_day=False,
        trade_date="2026-04-01",
        config=cfg,
        scenario="BASE",
    )
    assert bd is not None
    assert bd.sell_value == Decimal("0.00")
    assert bd.net == bd.gross - bd.charges - bd.fill_cost

    # 6. Float and bool strictly rejected in Decimal money path (L05)
    for bad in (1.25, True, False):
        try:
            CO.to_decimal(bad)
            raise AssertionError(f"Expected TypeError for {bad!r}")
        except TypeError:
            pass


def test_T44_proxy_scenario_with_and_without_greeks(tmp_path: Path) -> None:
    """T44: PROXY populates net_proxy when iv_greeks_1m present, '' when absent; zero verdict impact."""
    cfg = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    m_entry = DL.hhmm_to_minute("10:00")
    m_exit = DL.hhmm_to_minute("10:15")

    # Day WITH iv_greeks_1m
    dirs_with = make_base_dir(tmp_path / "with_greeks")
    build_updl_week(
        dirs_with["data_dir"] / "week_01",
        [UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, include_greeks=True)],
        write_csv=True,
    )
    day_with = DL.load_dataset(dirs_with["data_dir"], cfg, base_dir=dirs_with["base_dir"]).days["2026-11-20"]
    assert day_with.has_greeks is True
    opt_with = day_with.options[(52000, "CE")]
    assert opt_with.half_spread_proxy is not None

    px_with = CO.compute_trade_accounting(
        entry_price=Decimal("200.00"),
        exit_price=Decimal("220.00"),
        qty=30,
        entry_min=m_entry,
        exit_min=m_exit,
        is_expiry_day=False,
        trade_date="2026-11-20",
        config=cfg,
        scenario="PROXY",
        proxy_entry_hs=opt_with.half_spread_proxy[m_entry],
        proxy_exit_hs=opt_with.half_spread_proxy[m_exit],
    )
    assert px_with is not None
    assert px_with.net == px_with.gross - px_with.charges - px_with.fill_cost
    assert format_decimal(px_with.net, 2) != ""

    # Day WITHOUT iv_greeks_1m
    dirs_no = make_base_dir(tmp_path / "without_greeks")
    build_updl_week(
        dirs_no["data_dir"] / "week_01",
        [UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, include_greeks=False)],
        write_csv=True,
    )
    day_no = DL.load_dataset(dirs_no["data_dir"], cfg, base_dir=dirs_no["base_dir"]).days["2026-11-20"]
    assert day_no.has_greeks is False
    opt_no = day_no.options[(52000, "CE")]
    assert opt_no.half_spread_proxy is None

    px_without = CO.compute_trade_accounting(
        entry_price=Decimal("200.00"),
        exit_price=Decimal("220.00"),
        qty=30,
        entry_min=m_entry,
        exit_min=m_exit,
        is_expiry_day=False,
        trade_date="2026-11-20",
        config=cfg,
        scenario="PROXY",
        proxy_entry_hs=None,
        proxy_exit_hs=None,
    )
    assert px_without is None
    assert format_decimal(None, 2) == ""

    # BASE and STRESS results are identical regardless of iv_greeks_1m availability
    for scen in ("BASE", "STRESS"):
        b1 = CO.compute_trade_accounting(
            Decimal("200.00"), Decimal("220.00"), 30, m_entry, m_exit, False, "2026-11-20", cfg,
            scenario=scen, proxy_entry_hs=opt_with.half_spread_proxy[m_entry], proxy_exit_hs=opt_with.half_spread_proxy[m_exit],
        )
        b2 = CO.compute_trade_accounting(
            Decimal("200.00"), Decimal("220.00"), 30, m_entry, m_exit, False, "2026-11-20", cfg,
            scenario=scen, proxy_entry_hs=None, proxy_exit_hs=None,
        )
        assert b1 == b2
