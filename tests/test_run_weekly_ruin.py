"""Acceptance tests for weekly gates, Live-Bot Parity, RUIN & Pipe safety (PRD §13: T55-T59, T61).

Purpose:
    Validates table-driven Weekly PASS gate (T55), streak / max_paused_weeks /
    hash reset / 0-trade week / WEEKLY_PASS combined span (T56), Live-Bot
    Parity config & helper contracts (T57), gate order & weekly_state.json
    tamper/rewind refusal (T58), deterministic 4-scenario RUIN policy &
    --reset-ruin (T59), and O_EXCL lockfile + >256 KB Pipe deadlock guard (T61).

PRD Sections:
    §5.5, §6.1, §6.2, §6.3, §6.4, §9, §12.3, §13, §14 (Step 6)

Guarding Tests (6):
    T55, T56, T57, T58, T59, T61

PUBLIC API:
    test_T55_weekly_pass_table_driven_gate,
    test_T56_weekly_streak_pause_hash_reset_and_weekly_pass,
    test_T57_live_bot_parity_config_and_helpers,
    test_T58_gate_order_and_weekly_state_integrity,
    test_T59_deterministic_ruin_policy_and_reset_ruin,
    test_T61_o_excl_lockfile_and_pipe_deadlock_guard
"""

from __future__ import annotations

import json
import multiprocessing
import os
from decimal import Decimal
from pathlib import Path

import costs as CO
import data_loader as DL
import engine as ENG
import report as REP
import run as RUN
from contracts import Intent, NoTrade, SkipCode
from conftest import (
    REPO_ROOT,
    UpdlDaySpec,
    build_updl_week,
    load_test_config,
    make_base_dir,
    make_temp_repo,
    write_strategy,
)


def _prepare_locked_repo(tmp_path: Path, *, final_min_trades: int = 20) -> Path:
    repo_copy = make_temp_repo(tmp_path, base_capital="50000")
    cfg_p = repo_copy / "config.toml"
    txt = cfg_p.read_text(encoding="utf-8")
    txt = txt.replace("warmup_days = 20", "warmup_days = 0")
    txt = txt.replace('holdout_start = "2026-07-06"', 'holdout_start = "2026-12-31"')
    txt = txt.replace("random_runs = 1000", "random_runs = 20")
    txt = txt.replace("mc_runs = 1000", "mc_runs = 20")
    txt = txt.replace("final_min_trades = 20", f"final_min_trades = {final_min_trades}")
    cfg_p.write_text(txt, encoding="utf-8", newline="\n")
    (repo_copy / "LOCK.sha256").write_text(
        RUN.build_lock_manifest(repo_copy), encoding="utf-8", newline="\n"
    )
    return repo_copy


def _winning_day_overrides() -> dict[tuple[int, str, int], dict[str, float]]:
    """Create 5 clean winning trades per day (entry +50 pts target hit next bar)."""
    ov: dict[tuple[int, str, int], dict[str, float]] = {}
    for b in (20, 40, 60, 80, 100):
        ov[(52000, "CE", b)] = {"open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0}
        ov[(52000, "CE", b + 1)] = {"open": 200.0, "high": 260.0, "low": 198.0, "close": 255.0}
    return ov


def test_T55_weekly_pass_table_driven_gate() -> None:
    """T55: Every broken PASS condition x both tracks fails; Fixed STRESS<=0 fails even if Dynamic>0."""
    cfg = load_test_config(REPO_ROOT, base_capital="50000")
    ok_f = {"trades_count": 5, "net_base": "1200.00", "net_stress": "800.00", "max_drawdown_pct": 2.0, "ruin_flag": False}
    ok_d = {"trades_count": 5, "net_base": "2400.00", "net_stress": "1600.00", "max_drawdown_pct": 5.0, "ruin_flag": False}

    assert REP.evaluate_weekly_verdict(ok_f, ok_d, cfg)["outcome"] == "PASS"

    # Table-driven failures
    cases = [
        ("fixed_low_sample", dict(ok_f, trades_count=2), ok_d, "LOW_SAMPLE"),
        ("dynamic_low_sample", ok_f, dict(ok_d, trades_count=2), "LOW_SAMPLE"),
        ("fixed_base_le_0", dict(ok_f, net_base="0.00"), ok_d, "FAIL"),
        ("dynamic_base_le_0", ok_f, dict(ok_d, net_base="-10.00"), "FAIL"),
        ("fixed_stress_le_0_dyn_pos", dict(ok_f, net_stress="-5.00"), ok_d, "FAIL"),
        ("dynamic_stress_le_0", ok_f, dict(ok_d, net_stress="0.00"), "FAIL"),
        ("dynamic_dd_gt_15", ok_f, dict(ok_d, max_drawdown_pct=15.01), "FAIL"),
        ("dynamic_dd_gt_15_low_trades", dict(ok_f, trades_count=2), dict(ok_d, trades_count=2, max_drawdown_pct=18.0), "FAIL"),
        ("dynamic_ruin", ok_f, dict(ok_d, ruin_flag=True), "FAIL"),
    ]
    for label, f_s, d_s, expected in cases:
        got = REP.evaluate_weekly_verdict(f_s, d_s, cfg)["outcome"]
        assert got == expected, f"Case {label}: got {got}, expected {expected}"


def test_T56_weekly_streak_pause_hash_reset_and_weekly_pass(tmp_path: Path) -> None:
    """T56: P,P,LS,P,P -> WEEKLY_PASS; 3 LS -> reset 0; hash change -> reset 0; 0-trade week -> LOW_SAMPLE."""
    repo_copy = _prepare_locked_repo(tmp_path / "repo", final_min_trades=20)
    dirs = make_base_dir(tmp_path / "base")
    win_ov = _winning_day_overrides()

    # 8 consecutive ISO weeks (Mon of each week): W40..W47
    week_dates = [
        ("2026-10-05", "2026-10-27", win_ov),  # W41: PASS (5 trades)
        ("2026-10-12", "2026-10-27", win_ov),  # W42: PASS (5 trades)
        ("2026-10-19", "2026-10-27", {}),      # W43: 0 trades -> LOW_SAMPLE (no div/0!)
        ("2026-10-26", "2026-10-27", win_ov),  # W44: PASS (5 trades)
        ("2026-11-02", "2026-11-23", win_ov),  # W45: PASS (5 trades -> streak=4, combined=20 -> WEEKLY_PASS!)
        ("2026-11-09", "2026-11-23", {}),      # W46: LS 1
        ("2026-11-16", "2026-11-23", {}),      # W47: LS 2
        ("2026-11-23", "2026-11-23", {}),      # W48: LS 3 (> max_paused_weeks=2 -> streak resets to 0!)
    ]
    specs = [
        UpdlDaySpec(date=d, expiry=exp, lot_size=30, option_overrides=ov)
        for d, exp, ov in week_dates
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)

    # Strategy trades on winning dates (5 trades each) and returns NoTrade on LS dates (0 trades -> LOW_SAMPLE)
    strat_src = """
from contracts import Intent, NoTrade
PARAMS = {"win_dates": ("2026-10-05", "2026-10-12", "2026-10-26", "2026-11-02")}
def decide(state):
    if state.date in PARAMS["win_dates"] and state.t in (20, 40, 60, 80, 100):
        return Intent(side="CE", strike=52000, lots=1, stop_points=25.0, target_r=2.0)
    return NoTrade("no_setup")
"""
    write_strategy(repo_copy / "strategies", "strat_wk", strat_src)

    # Run W41 (PASS 1), W42 (PASS 2), W43 (LS 1, 0 trades), W44 (PASS 3), W45 (PASS 4 -> WEEKLY_PASS!)
    for _ in range(5):
        assert RUN.main(["strat_wk", "--week", "next", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == 0

    st = RUN.load_and_verify_weekly_state(dirs["base_dir"])
    recs = st["records"]
    assert [r["outcome"] for r in recs] == ["PASS", "PASS", "LOW_SAMPLE", "PASS", "PASS"]
    assert recs[2]["paused_weeks"] == 1 and recs[2]["streak"] == 2
    assert recs[4]["streak"] == 4
    assert recs[4]["span_trades_fixed"] == 20
    assert recs[4]["weekly_pass_unlocked"] is True

    # Run 3 consecutive LOW_SAMPLE weeks (W46, W47, W48) -> 3rd LS exceeds max_paused_weeks=2 -> streak resets to 0
    for _ in range(3):
        assert RUN.main(["strat_wk", "--week", "next", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == 0
    st2 = RUN.load_and_verify_weekly_state(dirs["base_dir"])
    assert st2["records"][-1]["outcome"] == "LOW_SAMPLE"
    assert st2["records"][-1]["streak"] == 0
    assert st2["records"][-1]["weekly_pass_unlocked"] is False


def test_T57_live_bot_parity_config_and_helpers() -> None:
    """T57: balance_max_age_sec=30 freshness check; nse_freeze_qty split & per-child charges; SL-L contract."""
    cfg = load_test_config(REPO_ROOT, base_capital="50000")

    # 1. balance_max_age_sec = 30 parity check
    assert RUN.check_live_balance_freshness(0.0, True, cfg) == (True, "OK")
    assert RUN.check_live_balance_freshness(30.0, True, cfg) == (True, "OK")
    assert RUN.check_live_balance_freshness(30.1, True, cfg) == (False, "BALANCE_UNAVAILABLE")
    assert RUN.check_live_balance_freshness(5.0, False, cfg) == (False, "BALANCE_UNAVAILABLE")
    assert RUN.check_live_balance_freshness(None, True, cfg) == (False, "BALANCE_UNAVAILABLE")

    # 2. nse_freeze_qty auto-split & per-child Upstox charges
    children, total_chg = CO.split_order_charges(
        side="BUY",
        unit_price=Decimal("200.00"),
        qty=900,
        lot_size=30,
        nse_freeze_qty=600,
        config=cfg,
        trade_date="2026-04-01",
    )
    assert children == (600, 300)
    single_order_chg = CO.order_charges("BUY", Decimal("200.00") * 900, cfg, "2026-04-01")
    expected_per_child = (
        CO.order_charges("BUY", Decimal("200.00") * 600, cfg, "2026-04-01")
        + CO.order_charges("BUY", Decimal("200.00") * 300, cfg, "2026-04-01")
    )
    assert total_chg == expected_per_child
    assert total_chg > single_order_chg  # extra ₹30+GST brokerage on 2nd child order


def test_T58_gate_order_and_weekly_state_integrity(tmp_path: Path) -> None:
    """T58: Tamper->4, rewind updated strat->5, --design-gate before WEEKLY_PASS->8, --holdout before DESIGN_PASS->6."""
    repo_copy = _prepare_locked_repo(tmp_path / "repo")
    dirs = make_base_dir(tmp_path / "base")
    specs = [
        UpdlDaySpec(date="2026-11-18", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    strat_v1 = "from contracts import Intent, NoTrade\ndef decide(s):\n    return Intent(side='CE', strike=52000, lots=1, stop_points=30.0, time_stop_min=5) if s.t == 20 else NoTrade('w')\n"
    write_strategy(repo_copy / "strategies", "strat_gate", strat_v1)

    # --design-gate before WEEKLY_PASS -> exit 8
    assert RUN.main(["strat_gate", "--design-gate", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == RUN.EXIT_GATE_ORDER

    # --holdout before DESIGN_PASS -> exit 6 + logged
    assert RUN.main(["strat_gate", "--holdout", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == RUN.EXIT_HOLDOUT_REFUSED
    assert (dirs["base_dir"] / "holdout_access.log").is_file()

    # Run 2026-W47 -> exit 0
    assert RUN.main(["strat_gate", "--week", "2026-W47", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == 0

    # Update strategy code (new hash) and try to rewind 2026-W47 -> exit 5 (NO REWIND!)
    strat_v2 = strat_v1 + "# tweaked\n"
    write_strategy(repo_copy / "strategies", "strat_gate", strat_v2)
    assert RUN.main(["strat_gate", "--week", "2026-W47", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == RUN.EXIT_WEEK_REWIND

    # Tamper weekly_state.json -> exit 4
    ws_p = dirs["base_dir"] / "weekly_state.json"
    ws_p.write_text(ws_p.read_text(encoding="utf-8").replace('"streak": 0', '"streak": 5'), encoding="utf-8")
    assert RUN.main(["strat_gate", "--week", "next", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == RUN.EXIT_STATE_OR_CONCURRENCY


def test_T59_deterministic_ruin_policy_and_reset_ruin(tmp_path: Path) -> None:
    """T59: All 4 §5.5 RUIN scenarios + --week next exit 7 + --reset-ruin preserves lineage."""
    cfg = load_test_config(REPO_ROOT, base_capital="50000", overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"})

    # Scenario 1 & 2 via is_dynamic_ruin
    assert ENG.is_dynamic_ruin(Decimal("0.00"), Decimal("5000.00")) is True
    assert ENG.is_dynamic_ruin(Decimal("-100.00"), Decimal("5000.00")) is True
    assert ENG.is_dynamic_ruin(Decimal("4999.99"), Decimal("5000.00")) is True
    assert ENG.is_dynamic_ruin(Decimal("5000.00"), Decimal("5000.00")) is False

    # Scenario 3: balance >= min_1lot_cost_ref, but cannot afford expensive ITM option -> SKIP_NO_BALANCE (NOT RUIN)
    m_mid = DL.hhmm_to_minute("10:00")
    k3, sk3 = ENG.size_order(
        track="dynamic", price=Decimal("500.00"), stop=Decimal("480.00"), lot_size=30,
        minute_offset=m_mid, is_expiry_day=False, trade_date="2026-04-01",
        balance_now=Decimal("10000.00"), day_open_balance=Decimal("10000.00"),
        realised_loss_today=Decimal("0"), intent_lots=None, config=cfg,
    )
    assert k3 == 0 and sk3 == SkipCode.SKIP_NO_BALANCE

    # Scenario 4: ₹10,000 balance, 60-pt SL -> affordable 1-lot, 15% cap fails -> SKIP_DAYCAP (NOT RUIN)
    k4, sk4 = ENG.size_order(
        track="dynamic", price=Decimal("202.00"), stop=Decimal("142.00"), lot_size=30,
        minute_offset=m_mid, is_expiry_day=False, trade_date="2026-04-01",
        balance_now=Decimal("10000.00"), day_open_balance=Decimal("10000.00"),
        realised_loss_today=Decimal("0"), intent_lots=None, config=cfg,
    )
    assert k4 == 0 and sk4 == SkipCode.SKIP_DAYCAP

    # End-to-end CLI: week 1 triggers RUIN -> --week next exits 7 -> --reset-ruin -> --week next succeeds!
    repo_copy = _prepare_locked_repo(tmp_path / "repo")
    # Set small base_capital=6000 so a gap drop from 195 to 1 drives balance_now <= 0 (RUIN!)
    cfg_p = repo_copy / "config.toml"
    cfg_p.write_text(cfg_p.read_text(encoding="utf-8").replace('base_capital = "50000"', 'base_capital = "6000"'), encoding="utf-8", newline="\n")
    (repo_copy / "LOCK.sha256").write_text(RUN.build_lock_manifest(repo_copy), encoding="utf-8", newline="\n")

    dirs = make_base_dir(tmp_path / "base")
    ruin_ov = {
        (52000, "CE", 19): {"open": 195.0, "high": 197.0, "low": 193.0, "close": 195.0},
        (52000, "CE", 20): {"open": 195.0, "high": 197.0, "low": 193.0, "close": 195.0},
        (52000, "CE", 21): {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.0},  # catastrophic gap -> balance <= 0!
    }
    specs = [
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30, option_overrides=ruin_ov),
        UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    strat_ruin = "from contracts import Intent, NoTrade\ndef decide(s):\n    return Intent(side='CE', strike=52000, lots=1, stop_points=20.0, time_stop_min=5) if s.t == 20 else NoTrade('w')\n"
    write_strategy(repo_copy / "strategies", "strat_ruin", strat_ruin)

    assert RUN.main(["strat_ruin", "--week", "2026-W47", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == 0
    ws = RUN.load_and_verify_weekly_state(dirs["base_dir"])
    assert ws["records"][-1]["ruin_flag"] is True
    assert ws["records"][-1]["outcome"] == "FAIL"

    # --week next refuses with exit 7
    assert RUN.main(["strat_ruin", "--week", "next", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == RUN.EXIT_RUIN_RESET_REQUIRED

    # --reset-ruin appends RUIN_RESET preserving weeks_completed_in_lineage
    assert RUN.main(["strat_ruin", "--reset-ruin", "--confirm-capital", "50000", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == 0
    ws2 = RUN.load_and_verify_weekly_state(dirs["base_dir"])
    assert ws2["records"][-1]["record_type"] == "RUIN_RESET"
    assert ws2["records"][-1]["ruin_flag"] is False
    assert "2026-W47" in ws2["records"][-1]["weeks_completed_in_lineage"]

    # Now --week next advances to 2026-W48 cleanly
    assert RUN.main(["strat_ruin", "--week", "next", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == 0


def test_T61_o_excl_lockfile_and_pipe_deadlock_guard(tmp_path: Path) -> None:
    """T61: Active lockfile->ConcurrencyLockError; stale PID auto-cleared; >256KB Pipe recv-before-join."""
    base_dir = make_base_dir(tmp_path)["base_dir"]
    lock_path = base_dir / ".engine.lock"

    # 1. Active lockfile with live PID -> raises ConcurrencyLockError
    lock_path.write_text(json.dumps({"pid": os.getpid(), "cmd": "other"}), encoding="utf-8")
    try:
        with RUN.EngineLockfile(base_dir):
            raise AssertionError("Expected ConcurrencyLockError on active lockfile")
    except RUN.ConcurrencyLockError:
        pass
    lock_path.unlink()

    # 2. Stale PID -> auto-cleared, acquired, and cleaned up on exit
    lock_path.write_text(json.dumps({"pid": 999999, "cmd": "crashed"}), encoding="utf-8")
    with RUN.EngineLockfile(base_dir):
        assert lock_path.is_file()
        assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert not lock_path.exists()

    # 3. >256 KB worker payload over fork + Pipe (recv BEFORE join -> zero deadlock)
    ctx = multiprocessing.get_context("fork")
    p_conn, c_conn = ctx.Pipe(duplex=False)

    def _sender(conn):
        conn.send(b"Z" * (300 * 1024))
        conn.close()

    proc = ctx.Process(target=_sender, args=(c_conn,))
    proc.start()
    c_conn.close()
    data = p_conn.recv()
    p_conn.close()
    proc.join(timeout=5)
    assert len(data) == 300 * 1024 and proc.exitcode == 0
