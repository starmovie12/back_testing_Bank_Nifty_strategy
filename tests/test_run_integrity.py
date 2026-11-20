"""Acceptance tests for run.py & report.py integrity (PRD §13: T08, T10, T12, T14, T15, T20).

Purpose:
    Validates bit-for-bit determinism (T08), LOCK.sha256 single-byte tamper
    refusal & no-.git handling (T10), cheat-strategy AST & runtime defense
    (T12), Dual-Track report completeness & labels (T14), paper account
    reconciliation on BASE & STRESS (T15), and parallel Fixed vs serial
    bit-exactness with zero forbidden concurrency primitives (T20).

PRD Sections:
    §3.2, §4, §11.1, §12.1, §12.3, §13, §14 (Step 6)

Guarding Tests (6):
    T08, T10, T12, T14, T15, T20

PUBLIC API:
    test_T08_bit_for_bit_determinism, test_T10_lock_integrity_check,
    test_T12_cheat_strategy_defense,
    test_T14_report_completeness_and_dual_track_labels,
    test_T15_paper_account_reconciliation,
    test_T20_parallel_fixed_vs_serial_and_no_pool_queue
"""

from __future__ import annotations

import ast
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import data_loader as DL
import engine as ENG
import run as RUN
from contracts import LookaheadError
from conftest import (
    REPO_ROOT,
    UpdlDaySpec,
    build_updl_week,
    load_test_config,
    make_base_dir,
    make_temp_repo,
    write_strategy,
)

_SIMPLE_STRAT = """
from contracts import Intent, NoTrade
PARAMS = {"bars": (20, 60)}
def decide(state):
    if state.t in PARAMS["bars"]:
        return Intent(side="CE", strike=52000, lots=None, stop_points=30.0, time_stop_min=5, reason="simple")
    return NoTrade("wait")
"""


def _prepare_locked_repo(tmp_path: Path, *, warmup_days: int = 0) -> Path:
    repo_copy = make_temp_repo(tmp_path, base_capital="50000")
    cfg_p = repo_copy / "config.toml"
    txt = cfg_p.read_text(encoding="utf-8")
    txt = txt.replace("warmup_days = 20", f"warmup_days = {warmup_days}")
    txt = txt.replace('holdout_start = "2026-07-06"', 'holdout_start = "2026-12-31"')
    txt = txt.replace("random_runs = 1000", "random_runs = 30")
    txt = txt.replace("mc_runs = 1000", "mc_runs = 30")
    cfg_p.write_text(txt, encoding="utf-8", newline="\n")
    (repo_copy / "LOCK.sha256").write_text(
        RUN.build_lock_manifest(repo_copy), encoding="utf-8", newline="\n"
    )
    return repo_copy


def test_T08_bit_for_bit_determinism(tmp_path: Path) -> None:
    """T08: Running identical command twice yields identical SHA-256 on CSVs & summary.json."""
    repo_copy = _prepare_locked_repo(tmp_path / "repo")
    write_strategy(repo_copy / "strategies", "strat_det", _SIMPLE_STRAT)

    specs = [
        UpdlDaySpec(date="2026-11-18", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
    ]
    outs: list[Path] = []
    for tag in ("run_a", "run_b"):
        dirs = make_base_dir(tmp_path / tag)
        build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
        rc = RUN.main(
            [
                "strat_det",
                "--week",
                "2026-W47",
                "--workers",
                "1",
                "--repo-root",
                str(repo_copy),
                "--base-dir",
                str(dirs["base_dir"]),
            ]
        )
        assert rc == RUN.EXIT_OK
        od = list(dirs["output_dir"].iterdir())[0]
        outs.append(od)

    for fname in ("trades_fixed.csv", "trades_dynamic.csv", "summary.json"):
        h1 = hashlib.sha256((outs[0] / fname).read_bytes()).hexdigest()
        h2 = hashlib.sha256((outs[1] / fname).read_bytes()).hexdigest()
        assert h1 == h2, f"Determinism mismatch on {fname}"


def test_T10_lock_integrity_check(tmp_path: Path) -> None:
    """T10: Flipping 1 byte in locked .py, config.toml, tests/, or LOCK.sha256 exits 3; no-.git -> NA."""
    specs = [UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30)]

    # 1. Clean no-.git repo succeeds and records git_check == "NA (no .git)"
    repo_clean = _prepare_locked_repo(tmp_path / "clean")
    write_strategy(repo_clean / "strategies", "strat_ok", _SIMPLE_STRAT)
    dirs = make_base_dir(tmp_path / "base_clean")
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    rc = RUN.main(
        ["strat_ok", "--week", "2026-W47", "--workers", "1", "--repo-root", str(repo_clean), "--base-dir", str(dirs["base_dir"])]
    )
    assert rc == RUN.EXIT_OK
    od = list(dirs["output_dir"].iterdir())[0]
    man = json.loads((od / "manifest.json").read_text(encoding="utf-8"))
    assert man["git_check"] == "NA (no .git)"

    # 2. Flip 1 byte in each of the 4 protected targets -> exit 3
    for label, rel_target in (
        ("locked_py", "costs.py"),
        ("config", "config.toml"),
        ("tests_tree", "tests/conftest.py"),
        ("lock_manifest", "LOCK.sha256"),
    ):
        r_tamper = _prepare_locked_repo(tmp_path / f"tamper_{label}")
        write_strategy(r_tamper / "strategies", "strat_ok", _SIMPLE_STRAT)
        target_fp = r_tamper / rel_target
        data = bytearray(target_fp.read_bytes())
        data[0] = ord(" ") if data[0] != ord(" ") else ord("#")
        target_fp.write_bytes(bytes(data))
        d_t = make_base_dir(tmp_path / f"base_{label}")
        build_updl_week(d_t["data_dir"] / "week_01", specs, write_csv=True)
        rc_t = RUN.main(
            ["strat_ok", "--week", "2026-W47", "--workers", "1", "--repo-root", str(r_tamper), "--base-dir", str(d_t["base_dir"])]
        )
        assert rc_t == RUN.EXIT_LOCK_OR_AST_FAIL, f"Tamper {label} returned {rc_t}, expected 3"


def test_T12_cheat_strategy_defense(tmp_path: Path) -> None:
    """T12: Cheat strategies (.base/.ctypes/.data/frombuffer/ndarray/open/numpy.random/get_chain(t)) blocked."""
    static_cheats = {
        "dot_base": "def decide(s):\n    x = s.view.index().close.base\n    return None\n",
        "dot_ctypes": "def decide(s):\n    x = s.view.index().close.ctypes\n    return None\n",
        "dot_data": "def decide(s):\n    x = s.view.index().close.data\n    return None\n",
        "frombuffer": "import numpy as np\ndef decide(s):\n    return np.frombuffer(b'1234')\n",
        "ndarray_buf": "import numpy as np\ndef decide(s):\n    return np.ndarray((1,), buffer=b'1234')\n",
        "builtin_open": "def decide(s):\n    return open('/etc/passwd')\n",
        "numpy_random": "import numpy as np\ndef decide(s):\n    return np.random.rand()\n",
    }
    for name, src in static_cheats.items():
        viols = RUN.check_strategy_ast(src, filename=f"{name}.py")
        assert len(viols) >= 1, f"Cheat {name} was NOT caught by AST static checker!"

    # Runtime lookahead cheat: get_chain(state.t) raises LookaheadError
    dirs = make_base_dir(tmp_path)
    build_updl_week(
        dirs["data_dir"] / "week_01",
        [UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30)],
        write_csv=True,
    )
    cfg = load_test_config(REPO_ROOT, overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"})
    day = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"]).days["2026-11-20"]

    cheat_chain = """
from contracts import NoTrade
def decide(state):
    state.view.get_chain(state.t)
    return NoTrade("cheat")
"""
    try:
        ENG.run_day(day=day, prior_days=(), strategy_source=cheat_chain, config=cfg, track="fixed", day_open_balance=Decimal("50000"))
        raise AssertionError("Expected LookaheadError for get_chain(state.t)")
    except LookaheadError:
        pass


def test_T14_report_completeness_and_dual_track_labels(tmp_path: Path) -> None:
    """T14: Report includes day taxonomy (trading/warm-up/excluded/holiday) & explicit Dual-Track labels."""
    repo_copy = _prepare_locked_repo(tmp_path / "repo", warmup_days=1)
    write_strategy(repo_copy / "strategies", "strat_rep", _SIMPLE_STRAT)
    dirs = make_base_dir(tmp_path / "base")
    specs = [
        UpdlDaySpec(date="2026-11-18", expiry="2026-11-23", lot_size=30),  # warm-up day
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30, session_type="MUHURAT"),  # excluded
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),  # trading day (Fri; Sat/Sun holidays)
        UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30),  # trading day (Mon)
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)

    rc = RUN.main(
        ["strat_rep", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]
    )
    assert rc == RUN.EXIT_OK
    od = list(dirs["output_dir"].iterdir())[0]
    md_text = (od / "REPORT.md").read_text(encoding="utf-8")
    html_text = (od / "REPORT.html").read_text(encoding="utf-8")

    for doc in (md_text, html_text):
        assert "[FIXED 1-LOT]" in doc
        assert "[DYNAMIC COMPOUNDING]" in doc
        assert "Calendar" in doc and "Excluded" in doc and "Warm-up" in doc


def test_T15_paper_account_reconciliation(tmp_path: Path) -> None:
    """T15: base_capital + sum(net) == final_balance (BASE & STRESS) on Dynamic; daily reset on Fixed."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(date="2026-11-18", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(REPO_ROOT, base_capital="50000", overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"})
    ds = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    dates = sorted(ds.days.keys())

    dual = ENG.run_dual_track(days=ds.days, trading_dates=dates, strategy_source=_SIMPLE_STRAT, config=cfg)
    base_cap = Decimal("50000.00")

    # Dynamic reconciliation on BASE & STRESS
    sum_dyn_base = sum((t.net_base for t in dual.dynamic.trades), Decimal("0.00"))
    sum_dyn_stress = sum((t.net_stress for t in dual.dynamic.trades), Decimal("0.00"))
    assert base_cap + sum_dyn_base == dual.dynamic.final_balance_base
    assert base_cap + sum_dyn_stress == dual.dynamic.final_balance_stress

    # Fixed daily reset: every day opens at base_capital
    for dr in dual.fixed.day_results:
        assert dr.day_open_balance == base_cap


def test_T20_parallel_fixed_vs_serial_and_no_pool_queue(tmp_path: Path) -> None:
    """T20: Fixed track workers=1,2,4 bit-exact identical; zero Pool/Queue/Lock/Manager/ProcessPoolExecutor."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(date="2026-11-17", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-18", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(REPO_ROOT, base_capital="50000", overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"})
    ds = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    dates = sorted(ds.days.keys())

    r1, st1 = RUN.run_fixed_parallel(days=ds.days, trading_dates=dates, strategy_source=_SIMPLE_STRAT, config=cfg, workers=1)
    r2, st2 = RUN.run_fixed_parallel(days=ds.days, trading_dates=dates, strategy_source=_SIMPLE_STRAT, config=cfg, workers=2)
    r4, st4 = RUN.run_fixed_parallel(days=ds.days, trading_dates=dates, strategy_source=_SIMPLE_STRAT, config=cfg, workers=4)

    assert st1 == "serial"
    assert st2 in ("fork_pipe", "serial_fallback")
    assert st4 in ("fork_pipe", "serial_fallback")
    assert r1.trades == r2.trades == r4.trades
    assert r1.final_balance_base == r2.final_balance_base == r4.final_balance_base

    # Grep/AST scan across all top-level engine .py modules for forbidden primitives
    forbidden = {
        "Po" + "ol",
        "Qu" + "eue",
        "Lo" + "ck",
        "Man" + "ager",
        "ProcessPool" + "Executor",
    }
    for py_fp in sorted(REPO_ROOT.glob("*.py")):
        tree = ast.parse(py_fp.read_text(encoding="utf-8"), filename=py_fp.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in forbidden, f"Forbidden {node.id} in {py_fp.name}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, f"Forbidden .{node.attr} in {py_fp.name}"
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    assert a.name not in forbidden, f"Forbidden import {a.name} in {py_fp.name}"
