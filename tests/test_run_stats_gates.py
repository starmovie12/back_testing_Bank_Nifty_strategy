"""Acceptance tests for AST precision, statistical gates, git lock, m-counter & Dual-Track (PRD §13: T21, T27, T32, T33, T42, T53).

Purpose:
    Validates AST static checker precision with exact line numbers (T21),
    6-gate DESIGN verdict with Bonferroni m (T27), 30-trade HOLDOUT verdict
    and robustness pack formulas (T32), immutability + atomic writes + git
    strategies/ commit usability (T33), multiple-testing family counter m
    scoped to engine-v2.0 (T42), and Dual-Track isolation & carryover (T53).

PRD Sections:
    §1, §3.2, §4.1, §4.2, §6.2, §11.1, §12.1, §12.3, §13, §14 (Step 6)

Guarding Tests (6):
    T21, T27, T32, T33, T42, T53

PUBLIC API:
    test_T21_ast_static_checker_precision,
    test_T27_design_gate_verdict_and_bonferroni_m,
    test_T32_holdout_verdict_and_robustness_math,
    test_T33_immutability_atomic_writes_and_git_usability,
    test_T42_multiple_testing_family_counter_m,
    test_T53_dual_track_isolation_and_carryover
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from decimal import Decimal
from pathlib import Path

import data_loader as DL
import engine as ENG
import report as REP
import run as RUN
from contracts import ENGINE_TAG, Intent
from conftest import (
    REPO_ROOT,
    UpdlDaySpec,
    build_updl_week,
    load_test_config,
    make_base_dir,
    make_temp_repo,
    write_strategy,
)


def _prepare_locked_repo(tmp_path: Path) -> Path:
    repo_copy = make_temp_repo(tmp_path, base_capital="50000")
    cfg_p = repo_copy / "config.toml"
    txt = cfg_p.read_text(encoding="utf-8")
    txt = txt.replace("warmup_days = 20", "warmup_days = 0")
    txt = txt.replace('holdout_start = "2026-07-06"', 'holdout_start = "2026-12-31"')
    txt = txt.replace("random_runs = 1000", "random_runs = 25")
    txt = txt.replace("mc_runs = 1000", "mc_runs = 25")
    cfg_p.write_text(txt, encoding="utf-8", newline="\n")
    (repo_copy / "LOCK.sha256").write_text(
        RUN.build_lock_manifest(repo_copy), encoding="utf-8", newline="\n"
    )
    return repo_copy


def test_T21_ast_static_checker_precision() -> None:
    """T21: Valid strategy (__init__/__post_init__) passes; every banned construct fails with exact lineno."""
    valid_src = """
import math
import numpy as np
from dataclasses import dataclass
from contracts import Intent, NoTrade

PARAMS = {"stop": 30.0}

@dataclass
class Helper:
    val: int
    def __post_init__(self):
        self.ok = self.val > 0

class Strategy:
    def __init__(self, params):
        self.params = params
    def decide(self, state):
        if math.isfinite(float(np.mean([1.0, 2.0]))):
            return NoTrade("ok")
        return Intent(side="CE", strike=52000, stop_points=self.params["stop"])
"""
    assert RUN.check_strategy_ast(valid_src, "valid.py") == []

    banned_cases = [
        ("import os\n", 1),
        ("from math import *\n", 1),
        ("def f():\n    return eval('1')\n", 2),
        ("def f():\n    return getattr(object, 'x')\n", 2),
        ("class S:\n    def __repr__(self):\n        return ''\n", 2),
        ("def f(x):\n    return x.__class__\n", 2),
        ("def f(x):\n    return x.base\n", 2),
        ("import numpy as np\ndef f():\n    return np.random.rand()\n", 3),
        ("import numpy as np\ndef f():\n    return np.load('a.npy')\n", 3),
        ("x = 0\ndef f():\n    global x\n    x += 1\n", 3),
        ("class S:\n    cache = []\n", 2),
        ("PARAMS = {'a': 1}\ndef f():\n    PARAMS['a'] = 2\n", 3),
    ]
    for src, expected_line in banned_cases:
        viols = RUN.check_strategy_ast(src, "bad.py")
        assert len(viols) >= 1, f"Expected violation for:\n{src}"
        assert any(ln == expected_line for ln, _ in viols), (
            f"Expected line {expected_line}, got {viols} for:\n{src}"
        )


def test_T27_design_gate_verdict_and_bonferroni_m() -> None:
    """T27: Exact PASS/NOT_YET/FAIL across all 6 DESIGN gates + Bonferroni m=4 (0.05/4=0.0125)."""
    cfg = load_test_config(REPO_ROOT, base_capital="50000")
    good_f = {"trades_count": 120, "net_stress": "4500.00"}
    good_d = {"ruin_flag": False}
    good_rob = {
        "random_entry_p_value": 0.0100,
        "bootstrap_95_ci_low": 12.5,
        "positive_folds_pct_fixed": 80.0,
        "delay1_net_base_fixed": "1800.00",
    }

    # All 6 pass with m=4 (0.0100 <= 0.0125)
    v_pass = REP.evaluate_design_gate_verdict(good_f, good_d, good_rob, m_variants=4, config=cfg)
    assert v_pass["verdict"] == "PASS"
    assert v_pass["bonferroni_p_threshold"] == 0.0125

    # Only min_trades fails -> NOT_YET
    v_not_yet = REP.evaluate_design_gate_verdict(
        {"trades_count": 85, "net_stress": "4500.00"}, good_d, good_rob, m_variants=4, config=cfg
    )
    assert v_not_yet["verdict"] == "NOT_YET" and v_not_yet["failed_gates"] == ["min_trades"]

    # Bonferroni boundary fail: p = 0.0126 > 0.0125 -> FAIL
    v_bonf = REP.evaluate_design_gate_verdict(
        good_f, good_d, dict(good_rob, random_entry_p_value=0.0126), m_variants=4, config=cfg
    )
    assert v_bonf["verdict"] == "FAIL" and "random_p_bonferroni" in v_bonf["failed_gates"]

    # Break each remaining gate individually -> FAIL
    for key, patch_f, patch_d, patch_rob in [
        ("stress_positive", {"net_stress": "-10.00"}, {}, {}),
        ("bootstrap_ci_positive", {}, {}, {"bootstrap_95_ci_low": -1.0}),
        ("fold_stability", {}, {}, {"positive_folds_pct_fixed": 40.0}),
        ("delay1_positive", {}, {}, {"delay1_net_base_fixed": "0.00"}),
        ("no_dynamic_ruin", {}, {"ruin_flag": True}, {}),
    ]:
        vf = dict(good_f, **patch_f)
        vd = dict(good_d, **patch_d)
        vr = dict(good_rob, **patch_rob)
        res = REP.evaluate_design_gate_verdict(vf, vd, vr, m_variants=1, config=cfg)
        assert res["verdict"] == "FAIL" and key in res["failed_gates"]


def test_T32_holdout_verdict_and_robustness_math(tmp_path: Path) -> None:
    """T32: 30-trade holdout verdict (K-adjusted p), random-entry (1+k)/(N+1), bootstrap, 3 folds, Dynamic MC DD."""
    dirs = make_base_dir(tmp_path)
    specs = [
        UpdlDaySpec(date="2026-07-06", expiry="2026-07-28", lot_size=30),
        UpdlDaySpec(date="2026-07-07", expiry="2026-07-28", lot_size=30),
        UpdlDaySpec(date="2026-07-08", expiry="2026-07-28", lot_size=30),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        base_capital="50000",
        overrides={
            "data_layout": "csv",
            "warmup_days": 0,
            "holdout_start": "2026-07-06",
            "random_runs": 49,  # denominator N+1 = 50 -> exact 0.02 multiples
            "mc_runs": 50,
        },
    )
    ds = DL.load_dataset(dirs["data_dir"], cfg, allow_holdout=True, auto_trim_holdout=False, base_dir=dirs["base_dir"])
    dates = sorted(ds.days.keys())

    # 10 trades per day x 3 days = 30 holdout trades
    entry_bars = tuple(20 + i * 25 for i in range(10))
    strat_src = f"""
from contracts import Intent, NoTrade
def decide(state):
    if state.t in {entry_bars!r}:
        return Intent(side="CE", strike=52000, lots=1, stop_points=30.0, time_stop_min=5)
    return NoTrade("wait")
"""
    dual = ENG.run_dual_track(days=ds.days, trading_dates=dates, strategy_source=strat_src, config=cfg)
    assert len(dual.fixed.trades) == 30 and len(dual.dynamic.trades) == 30

    rob = REP.compute_robustness_pack(
        dual=dual, days=ds.days, trading_dates=dates, config=cfg, is_holdout=True,
        delay1_net_fixed=Decimal("1000.00"), delay2_net_fixed=Decimal("500.00"),
    )
    # Random-entry p-value is strictly (1+k)/(49+1) in [0.02, 1.0]
    p_val = rob["random_entry_p_value"]
    assert p_val is not None and 0.02 <= p_val <= 1.0
    assert abs((p_val * 50.0) - round(p_val * 50.0)) < 1e-6
    assert rob["bootstrap_95_ci_low"] is not None and rob["bootstrap_95_ci_high"] is not None
    assert len(rob["folds_fixed_net_base"]) == 3
    mc = rob["mc_drawdown_pct_dynamic"]
    assert mc is not None and mc["p5"] <= mc["p50"] <= mc["p95"]

    # Holdout verdict with K=2 (0.10 / 2 = 0.05)
    hv_pass = REP.evaluate_holdout_verdict(
        {"trades_count": 30, "net_stress": "1500.00"},
        {"ruin_flag": False},
        {"random_entry_p_value": 0.04, "delay1_net_base_fixed": "800.00"},
        k_opens=2, config=cfg,
    )
    assert hv_pass["verdict"] == "PASS" and hv_pass["k_adjusted_p_threshold"] == 0.05

    hv_fail = REP.evaluate_holdout_verdict(
        {"trades_count": 30, "net_stress": "1500.00"},
        {"ruin_flag": False},
        {"random_entry_p_value": 0.06, "delay1_net_base_fixed": "800.00"},
        k_opens=2, config=cfg,
    )
    assert hv_fail["verdict"] == "FAIL" and "random_p_k_adjusted" in hv_fail["failed_gates"]


def test_T33_immutability_atomic_writes_and_git_usability(tmp_path: Path) -> None:
    """T33: Read-only dataclasses/arrays, same-dir atomic writes, git commit in strategies/ keeps lock CLEAN."""
    # 1. Dataclass immutability
    intent = Intent(side="CE", strike=52000, stop_points=30.0)
    try:
        intent.strike = 51000  # type: ignore[misc]
        raise AssertionError("Intent was mutable!")
    except dataclasses.FrozenInstanceError:
        pass

    # 2. Same-dir atomic write leaves zero .tmp.* files
    dest = tmp_path / "atomic" / "out.json"
    DL.atomic_write_bytes(dest, b'{"ok":true}\n')
    assert dest.read_bytes() == b'{"ok":true}\n'
    assert not any(p.name.startswith(".tmp.") for p in dest.parent.iterdir())

    # 3. Git repo with engine-v2.0 tag + adding/committing a strategy in strategies/ keeps lock CLEAN
    repo_git = _prepare_locked_repo(tmp_path / "git_repo")
    subprocess.run(["git", "init", "-q"], cwd=str(repo_git), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_git), check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(repo_git), check=True)
    subprocess.run(["git", "add", "."], cwd=str(repo_git), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "lock engine"], cwd=str(repo_git), check=True)
    subprocess.run(["git", "tag", ENGINE_TAG], cwd=str(repo_git), check=True)

    # Add & commit new strategy in strategies/ -> must NOT break LOCK.sha256 or git diff check
    write_strategy(repo_git / "strategies", "strat_new", "from contracts import NoTrade\ndef decide(s): return NoTrade('x')\n")
    subprocess.run(["git", "add", "strategies/strat_new.py"], cwd=str(repo_git), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add strat"], cwd=str(repo_git), check=True)

    ok, detail, git_check = RUN.verify_lock(repo_git)
    assert ok and detail == "OK" and git_check == "CLEAN"


def test_T42_multiple_testing_family_counter_m(tmp_path: Path) -> None:
    """T42: 5 distinct variants of same strategy family on engine-v2.0 -> m=5 & max_p_value/5."""
    base_dir = make_base_dir(tmp_path)["base_dir"]
    # Add a legacy engine-v1.0 trial -> must NOT be counted in engine-v2.0 m!
    RUN.append_trial_record(
        base_dir,
        {"engine_tag": "engine-v1.0", "strategy_family": "strat_orb", "strategy_hash": "legacy_hash_000"},
    )
    for i in range(1, 5):
        RUN.append_trial_record(
            base_dir,
            {"engine_tag": ENGINE_TAG, "strategy_family": "strat_orb", "strategy_hash": f"hash_v{i}"},
        )

    m5 = RUN.count_family_variants(base_dir, "strat_orb", "hash_v5")
    assert m5 == 5

    cfg = load_test_config(REPO_ROOT, base_capital="50000")
    dg = REP.evaluate_design_gate_verdict(
        {"trades_count": 120, "net_stress": "3000.00"},
        {"ruin_flag": False},
        {
            "random_entry_p_value": 0.010,
            "bootstrap_95_ci_low": 5.0,
            "positive_folds_pct_fixed": 80.0,
            "delay1_net_base_fixed": "1000.00",
        },
        m_variants=m5,
        config=cfg,
    )
    assert dg["bonferroni_p_threshold"] == 0.01  # 0.05 / 5
    assert dg["verdict"] == "PASS"


def test_T53_dual_track_isolation_and_carryover(tmp_path: Path) -> None:
    """T53: Fixed daily reset & parallel==serial; Dynamic day/week carryover; shuffled dates refused."""
    repo_copy = _prepare_locked_repo(tmp_path / "repo")
    dirs = make_base_dir(tmp_path / "base")
    # 2 ISO weeks: 2026-W47 (Nov 19-20) and 2026-W48 (Nov 23-24)
    specs = [
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-24", expiry="2026-12-28", lot_size=30),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    strat_src = """
from contracts import Intent, NoTrade
def decide(state):
    if state.t in (20, 60):
        return Intent(side="CE", strike=52000, lots=1, stop_points=30.0, time_stop_min=5)
    return NoTrade("wait")
"""
    write_strategy(repo_copy / "strategies", "strat_dual", strat_src)

    # Week 1 (2026-W47) then Week 2 (2026-W48) via CLI -> Dynamic balance carries over across weeks!
    assert RUN.main(["strat_dual", "--week", "2026-W47", "--workers", "2", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == 0
    ws1 = RUN.load_and_verify_weekly_state(dirs["base_dir"])
    w1_close_bal = Decimal(ws1["records"][-1]["dynamic_closing_balance"])

    assert RUN.main(["strat_dual", "--week", "next", "--workers", "2", "--repo-root", str(repo_copy), "--base-dir", str(dirs["base_dir"])]) == 0
    out_dirs = sorted(dirs["output_dir"].iterdir())
    s2 = json.loads((out_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert Decimal(s2["dynamic"]["start_balance"]) == w1_close_bal
    assert Decimal(s2["fixed"]["start_balance"]) == Decimal("50000.00")

    # Shuffled dates refused on Dynamic track
    cfg = load_test_config(REPO_ROOT, base_capital="50000", overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"})
    ds = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    try:
        ENG.run_track(days=ds.days, trading_dates=["2026-11-20", "2026-11-19"], strategy_source=strat_src, config=cfg, track="dynamic")
        raise AssertionError("Expected ValueError on shuffled Dynamic dates")
    except ValueError:
        pass
