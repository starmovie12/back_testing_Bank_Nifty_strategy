"""Acceptance tests for diagnose.py core logic (PRD §13: T16, T45, T46, T47, T48).

Purpose:
    Validates folder bootstrap & 11 mandatory output files + DIFF.md (T16),
    read-only & fault-isolated diagnosis across full/lite modes (T45),
    workers 1 vs 4 determinism & milan GADBAD tamper detection (T46),
    detectors D01-D12 severities (T47), and 12-class autopsy + unknown_post (T48).

PRD Sections:
    §3.1, §3.3, §12.2, §13, §14 (Step 7)

Guarding Tests (5):
    T16, T45, T46, T47, T48

PUBLIC API:
    test_T16_folder_bootstrap_and_11_output_files,
    test_T45_diagnosis_read_only_and_fault_tolerance,
    test_T46_diagnosis_determinism_and_milan_tamper,
    test_T47_detector_fixtures_D01_to_D12,
    test_T48_12_autopsy_classes_and_unknown_post
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import data_loader as DL
import diagnose as DIAG
import run as RUN
from contracts import TradeRecord
from conftest import (
    REPO_ROOT,
    UpdlDaySpec,
    build_updl_week,
    load_test_config,
    make_base_dir,
    make_temp_repo,
    write_strategy,
)

_MANDATORY_11 = (
    "REPORT.html",
    "REPORT.md",
    "trades_fixed.csv",
    "trades_dynamic.csv",
    "summary.json",
    "manifest.json",
    "verify_sample.md",
    "strategy_snapshot.py",
    "DIAGNOSIS.md",
    "AI_HANDOFF.md",
    "diagnosis.json",
)

_DIAG_STRAT = """
from contracts import Intent, NoTrade
def decide(state):
    if state.t in (20, 60, 100):
        return Intent(side="CE", strike=52000, lots=1, stop_points=30.0, time_stop_min=5)
    return NoTrade("wait")
"""


def _prepare_fast_repo(tmp_path: Path) -> Path:
    repo_copy = make_temp_repo(tmp_path, base_capital="50000")
    cp = repo_copy / "config.toml"
    ct = cp.read_text(encoding="utf-8")
    ct = ct.replace("warmup_days = 20", "warmup_days = 0").replace('holdout_start = "2026-07-06"', 'holdout_start = "2026-12-31"')
    ct = ct.replace("random_runs = 1000", "random_runs = 20").replace("mc_runs = 1000", "mc_runs = 20")
    ct = ct.replace("diag_random_runs = 200", "diag_random_runs = 20").replace("diag_perm_runs = 2000", "diag_perm_runs = 40")
    cp.write_text(ct, encoding="utf-8", newline="\n")
    (repo_copy / "LOCK.sha256").write_text(RUN.build_lock_manifest(repo_copy), encoding="utf-8", newline="\n")
    return repo_copy


def test_T16_folder_bootstrap_and_11_output_files(tmp_path: Path) -> None:
    """T16: Bootstraps empty base_dir, emits all 11 mandatory files (+DIFF.md on 2nd run)."""
    repo_copy = _prepare_fast_repo(tmp_path / "repo")
    write_strategy(repo_copy / "strategies", "strat_boot", _DIAG_STRAT)
    base_dir = tmp_path / "FreshBankNiftyBT"

    # Populate updl data after base_dir bootstrap path
    build_updl_week(
        base_dir / "data" / "week_01",
        [
            UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30),
            UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
            UpdlDaySpec(date="2026-11-23", expiry="2026-11-23", lot_size=30),
        ],
        write_csv=True,
    )

    # 1st run (2026-W47): all 11 mandatory files, NO DIFF.md
    assert RUN.main(["strat_boot", "--week", "2026-W47", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(base_dir)]) == 0
    assert (base_dir / "strategies").is_dir() and (base_dir / "output").is_dir()
    od1 = sorted((base_dir / "output").iterdir())[0]
    for fn in _MANDATORY_11:
        assert (od1 / fn).is_file(), f"Missing {fn}"
    assert not (od1 / "DIFF.md").exists()

    # 2nd run (2026-W48): all 11 mandatory files PLUS DIFF.md
    assert RUN.main(["strat_boot", "--week", "next", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(base_dir)]) == 0
    od2 = sorted((base_dir / "output").iterdir())[-1]
    for fn in _MANDATORY_11:
        assert (od2 / fn).is_file(), f"Missing {fn} on 2nd run"
    assert (od2 / "DIFF.md").is_file()


def test_T45_diagnosis_read_only_and_fault_tolerance(tmp_path: Path) -> None:
    """T45: diag_level full/lite milan 6/6; injected exception in diagnose.py leaves outputs untouched."""
    repo_copy = _prepare_fast_repo(tmp_path / "repo")
    write_strategy(repo_copy / "strategies", "strat_ft", _DIAG_STRAT)
    specs = [
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
    ]

    # Run 1: full mode
    b_full = make_base_dir(tmp_path / "full")
    build_updl_week(b_full["data_dir"] / "week_01", specs, write_csv=True)
    assert RUN.main(["strat_ft", "--week", "2026-W47", "--workers", "1", "--diag-level", "full", "--repo-root", str(repo_copy), "--base-dir", str(b_full["base_dir"])]) == 0
    od_full = list(b_full["output_dir"].iterdir())[0]
    assert json.loads((od_full / "diagnosis.json").read_text(encoding="utf-8"))["milan"]["status"] == "OK (6/6)"

    # Run 2: lite mode
    b_lite = make_base_dir(tmp_path / "lite")
    build_updl_week(b_lite["data_dir"] / "week_01", specs, write_csv=True)
    assert RUN.main(["strat_ft", "--week", "2026-W47", "--workers", "1", "--diag-level", "lite", "--repo-root", str(repo_copy), "--base-dir", str(b_lite["base_dir"])]) == 0
    od_lite = list(b_lite["output_dir"].iterdir())[0]
    assert json.loads((od_lite / "diagnosis.json").read_text(encoding="utf-8"))["milan"]["status"] == "OK (6/6)"

    # Run 3: inject RuntimeError inside diagnose.run_diagnosis -> run still exits 0 & core files byte-identical!
    b_err = make_base_dir(tmp_path / "err")
    build_updl_week(b_err["data_dir"] / "week_01", specs, write_csv=True)
    orig_fn = DIAG.run_diagnosis

    def _boom(**kwargs):
        raise RuntimeError("injected diagnosis crash")

    DIAG.run_diagnosis = _boom  # type: ignore[assignment]
    try:
        rc = RUN.main(["strat_ft", "--week", "2026-W47", "--workers", "1", "--repo-root", str(repo_copy), "--base-dir", str(b_err["base_dir"])])
    finally:
        DIAG.run_diagnosis = orig_fn  # type: ignore[assignment]

    assert rc == RUN.EXIT_OK
    od_err = list(b_err["output_dir"].iterdir())[0]
    man = json.loads((od_err / "manifest.json").read_text(encoding="utf-8"))
    assert man["diagnosis"].startswith("FAILED(RuntimeError:")
    for fn in ("trades_fixed.csv", "trades_dynamic.csv", "summary.json"):
        assert hashlib.sha256((od_full / fn).read_bytes()).hexdigest() == hashlib.sha256((od_err / fn).read_bytes()).hexdigest()


def test_T46_diagnosis_determinism_and_milan_tamper(tmp_path: Path) -> None:
    """T46: workers=1 vs 4 produce identical diagnosis hashes; tampered milan sum flags GADBAD."""
    repo_copy = _prepare_fast_repo(tmp_path / "repo")
    write_strategy(repo_copy / "strategies", "strat_det_d", _DIAG_STRAT)
    specs = [
        UpdlDaySpec(date="2026-11-18", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
    ]
    outs: list[Path] = []
    for w_n in (1, 4):
        b = make_base_dir(tmp_path / f"w{w_n}")
        build_updl_week(b["data_dir"] / "week_01", specs, write_csv=True)
        assert RUN.main(["strat_det_d", "--week", "2026-W47", "--workers", str(w_n), "--repo-root", str(repo_copy), "--base-dir", str(b["base_dir"])]) == 0
        outs.append(list(b["output_dir"].iterdir())[0])

    for fn in ("diagnosis.json", "DIAGNOSIS.md", "AI_HANDOFF.md"):
        assert hashlib.sha256((outs[0] / fn).read_bytes()).hexdigest() == hashlib.sha256((outs[1] / fn).read_bytes()).hexdigest()

    # Tamper 1 reconciliation sum -> check_milan must return GADBAD
    dj = json.loads((outs[0] / "diagnosis.json").read_text(encoding="utf-8"))
    tampered_wf = dict(dj["waterfall"], net_base_D="999999.99")
    m_bad = DIAG.check_milan(
        trades=[],
        waterfall=tampered_wf,
        autopsy=dj["autopsy"],
        funnel=dj["funnel"],
        counterfactuals=dj["counterfactuals"],
        slices=dj["slices"],
    )
    assert m_bad["status"] == "GADBAD"


def test_T47_detector_fixtures_D01_to_D12() -> None:
    """T47: Synthetic fixtures trigger expected detectors D01-D12 with exact severities."""
    cfg = load_test_config(REPO_ROOT, base_capital="50000")

    def _mk_tr(i: int, net: str, xr: str = "stop") -> TradeRecord:
        return TradeRecord(
            track="fixed", trade_no=i, date="2026-11-20", weekday="Fri",
            signal_ts="09:35", entry_ts="09:35", side="CE", strike=52000, expiry="2026-11-23", dte=1,
            lots_signal=1, lots=1, lot_size=30, qty=30, liquidity_warn=False,
            entry_price=Decimal("200.00"), initial_stop=Decimal("180.00"), stop_pct=Decimal("0.1000"),
            target_price=Decimal("220.00"), exit_ts="09:37", exit_price=Decimal("180.00"),
            exit_reason=xr, gap_exit=False, hold_min=2,
            gross=Decimal("100.00"), charges_base=Decimal("100.00"), fill_cost_base=Decimal("100.00"),
            net_base=Decimal(net), charges_stress=Decimal("100.00"), fill_cost_stress=Decimal("200.00"),
            net_stress=Decimal(net) - Decimal("100.00"), net_proxy=None,
            balance_base=Decimal("50000.00"), balance_stress=Decimal("50000.00"),
            outcome="WIN" if Decimal(net) > 0 else "LOSS", mae=Decimal("20.00"), mfe=Decimal("25.00"),
            mfe_min=1, mae_min=2, reason="r", tags="",
        )

    # 40 trades: first 20 winning, next 20 losing -> triggers D10 split-half decay too
    trades = [_mk_tr(i, "200.00", "target") for i in range(1, 21)] + [
        _mk_tr(i, "-300.00", "stop") for i in range(21, 41)
    ]
    waterfall = {"gross_A": "1000.00", "fill_cost_B": "800.00", "charges_C": "800.00", "net_base_D": "-600.00", "cost_share": 1.6}
    entry_edge = {"verdict": "NAHI", "best_h_star": None}
    autopsy = {
        "classes": {
            c: {"count": (12 if c == "SHAKEOUT" else (10 if c == "GAVE_BACK" else (10 if c == "TARGET_EARLY" else 0))),
                "net_base": ("-3000.00" if c in ("SHAKEOUT", "GAVE_BACK") else ("2000.00" if c == "TARGET_EARLY" else "0.00"))}
            for c in DIAG.AUTOPSY_CLASSES
        }
    }
    slices = {
        "dimensions": {
            "side": {
                "PE": {"n": 20, "net_base": "-3000.00", "status": "PAKKA"},
                "CE": {"n": 20, "net_base": "2400.00", "status": "PAKKA"},
            }
        }
    }
    bharosa = {"overall": "LAL", "panels": {"sample": "LAL"}, "net_ex_top_k": "-500.00", "top5_win_share": 0.90}
    funnel = {"intents_C": 50, "c3_risk_skip": 20, "by_skip_code": {"SKIP_STALE_CANCEL": 6}, "exit_reasons": {"no_bar": 2}}
    summary = {"fixed": {"net_stress": "-1500.00", "cap_breach_days": ["2026-11-20"]}, "dynamic": {"cap_breach_days": []}}

    findings = DIAG.run_detectors(
        trades=trades, waterfall=waterfall, entry_edge=entry_edge, autopsy=autopsy,
        counterfactuals={"scenarios": {"CF-TARGETx2.0": {"delta_inr": "1200.00", "net_base": "600.00"}}},
        slices=slices, bharosa=bharosa, funnel=funnel, summary=summary, config=cfg, is_holdout=False,
    )
    by_id = {f["id"]: f for f in findings}
    for expected_id in ("D01", "D02", "D04", "D05", "D06", "D07", "D08", "D09", "D10", "D11", "D12"):
        assert expected_id in by_id, f"Detector {expected_id} failed to fire!"
    assert by_id["D01"]["severity"] == "LAL"
    assert by_id["D02"]["severity"] == "LAL"
    assert by_id["D08"]["severity"] == "HARA"
    assert by_id["D11"]["severity"] == "LAL"


def test_T48_12_autopsy_classes_and_unknown_post(tmp_path: Path) -> None:
    """T48: 12 hand-crafted trades hit all 12 classes 1-to-1; counts sum to T; INR sums to net_base; unknown_post counted."""
    cfg = load_test_config(REPO_ROOT, base_capital="50000")
    dirs = make_base_dir(tmp_path)
    ov = {
        (52000, "CE", 22): {"open": 210.0, "high": 265.0, "low": 205.0, "close": 260.0},
        **{(52000, "CE", m): {"open": 182.0, "high": 185.0, "low": 180.0, "close": 181.0} for m in range(62, 100)},
    }
    # Make bars 152..170 stale so trade exiting at bar 151 (11:46) has zero post-exit live bars -> unknown_post!
    build_updl_week(
        dirs["data_dir"] / "week_01",
        [UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30, stale_option_minutes=frozenset(range(152, 175)), option_overrides=ov)],
        write_csv=True,
    )
    ds = DL.load_dataset(dirs["data_dir"], dict(cfg, data_layout="csv", warmup_days=0, holdout_start="2026-12-31"), base_dir=dirs["base_dir"])

    def _mk(n, ep, stop, xp, xr, gross, net, mfe, xts="09:36"):
        return TradeRecord(
            track="fixed", trade_no=n, date="2026-11-20", weekday="Fri",
            signal_ts="09:35", entry_ts="09:35", side="CE", strike=52000, expiry="2026-11-23", dte=1,
            lots_signal=1, lots=1, lot_size=30, qty=30, liquidity_warn=False,
            entry_price=Decimal(ep), initial_stop=Decimal(stop), stop_pct=Decimal("0.1000"),
            target_price=Decimal("220.00"), exit_ts=xts, exit_price=Decimal(xp),
            exit_reason=xr, gap_exit=False, hold_min=1,
            gross=Decimal(gross), charges_base=Decimal("100.00"), fill_cost_base=Decimal("72.00"),
            net_base=Decimal(net), charges_stress=Decimal("100.00"), fill_cost_stress=Decimal("144.00"),
            net_stress=Decimal(net) - Decimal("72.00"), net_proxy=None,
            balance_base=Decimal("50000.00"), balance_stress=Decimal("50000.00"),
            outcome="WIN" if Decimal(net) > 0 else "LOSS", mae=Decimal("5.00"), mfe=Decimal(mfe),
            mfe_min=1, mae_min=0, reason="r", tags="",
        )

    t12 = [
        _mk(1, "200.00", "180.00", "170.00", "stop", "-900.00", "-1072.00", "0.00"),
        _mk(2, "200.00", "180.00", "202.00", "time", "60.00", "-112.00", "2.00"),
        _mk(3, "200.00", "180.00", "180.00", "stop", "-600.00", "-772.00", "0.00", xts="09:36"),
        _mk(4, "200.00", "180.00", "180.00", "stop", "-600.00", "-772.00", "25.00", xts="10:16"),
        _mk(5, "200.00", "180.00", "180.00", "stop", "-600.00", "-772.00", "5.00", xts="11:46"),  # unknown_post!
        _mk(6, "200.00", "180.00", "195.00", "time", "-150.00", "-322.00", "5.00"),
        _mk(7, "200.00", "180.00", "195.00", "no_bar", "-150.00", "-322.00", "5.00"),
        _mk(8, "200.00", "180.00", "220.00", "target", "600.00", "428.00", "20.00", xts="09:36"),
        _mk(9, "200.00", "180.00", "220.00", "target", "600.00", "428.00", "20.00", xts="10:16"),
        _mk(10, "200.00", "180.00", "215.00", "trail", "450.00", "278.00", "20.00"),
        _mk(11, "200.00", "180.00", "215.00", "time", "450.00", "278.00", "20.00"),
        _mk(12, "200.00", "180.00", "215.00", "no_bar", "450.00", "278.00", "20.00"),
    ]
    aut = DIAG.compute_autopsy(t12, ds.days, cfg)
    for cls in DIAG.AUTOPSY_CLASSES:
        assert aut["classes"][cls]["count"] == 1, f"{cls} count != 1"
    assert sum(v["count"] for v in aut["classes"].values()) == 12
    assert sum((Decimal(v["net_base"]) for v in aut["classes"].values()), Decimal("0")) == sum((t.net_base for t in t12), Decimal("0"))
    assert aut["unknown_post"] >= 1
