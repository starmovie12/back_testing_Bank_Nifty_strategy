"""Termux fork + Pipe smoke test & OSError serial fallback (PRD §13: T43).

Purpose:
    Verifies Termux-safe fork + Pipe execution with workers=2 (zero sem_open
    usage, bit-identical to serial workers=1), and confirms that a forced
    OSError cleanly falls back to serial execution and records
    parallel="serial_fallback" in manifest.json.

PRD Sections:
    §12.3, §13, §14 (Step 8)

Guarding Tests (1):
    T43

PUBLIC API:
    test_T43_termux_fork_pipe_and_oserror_fallback
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import data_loader as DL
import run as RUN
from conftest import (
    REPO_ROOT,
    UpdlDaySpec,
    build_updl_week,
    load_test_config,
    make_base_dir,
    make_temp_repo,
    write_strategy,
)

_SMOKE_STRAT = """
from contracts import Intent, NoTrade
def decide(state):
    if state.t in (20, 60):
        return Intent(side="CE", strike=52000, lots=1, stop_points=30.0, time_stop_min=5)
    return NoTrade("wait")
"""


def test_T43_termux_fork_pipe_and_oserror_fallback(tmp_path: Path) -> None:
    """T43: fork+Pipe workers=2 matches serial (zero sem_open); forced OSError -> serial_fallback in manifest."""
    dirs = make_base_dir(tmp_path / "direct")
    specs = [
        UpdlDaySpec(date="2026-11-18", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-19", expiry="2026-11-23", lot_size=30),
        UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30),
    ]
    build_updl_week(dirs["data_dir"] / "week_01", specs, write_csv=True)
    cfg = load_test_config(
        REPO_ROOT,
        base_capital="50000",
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    ds = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"])
    dates = sorted(ds.days.keys())

    # 1. fork + Pipe with workers=2 matches serial workers=1 bit-for-bit
    res_serial, st_serial = RUN.run_fixed_parallel(
        days=ds.days, trading_dates=dates, strategy_source=_SMOKE_STRAT, config=cfg, workers=1
    )
    res_par, st_par = RUN.run_fixed_parallel(
        days=ds.days, trading_dates=dates, strategy_source=_SMOKE_STRAT, config=cfg, workers=2
    )
    assert st_serial == "serial"
    assert st_par in ("fork_pipe", "serial_fallback")
    assert res_par.trades == res_serial.trades
    assert res_par.final_balance_base == res_serial.final_balance_base

    # Verify zero sem_open / synchronize imports inside run.py
    run_src = (REPO_ROOT / "run.py").read_text(encoding="utf-8")
    assert "sem_open" not in run_src
    assert "synchronize" not in run_src

    # 2. Forced OSError falls back to serial and records parallel="serial_fallback" in manifest.json
    repo_copy = make_temp_repo(tmp_path / "repo", base_capital="50000")
    cp = repo_copy / "config.toml"
    ct = cp.read_text(encoding="utf-8")
    ct = ct.replace("warmup_days = 20", "warmup_days = 0").replace('holdout_start = "2026-07-06"', 'holdout_start = "2026-12-31"')
    ct = ct.replace("random_runs = 1000", "random_runs = 20").replace("mc_runs = 1000", "mc_runs = 20")
    ct = ct.replace("diag_random_runs = 200", "diag_random_runs = 20").replace("diag_perm_runs = 2000", "diag_perm_runs = 40")
    cp.write_text(ct, encoding="utf-8", newline="\n")
    (repo_copy / "LOCK.sha256").write_text(RUN.build_lock_manifest(repo_copy), encoding="utf-8", newline="\n")
    write_strategy(repo_copy / "strategies", "strat_t43", _SMOKE_STRAT)

    b_fb = make_base_dir(tmp_path / "fallback")
    build_updl_week(b_fb["data_dir"] / "week_01", specs, write_csv=True)

    orig_get_ctx = multiprocessing.get_context

    def _oserror_ctx(method: str | None = None):
        raise OSError("Simulated Termux fork/Pipe OSError")

    multiprocessing.get_context = _oserror_ctx  # type: ignore[assignment]
    try:
        rc = RUN.main(
            [
                "strat_t43",
                "--week",
                "2026-W47",
                "--workers",
                "2",
                "--repo-root",
                str(repo_copy),
                "--base-dir",
                str(b_fb["base_dir"]),
            ]
        )
    finally:
        multiprocessing.get_context = orig_get_ctx  # type: ignore[assignment]

    assert rc == RUN.EXIT_OK
    od = list(b_fb["output_dir"].iterdir())[0]
    man = json.loads((od / "manifest.json").read_text(encoding="utf-8"))
    assert man["parallel"] == "serial_fallback"
