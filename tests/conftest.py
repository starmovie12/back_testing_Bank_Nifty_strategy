"""Shared pytest fixtures and deterministic synthetic updl dataset builders.

Purpose:
    Provides repo-root sys.path setup, deterministic CSV/Parquet updl week
    builders with controllable market anomalies (stale, filled, zero-volume,
    gaps, expiry day, lot-size revisions, special sessions, duplicate/reversed
    timestamps, bad OHLC, late avail_ts, optional VIX/futures/greeks/stocks),
    tmp base_dir and synthetic strategy helpers, and a temporary locked-repo
    clone builder for end-to-end tests.

PRD Sections:
    §3.1, §3.2, §10.1, §10.2, §13

Niyam Covered:
    - Test isolation: all generated data, strategies, and repo copies live
      strictly inside pytest tmp_path; the real repo tree is never modified.

Guarding Tests:
    Used by T01..T62 across tests/test_*.py

PUBLIC API:
    Constants:
        REPO_ROOT, HAS_PYARROW
    Dataclasses:
        UpdlDaySpec
    Functions:
        load_test_config, make_base_dir, write_strategy, build_updl_week,
        make_temp_repo
    Fixtures:
        tmp_base_dir, test_config
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

# Ensure repository root is importable regardless of working directory
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_loader as DL  # noqa: E402

try:
    import pyarrow as _pa  # type: ignore[import-not-found]
    import pyarrow.parquet as _pq  # type: ignore[import-not-found]

    HAS_PYARROW: bool = True
except ImportError:
    _pa = None
    _pq = None
    HAS_PYARROW = False

try:
    import pytest as _pytest  # type: ignore[import-not-found]
except ImportError:
    _pytest = None


def _fixture(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Apply @pytest.fixture when pytest is installed; otherwise return fn unchanged."""
    if _pytest is not None:
        return _pytest.fixture(fn)
    return fn


# ------------------------------------------------------------------------------
# Synthetic updl Day Specification & Builder (§10.1, §10.2, §13)
# ------------------------------------------------------------------------------
@dataclass
class UpdlDaySpec:
    """Declarative specification for one synthetic updl market day."""

    date: str
    expiry: str
    lot_size: int | None = 30
    spot_open: float = 52000.0
    spot_step: float = 0.2
    strikes: tuple[int, ...] = (51500, 52000, 52500)
    option_base_price: float = 200.0
    n_index_bars: int = DL.STANDARD_INDEX_MINUTES
    n_fno_bars: int = DL.STANDARD_INDEX_MINUTES
    session_type: str = "NORMAL"
    stale_option_minutes: frozenset[int] = field(default_factory=frozenset)
    filled_option_minutes: frozenset[int] = field(default_factory=frozenset)
    zero_vol_option_minutes: frozenset[int] = field(default_factory=frozenset)
    gap_index_minutes: frozenset[int] = field(default_factory=frozenset)
    gap_option_minutes: frozenset[int] = field(default_factory=frozenset)
    late_avail_option_minutes: dict[int, int] = field(default_factory=dict)
    duplicate_index_minutes: frozenset[int] = field(default_factory=frozenset)
    reverse_index_pair: tuple[int, int] | None = None
    bad_ohlc_index_minutes: dict[int, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    bad_ohlc_option_minutes: dict[int, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    option_overrides: dict[tuple[int, str, int], dict[str, Any]] = field(
        default_factory=dict
    )
    include_vix: bool = False
    include_futures: bool = False
    include_greeks: bool = False
    include_stocks: tuple[str, ...] = ()
    split_csv_from_date: str | None = None
    omit_options: bool = False


def _write_csv(path: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 CSV with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write Parquet table using pyarrow (caller must verify HAS_PYARROW)."""
    if not HAS_PYARROW or _pa is None or _pq is None:
        raise RuntimeError("pyarrow is required to write Parquet test fixtures")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _pa.Table.from_pylist(rows)
    _pq.write_table(table, path)


def build_updl_week(
    week_dir: Path,
    day_specs: list[UpdlDaySpec],
    *,
    write_csv: bool = True,
    write_parquet: bool = False,
) -> Path:
    """Materialize a deterministic updl week folder (CSV and/or Parquet) from UpdlDaySpecs."""
    if write_parquet and not HAS_PYARROW:
        raise RuntimeError("write_parquet=True requested but pyarrow is not installed")

    spot_rows: list[dict[str, Any]] = []
    vix_rows: list[dict[str, Any]] = []
    fut_rows: list[dict[str, Any]] = []
    greeks_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    opt_rows_all: list[dict[str, Any]] = []
    opt_rows_by_file: dict[str, list[dict[str, Any]]] = {}
    contracts_map: dict[str, int | None] = {}

    for spec in day_specs:
        if spec.expiry:
            contracts_map[spec.expiry] = spec.lot_size

        # Build index rows
        day_spot: list[dict[str, Any]] = []
        for m in range(spec.n_index_bars):
            if m in spec.gap_index_minutes:
                continue
            hhmm = DL.minute_to_hhmm(m)
            ts = f"{spec.date} {hhmm}:00+05:30"
            c = round(spec.spot_open + m * spec.spot_step, 2)
            o, h, l, cl = round(c - 1.0, 2), round(c + 2.0, 2), round(c - 2.0, 2), c
            if m in spec.bad_ohlc_index_minutes:
                o, h, l, cl = spec.bad_ohlc_index_minutes[m]
            row = {
                "ts_open": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": cl,
                "volume": 1000.0,
                "oi": 0.0,
                "is_stale": False,
                "is_filled": False,
                "available_from_ts": "",
                "session_type": spec.session_type,
            }
            day_spot.append(row)
            if m in spec.duplicate_index_minutes:
                day_spot.append(dict(row))

        if spec.reverse_index_pair is not None:
            i_a, i_b = spec.reverse_index_pair
            if 0 <= i_a < len(day_spot) and 0 <= i_b < len(day_spot):
                day_spot[i_a], day_spot[i_b] = day_spot[i_b], day_spot[i_a]
        spot_rows.extend(day_spot)

        # Optional VIX rows
        if spec.include_vix:
            for m in range(spec.n_index_bars):
                if m in spec.gap_index_minutes:
                    continue
                ts = f"{spec.date} {DL.minute_to_hhmm(m)}:00+05:30"
                v_val = round(14.0 + (m % 10) * 0.05, 2)
                vix_rows.append(
                    {
                        "ts_open": ts,
                        "open": v_val,
                        "high": round(v_val + 0.1, 2),
                        "low": round(v_val - 0.1, 2),
                        "close": v_val,
                        "volume": 0.0,
                        "oi": 0.0,
                        "is_stale": False,
                        "is_filled": False,
                        "available_from_ts": "",
                        "session_type": spec.session_type,
                    }
                )

        # Optional Futures rows
        if spec.include_futures:
            for m in range(spec.n_fno_bars):
                ts = f"{spec.date} {DL.minute_to_hhmm(m)}:00+05:30"
                fp = round(spec.spot_open + 50.0 + m * spec.spot_step, 2)
                fut_rows.append(
                    {
                        "ts_open": ts,
                        "open": round(fp - 1.0, 2),
                        "high": round(fp + 2.0, 2),
                        "low": round(fp - 2.0, 2),
                        "close": fp,
                        "volume": 500.0,
                        "oi": 50000.0 + m,
                        "is_stale": False,
                        "is_filled": False,
                        "available_from_ts": "",
                        "session_type": spec.session_type,
                    }
                )

        # Optional constituent stock rows
        for sym in spec.include_stocks:
            for m in range(spec.n_index_bars):
                ts = f"{spec.date} {DL.minute_to_hhmm(m)}:00+05:30"
                sp = round(1500.0 + m * 0.1, 2)
                stock_rows.append(
                    {
                        "symbol": sym,
                        "ts_open": ts,
                        "open": sp,
                        "high": round(sp + 1.0, 2),
                        "low": round(sp - 1.0, 2),
                        "close": sp,
                        "volume": 2000.0,
                        "oi": 0.0,
                        "is_stale": False,
                        "is_filled": False,
                        "available_from_ts": "",
                        "session_type": spec.session_type,
                    }
                )

        # Option rows (+ optional iv_greeks_1m)
        if not spec.omit_options:
            csv_stem = (
                f"{spec.expiry}_from_{spec.split_csv_from_date}.csv"
                if spec.split_csv_from_date
                else f"{spec.expiry}.csv"
            )
            file_bucket = opt_rows_by_file.setdefault(csv_stem, [])
            for strike in spec.strikes:
                for side in ("CE", "PE"):
                    last_o = spec.option_base_price
                    last_h = spec.option_base_price + 5.0
                    last_l = spec.option_base_price - 5.0
                    last_c = spec.option_base_price + 1.0
                    last_oi = 10000.0
                    for m in range(spec.n_fno_bars):
                        if m in spec.gap_option_minutes:
                            continue
                        ts = f"{spec.date} {DL.minute_to_hhmm(m)}:00+05:30"
                        base_p = round(spec.option_base_price + m * 0.1, 2)
                        is_st = m in spec.stale_option_minutes
                        is_fl = m in spec.filled_option_minutes
                        is_zv = m in spec.zero_vol_option_minutes
                        if is_st or is_fl or is_zv:
                            # Copy-forward prior OHLC & OI with volume=0 (T37)
                            o, h, l, c = last_o, last_h, last_l, last_c
                            vol = 0.0
                            oi_val = last_oi
                        else:
                            o = base_p
                            h = round(base_p + 5.0, 2)
                            l = round(base_p - 5.0, 2)
                            c = round(base_p + 1.0, 2)
                            vol = 500.0
                            oi_val = 10000.0 + m
                            last_o, last_h, last_l, last_c, last_oi = o, h, l, c, oi_val

                        if m in spec.bad_ohlc_option_minutes:
                            o, h, l, c = spec.bad_ohlc_option_minutes[m]

                        avail_str = ""
                        if m in spec.late_avail_option_minutes:
                            av_off = spec.late_avail_option_minutes[m]
                            avail_str = f"{spec.date} {DL.minute_to_hhmm(av_off)}:01+05:30"

                        row = {
                            "ts_open": ts,
                            "expiry": spec.expiry,
                            "strike": strike,
                            "option_type": side,
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": c,
                            "volume": vol,
                            "oi": oi_val,
                            "is_stale": is_st,
                            "is_filled": is_fl,
                            "available_from_ts": avail_str,
                            "session_type": spec.session_type,
                        }
                        ov = spec.option_overrides.get((strike, side, m))
                        if ov:
                            row.update(ov)
                        file_bucket.append(row)
                        opt_rows_all.append(row)

                        if spec.include_greeks:
                            greeks_rows.append(
                                {
                                    "ts_open": ts,
                                    "expiry": spec.expiry,
                                    "strike": strike,
                                    "option_type": side,
                                    "iv_status": "OK",
                                    "half_spread_PROXY": 0.45,
                                    "available_from_ts": avail_str,
                                }
                            )

    contracts_rows = [
        {"expiry": exp, "lot_size": ("" if ls is None else ls)}
        for exp, ls in sorted(contracts_map.items())
    ]

    if write_csv:
        csv_root = week_dir / "BANKNIFTY"
        if spot_rows:
            _write_csv(csv_root / "spot_1m.csv", list(spot_rows[0].keys()), spot_rows)
        if vix_rows:
            _write_csv(csv_root / "vix_1m.csv", list(vix_rows[0].keys()), vix_rows)
        if fut_rows:
            _write_csv(csv_root / "futures_1m.csv", list(fut_rows[0].keys()), fut_rows)
        if stock_rows:
            _write_csv(csv_root / "stock_1m.csv", list(stock_rows[0].keys()), stock_rows)
        if greeks_rows:
            _write_csv(csv_root / "iv_greeks_1m.csv", list(greeks_rows[0].keys()), greeks_rows)
        for fname, rows in sorted(opt_rows_by_file.items()):
            if rows:
                _write_csv(csv_root / "options_1m" / fname, list(rows[0].keys()), rows)
        _write_csv(csv_root / "contracts.csv", ["expiry", "lot_size"], contracts_rows)

    if write_parquet:
        pq_root = week_dir / "store"
        if spot_rows:
            _write_parquet(pq_root / "spot_1m.parquet", spot_rows)
        if vix_rows:
            _write_parquet(pq_root / "vix_1m.parquet", vix_rows)
        if fut_rows:
            _write_parquet(pq_root / "fut_1m.parquet", fut_rows)
        if stock_rows:
            _write_parquet(pq_root / "stock_1m.parquet", stock_rows)
        if greeks_rows:
            _write_parquet(pq_root / "iv_greeks_1m.parquet", greeks_rows)
        if opt_rows_all:
            _write_parquet(pq_root / "opt_1m.parquet", opt_rows_all)
        _write_csv(pq_root / "contracts.csv", ["expiry", "lot_size"], contracts_rows)

    return week_dir


# ------------------------------------------------------------------------------
# Config, Base Dir, Strategy & Temp Repo Helpers (§3.1, §3.2, §13)
# ------------------------------------------------------------------------------
def load_test_config(
    repo_root: Path = REPO_ROOT,
    *,
    base_capital: str = "50000",
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config.toml with base_capital populated and optional overrides applied."""
    cfg = tomllib.loads((repo_root / "config.toml").read_text(encoding="utf-8"))
    cfg["base_capital"] = base_capital
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v
    return cfg


def make_base_dir(tmp_path: Path) -> dict[str, Path]:
    """Create isolated BankNiftyBT base_dir and internal cache_dir under tmp_path."""
    base = tmp_path / "BankNiftyBT"
    data = base / "data"
    strategies = base / "strategies"
    output = base / "output"
    cache = tmp_path / "cache"
    for p in (data, strategies, output, cache):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "base_dir": base,
        "data_dir": data,
        "strategies_dir": strategies,
        "output_dir": output,
        "cache_dir": cache,
    }


def write_strategy(strategies_dir: Path, name: str, source: str) -> Path:
    """Write a synthetic strategy Python source file into strategies_dir."""
    strategies_dir.mkdir(parents=True, exist_ok=True)
    fname = name if name.endswith(".py") else f"{name}.py"
    dest = strategies_dir / fname
    dest.write_text(source, encoding="utf-8", newline="\n")
    return dest


def _compute_tests_tree_hash(tests_dir: Path) -> str:
    """Deterministic sorted tree SHA-256 of tests/, ignoring cache directories."""
    h = hashlib.sha256()
    if not tests_dir.is_dir():
        return h.hexdigest()
    files = sorted(
        p
        for p in tests_dir.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
        and not p.name.endswith(".pyc")
    )
    for fp in files:
        rel = fp.relative_to(tests_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(fp.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


def make_temp_repo(
    tmp_path: Path,
    *,
    base_capital: str = "50000",
    config_text_replacements: Mapping[str, str] | None = None,
) -> Path:
    """Build an isolated copy of the repo in tmp_path with base_capital set and valid LOCK.sha256."""
    dst = tmp_path / "repo_copy"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "strategies").mkdir(exist_ok=True)

    # Copy top-level locked/project files that exist so far
    for fp in sorted(REPO_ROOT.iterdir()):
        if fp.is_file() and fp.name != "LOCK.sha256":
            shutil.copy2(fp, dst / fp.name)

    # Copy tests/ excluding cache directories
    src_tests = REPO_ROOT / "tests"
    if src_tests.is_dir():
        shutil.copytree(
            src_tests,
            dst / "tests",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )

    # Update config.toml with base_capital and optional replacements
    cfg_path = dst / "config.toml"
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_text = cfg_text.replace('base_capital = ""', f'base_capital = "{base_capital}"', 1)
    if config_text_replacements:
        for old, new in config_text_replacements.items():
            cfg_text = cfg_text.replace(old, new)
    cfg_path.write_text(cfg_text, encoding="utf-8", newline="\n")

    # Generate valid LOCK.sha256 (delegating to run.py once it exists)
    lock_lines: list[str] = []
    run_py = dst / "run.py"
    if run_py.is_file():
        run_mod = importlib.import_module("run")
        if hasattr(run_mod, "build_lock_manifest"):
            manifest_text = run_mod.build_lock_manifest(dst)
            (dst / "LOCK.sha256").write_text(manifest_text, encoding="utf-8", newline="\n")
            return dst

    lock_targets = [
        "config.toml",
        "contracts.py",
        "data_loader.py",
        "costs.py",
        "indicators.py",
        "engine.py",
        "report.py",
        "diagnose.py",
        "run.py",
    ]
    for name in lock_targets:
        fp = dst / name
        if fp.is_file():
            digest = hashlib.sha256(fp.read_bytes()).hexdigest()
            lock_lines.append(f"{digest}  {name}")
    tests_digest = _compute_tests_tree_hash(dst / "tests")
    lock_lines.append(f"{tests_digest}  tests/")
    (dst / "LOCK.sha256").write_text(
        "\n".join(lock_lines) + "\n", encoding="utf-8", newline="\n"
    )
    return dst


@_fixture
def tmp_base_dir(tmp_path: Path) -> dict[str, Path]:
    """Pytest fixture providing isolated BankNiftyBT folders under tmp_path."""
    return make_base_dir(tmp_path)


@_fixture
def test_config() -> dict[str, Any]:
    """Pytest fixture providing validated config.toml with base_capital='50000'."""
    return load_test_config(REPO_ROOT, base_capital="50000")
