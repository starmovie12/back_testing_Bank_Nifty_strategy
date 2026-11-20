"""CLI orchestrator, lock & AST checks, weekly_state.json chain, and fork+Pipe runner.

Purpose:
    Provides the locked command-line entrypoint for engine-v2.0: validates
    config.toml (rejecting empty base_capital and legacy §17 keys), verifies
    LOCK.sha256 and git diff, runs AST static checks on strategies, enforces
    FUSE-safe O_EXCL concurrency locking, manages the append-only sha256
    hash-chained weekly_state.json and trials.jsonl, executes Fixed 1-Lot
    in parallel via fork + Pipe (draining recv() before join(), with OSError
    serial fallback) and Dynamic Compounding strictly serially, generates
    all mandatory reports, and invokes diagnose.py inside a fault-isolated
    boundary.

PRD Sections:
    §3.1, §3.2, §3.3, §6, §7, §9, §11.1, §12.3, §17

Niyam Covered:
    - Niyam 12: Determinism, LOCK.sha256 integrity, holdout gate order & logging
    - Niyam 14: Strategy AST static check preventing state leakage

Guarding Tests:
    T08, T10, T12, T16, T20, T21, T33, T42, T43, T53, T55, T56, T57, T58, T59, T61

PUBLIC API:
    Constants:
        EXIT_OK, EXIT_RUNTIME_ERROR, EXIT_INVALID_CLI_OR_CONFIG,
        EXIT_LOCK_OR_AST_FAIL, EXIT_STATE_OR_CONCURRENCY, EXIT_WEEK_REWIND,
        EXIT_HOLDOUT_REFUSED, EXIT_RUIN_RESET_REQUIRED, EXIT_GATE_ORDER,
        LOCKED_FILES, GENESIS_PREV_HASH, LEGACY_CONFIG_KEYS,
        ALLOWED_STRATEGY_IMPORTS
    Exceptions:
        StateCorruptError, ConcurrencyLockError
    Classes:
        EngineLockfile
    Functions:
        load_and_validate_config, check_live_balance_freshness,
        compute_tests_tree_hash, build_lock_manifest, verify_lock,
        check_strategy_ast, compute_record_hash, load_and_verify_weekly_state,
        append_weekly_record, group_dates_by_iso_week, append_trial_record,
        count_family_variants, run_fixed_parallel, build_arg_parser, main
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import multiprocessing
import os
import subprocess
import sys
import tomllib
from datetime import date as dt_date, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from contracts import ConfigError, ENGINE_TAG, HoldoutLocked
from costs import ZERO, round_paise, to_decimal
from data_loader import (
    DayData,
    IST_TZ,
    append_holdout_log,
    atomic_write_bytes,
    export_csv_tables,
    get_prior_days,
    load_dataset,
)
from engine import (
    DayResult,
    DualTrackResult,
    TrackRunResult,
    run_day,
    run_dual_track,
    run_track,
)
from report import (
    evaluate_weekly_verdict,
    generate_reports,
    write_manifest_json,
)

# ------------------------------------------------------------------------------
# Standard Exit Codes (§6.3) & Locked Manifest Constants (§3.2)
# ------------------------------------------------------------------------------
EXIT_OK: int = 0
EXIT_RUNTIME_ERROR: int = 1
EXIT_INVALID_CLI_OR_CONFIG: int = 2
EXIT_LOCK_OR_AST_FAIL: int = 3
EXIT_STATE_OR_CONCURRENCY: int = 4
EXIT_WEEK_REWIND: int = 5
EXIT_HOLDOUT_REFUSED: int = 6
EXIT_RUIN_RESET_REQUIRED: int = 7
EXIT_GATE_ORDER: int = 8

LOCKED_FILES: tuple[str, ...] = (
    "config.toml",
    "contracts.py",
    "data_loader.py",
    "costs.py",
    "indicators.py",
    "engine.py",
    "report.py",
    "diagnose.py",
    "run.py",
)
GENESIS_PREV_HASH: str = "0" * 64
LEGACY_CONFIG_KEYS: frozenset[str] = frozenset(
    {"max_lookback_days", "day_max_loss", "per_trade_max_risk"}
)
ALLOWED_STRATEGY_IMPORTS: frozenset[str] = frozenset(
    {
        "math",
        "numpy",
        "statistics",
        "itertools",
        "dataclasses",
        "collections",
        "typing",
        "contracts",
        "indicators",
    }
)
_BANNED_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "memoryview",
        "input",
        "breakpoint",
    }
)
_BANNED_ATTRS: frozenset[str] = frozenset({"base", "ctypes", "data"})
_BANNED_NUMPY_ATTRS: frozenset[str] = frozenset(
    {
        "random",
        "load",
        "save",
        "savez",
        "savez_compressed",
        "loadtxt",
        "genfromtxt",
        "fromfile",
        "frombuffer",
        "memmap",
        "ctypeslib",
        "lib",
        "stride_tricks",
        "ndarray",
        "seterr",
        "set_printoptions",
    }
)
_ALLOWED_DUNDER_DEFS: frozenset[str] = frozenset({"__init__", "__post_init__"})


class StateCorruptError(Exception):
    """Raised when weekly_state.json fails schema or hash-chain verification."""


class ConcurrencyLockError(Exception):
    """Raised when another active process holds <base_dir>/.engine.lock."""


# ------------------------------------------------------------------------------
# Config Validation & Live-Bot Parity Validators (§7, §9, §17)
# ------------------------------------------------------------------------------
def load_and_validate_config(
    config_path: Path,
    cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config.toml, reject legacy §17 keys & empty base_capital, validate parity keys."""
    if not config_path.is_file():
        raise ConfigError(f"config.toml not found at {config_path}")
    try:
        cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Malformed config.toml: {exc}") from exc

    # Reject legacy V3.1 keys (§17)
    bad_legacy = sorted(k for k in cfg.keys() if k in LEGACY_CONFIG_KEYS)
    if bad_legacy:
        raise ConfigError(f"Legacy V3.1 config keys forbidden: {', '.join(bad_legacy)}")

    # Apply Machine Knob overrides only (§7)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                cfg[k] = v

    # base_capital REQUIRED (empty -> exit code 2; §7)
    bc_raw = cfg.get("base_capital", "")
    if bc_raw is None or str(bc_raw).strip() == "":
        raise ConfigError("base_capital is empty in config.toml (OWNER_DECISION_NEEDED)")
    try:
        bc_val = to_decimal(bc_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid base_capital: {bc_raw!r}") from exc
    if bc_val <= ZERO:
        raise ConfigError(f"base_capital must be > 0, got {bc_val}")

    # Live-Bot Parity validators (§9, T57)
    bma = cfg.get("balance_max_age_sec")
    if isinstance(bma, bool) or not isinstance(bma, int) or bma <= 0:
        raise ConfigError(f"balance_max_age_sec must be int > 0, got {bma!r}")

    nfq = cfg.get("nse_freeze_qty", "")
    if nfq not in ("", None):
        if isinstance(nfq, bool) or not isinstance(nfq, int) or nfq <= 0:
            raise ConfigError(f"nse_freeze_qty must be int > 0 or empty, got {nfq!r}")

    # Resolve auto workers == 0 -> min(4, max(1, cpu_count // 2))
    w_raw = cfg.get("workers", 0)
    if isinstance(w_raw, bool) or not isinstance(w_raw, int) or w_raw < 0:
        raise ConfigError(f"workers must be int >= 0, got {w_raw!r}")
    if w_raw == 0:
        cpus = os.cpu_count() or 2
        cfg["workers"] = min(4, max(1, cpus // 2))

    for knob in ("batch_days", "peak_memory_mb", "paper_min_trades"):
        kv = cfg.get(knob)
        if isinstance(kv, bool) or not isinstance(kv, int) or kv <= 0:
            raise ConfigError(f"{knob} must be int > 0, got {kv!r}")

    return cfg


def check_live_balance_freshness(
    age_sec: float | None,
    api_ok: bool,
    config: Mapping[str, Any],
) -> tuple[bool, str]:
    """Live-Bot Parity pure helper (§9, T57): reject stale or failed balance quotes."""
    max_age = int(config["balance_max_age_sec"])
    if not api_ok or age_sec is None:
        return False, "BALANCE_UNAVAILABLE"
    if float(age_sec) < 0.0 or float(age_sec) > float(max_age):
        return False, "BALANCE_UNAVAILABLE"
    return True, "OK"


# ------------------------------------------------------------------------------
# LOCK.sha256 Verification & Git Diff Check (§3.2, T10, T33)
# ------------------------------------------------------------------------------
def compute_tests_tree_hash(tests_dir: Path) -> str:
    """Compute deterministic sorted SHA-256 over tests/ ignoring cache dirs."""
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


def build_lock_manifest(repo_root: Path, *, require_all: bool = False) -> str:
    """Build canonical LOCK.sha256 text (locked files + combined tests/ tree hash)."""
    lines: list[str] = []
    for name in LOCKED_FILES:
        fp = repo_root / name
        if not fp.is_file():
            if require_all:
                raise FileNotFoundError(f"Locked file missing: {fp}")
            continue
        digest = hashlib.sha256(fp.read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    tests_digest = compute_tests_tree_hash(repo_root / "tests")
    lines.append(f"{tests_digest}  tests/")
    return "\n".join(lines) + "\n"


def verify_lock(repo_root: Path) -> tuple[bool, str, str]:
    """Verify LOCK.sha256 and git diff (if .git exists). Returns (ok, detail, git_check)."""
    lock_path = repo_root / "LOCK.sha256"
    if not lock_path.is_file():
        return False, "LOCK.sha256 missing", "NA"

    raw_lock_bytes = lock_path.read_bytes()
    if b"\r" in raw_lock_bytes or b"\x0b" in raw_lock_bytes or not raw_lock_bytes.endswith(b"\n"):
        return False, "LOCK.sha256 malformed line endings or trailing byte", "NA"
    raw_lines = [
        ln.strip()
        for ln in raw_lock_bytes.decode("utf-8").splitlines()
        if ln.strip()
    ]
    if not raw_lines:
        return False, "LOCK.sha256 is empty", "NA"

    entries: dict[str, str] = {}
    for ln in raw_lines:
        parts = ln.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            return False, f"Malformed LOCK.sha256 line: {ln!r}", "NA"
        entries[parts[1]] = parts[0]

    for target, expected_hex in entries.items():
        if target == "tests/":
            actual = compute_tests_tree_hash(repo_root / "tests")
        else:
            fp = repo_root / target
            if not fp.is_file():
                return False, f"Locked file missing: {target}", "NA"
            actual = hashlib.sha256(fp.read_bytes()).hexdigest()
        if actual != expected_hex:
            return False, f"SHA-256 mismatch on {target}", "NA"

    # Git diff check if .git exists (§3.2, T10, T33)
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return True, "OK", "NA (no .git)"

    tag_check = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{ENGINE_TAG}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_check.returncode != 0:
        return True, "OK", f"NA (tag {ENGINE_TAG} absent)"

    tracked_paths = [k for k in entries.keys()] + ["LOCK.sha256"]
    diff_res = subprocess.run(
        ["git", "diff", ENGINE_TAG, "--", *tracked_paths],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if diff_res.returncode != 0 or diff_res.stdout.strip() != "":
        return False, f"git diff against {ENGINE_TAG} is non-empty on locked files", "FAILED"
    return True, "OK", "CLEAN"


# ------------------------------------------------------------------------------
# Strategy AST Static Checker (§11.1, T12, T21, T34)
# ------------------------------------------------------------------------------
def check_strategy_ast(
    source: str,
    filename: str = "<strategy>",
) -> list[tuple[int, str]]:
    """Inspect strategy Python source via AST and return (lineno, reason) violations."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [(exc.lineno or 1, f"SyntaxError: {exc.msg}")]

    violations: list[tuple[int, str]] = []
    numpy_aliases: set[str] = {"numpy"}

    for node in ast.walk(tree):
        # 1. Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in ALLOWED_STRATEGY_IMPORTS:
                    violations.append((node.lineno, f"Forbidden import: {alias.name}"))
                elif root_mod == "numpy":
                    if "." in alias.name:
                        sub = alias.name.split(".")[1]
                        if sub in _BANNED_NUMPY_ATTRS:
                            violations.append((node.lineno, f"Forbidden numpy submodule: {alias.name}"))
                    numpy_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or not node.module:
                violations.append((node.lineno, "Relative import forbidden"))
                continue
            root_mod = node.module.split(".")[0]
            if root_mod not in ALLOWED_STRATEGY_IMPORTS:
                violations.append((node.lineno, f"Forbidden import from: {node.module}"))
            if root_mod == "numpy" and "." in node.module:
                sub = node.module.split(".")[1]
                if sub in _BANNED_NUMPY_ATTRS:
                    violations.append((node.lineno, f"Forbidden numpy submodule: {node.module}"))
            for alias in node.names:
                if alias.name == "*":
                    violations.append((node.lineno, "Wildcard import 'from ... import *' forbidden"))
                elif root_mod == "numpy" and alias.name in _BANNED_NUMPY_ATTRS:
                    violations.append((node.lineno, f"Forbidden numpy import: {alias.name}"))

        # 2. Global / Nonlocal (Niyam 14, T34)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append((node.lineno, "global/nonlocal state persistence forbidden (Niyam 14)"))

        # 3. Banned builtins & dunders
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_BUILTINS:
                violations.append((node.lineno, f"Forbidden builtin/name: {node.id}"))
            elif node.id.startswith("__") and node.id.endswith("__"):
                violations.append((node.lineno, f"Forbidden dunder name: {node.id}"))

        # 4. Banned attributes (.base, .ctypes, .data, __dunder__, numpy banned attrs)
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                violations.append((node.lineno, f"Forbidden attribute access: .{node.attr}"))
            elif node.attr.startswith("__") and node.attr.endswith("__"):
                violations.append((node.lineno, f"Forbidden dunder attribute: .{node.attr}"))
            elif isinstance(node.value, ast.Name) and node.value.id in numpy_aliases:
                if node.attr in _BANNED_NUMPY_ATTRS:
                    violations.append((node.lineno, f"Forbidden numpy attribute: {node.value.id}.{node.attr}"))

        # 5. Function defs: forbid dunders except __init__ / __post_init__
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name.endswith("__"):
                if node.name not in _ALLOWED_DUNDER_DEFS:
                    violations.append((node.lineno, f"Forbidden dunder method: {node.name}"))

        # 6. Class-level mutable state (Niyam 14, T34)
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    if isinstance(stmt.value, (ast.List, ast.Dict, ast.Set)):
                        violations.append((stmt.lineno, "Class-level mutable state forbidden (Niyam 14)"))
                elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    if isinstance(stmt.value, (ast.List, ast.Dict, ast.Set)):
                        violations.append((stmt.lineno, "Class-level mutable state forbidden (Niyam 14)"))

        # 7. PARAMS[...] = ... mutation
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id == "PARAMS":
                    violations.append((node.lineno, "Mutating PARAMS[...] is forbidden"))

    violations.sort(key=lambda x: (x[0], x[1]))
    return violations


# ------------------------------------------------------------------------------
# FUSE-Safe O_EXCL Concurrency Lockfile (§6.4, T61)
# ------------------------------------------------------------------------------
class EngineLockfile:
    """Android FUSE-safe O_EXCL lockfile with stale-PID auto-recovery."""

    def __init__(self, base_dir: Path, cmd_label: str = "run.py") -> None:
        self._path = base_dir / ".engine.lock"
        self._cmd = cmd_label
        self._acquired = False

    def _try_acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            payload = json.dumps(
                {
                    "pid": os.getpid(),
                    "cmd": self._cmd,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            os.write(fd, payload + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        self._acquired = True

    def acquire(self) -> None:
        try:
            self._try_acquire()
            return
        except FileExistsError:
            pass

        # Inspect existing lockfile PID liveness
        stale = False
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            pid = int(data.get("pid", -1))
            if pid <= 0:
                stale = True
            else:
                try:
                    os.kill(pid, 0)
                    # Process is alive
                    raise ConcurrencyLockError(f"CONCURRENT_RUN_ACTIVE (pid={pid})")
                except ProcessLookupError:
                    stale = True
                except PermissionError:
                    raise ConcurrencyLockError(f"CONCURRENT_RUN_ACTIVE (pid={pid})")
        except ConcurrencyLockError:
            raise
        except Exception:
            stale = True

        if stale:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
            try:
                self._try_acquire()
                return
            except FileExistsError as exc:
                raise ConcurrencyLockError("CONCURRENT_RUN_ACTIVE") from exc

    def release(self) -> None:
        if self._acquired:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
            self._acquired = False

    def __enter__(self) -> EngineLockfile:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


# ------------------------------------------------------------------------------
# Hash-Chained weekly_state.json & trials.jsonl (§6.2, T42, T55-T59)
# ------------------------------------------------------------------------------
def compute_record_hash(record_without_hash: Mapping[str, Any]) -> str:
    """Compute SHA-256 of canonical JSON (sorted keys, fixed separators, UTF-8) without record_hash."""
    clean = {k: v for k, v in record_without_hash.items() if k != "record_hash"}
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_and_verify_weekly_state(base_dir: Path) -> dict[str, Any]:
    """Load and verify full hash-chain of <base_dir>/weekly_state.json (exit 4 on tamper)."""
    state_path = base_dir / "weekly_state.json"
    if not state_path.is_file():
        return {"schema_version": 2, "engine_tag": ENGINE_TAG, "records": []}

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StateCorruptError("STATE_CORRUPT_OR_TAMPERED: invalid JSON") from exc

    if not isinstance(data, dict):
        raise StateCorruptError("STATE_CORRUPT_OR_TAMPERED: root must be object")
    if data.get("schema_version") != 2 or data.get("engine_tag") != ENGINE_TAG:
        raise StateCorruptError("STATE_CORRUPT_OR_TAMPERED: schema_version or engine_tag mismatch")

    records = data.get("records")
    if not isinstance(records, list):
        raise StateCorruptError("STATE_CORRUPT_OR_TAMPERED: records must be list")

    expected_prev = GENESIS_PREV_HASH
    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            raise StateCorruptError(f"STATE_CORRUPT_OR_TAMPERED: record {idx} not dict")
        if rec.get("seq") != idx:
            raise StateCorruptError(f"STATE_CORRUPT_OR_TAMPERED: seq mismatch at {idx}")
        if rec.get("prev_hash") != expected_prev:
            raise StateCorruptError(f"STATE_CORRUPT_OR_TAMPERED: prev_hash mismatch at seq {idx}")
        stored_hash = rec.get("record_hash")
        recomputed = compute_record_hash(rec)
        if stored_hash != recomputed:
            raise StateCorruptError(f"STATE_CORRUPT_OR_TAMPERED: record_hash mismatch at seq {idx}")
        expected_prev = stored_hash

    return data


def append_weekly_record(
    base_dir: Path,
    state: dict[str, Any],
    record_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a hash-chained record to weekly_state.json and persist atomically."""
    records: list[dict[str, Any]] = state["records"]
    seq = len(records) + 1
    prev_hash = records[-1]["record_hash"] if records else GENESIS_PREV_HASH
    rec = dict(record_fields)
    rec["seq"] = seq
    rec["prev_hash"] = prev_hash
    rec["record_hash"] = compute_record_hash(rec)
    records.append(rec)

    payload = json.dumps(state, sort_keys=True, indent=2) + "\n"
    atomic_write_bytes(base_dir / "weekly_state.json", payload.encode("utf-8"))
    return rec


def _latest_family_record(state: Mapping[str, Any], family: str) -> dict[str, Any] | None:
    for rec in reversed(state.get("records", [])):
        if rec.get("strategy_family") == family:
            return rec
    return None


def append_trial_record(base_dir: Path, entry: Mapping[str, Any]) -> None:
    """Append one JSON line to <base_dir>/trials.jsonl (parent-only, flush+fsync)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "trials.jsonl"
    line = json.dumps(dict(entry), sort_keys=True, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def count_family_variants(base_dir: Path, strategy_family: str, current_hash: str) -> int:
    """Count distinct strategy_hash variants for strategy_family strictly on ENGINE_TAG (T42)."""
    seen: set[str] = {current_hash}
    path = base_dir / "trials.jsonl"
    if path.is_file():
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if (
                rec.get("engine_tag") == ENGINE_TAG
                and rec.get("strategy_family") == strategy_family
                and rec.get("strategy_hash")
            ):
                seen.add(str(rec["strategy_hash"]))
    return len(seen)


# ------------------------------------------------------------------------------
# Data-Driven ISO Week Resolution (§6.1, T55, T56)
# ------------------------------------------------------------------------------
def group_dates_by_iso_week(dates: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group sorted YYYY-MM-DD dates by ISO week 'YYYY-Www' present in data."""
    buckets: dict[str, list[str]] = {}
    for d_str in sorted(dates):
        iso_y, iso_w, _ = dt_date.fromisoformat(d_str).isocalendar()
        wk = f"{iso_y:04d}-W{iso_w:02d}"
        buckets.setdefault(wk, []).append(d_str)
    return [(wk, buckets[wk]) for wk in sorted(buckets.keys())]


# ------------------------------------------------------------------------------
# Termux-Safe fork + Pipe Fixed-Track Parallelism (§12.3, T20, T43, T61)
# ------------------------------------------------------------------------------
def _fixed_worker_target(
    conn: Any,
    days_subset: dict[str, DayData],
    batch_dates: list[str],
    all_days_ref: Mapping[str, DayData],
    strategy_source: str,
    config: Mapping[str, Any],
    delay_d: int,
    mode: str,
) -> None:
    """Worker entrypoint: runs Fixed 1-Lot days and sends results over Pipe."""
    warmup_n = int(config["warmup_days"])
    holdout_start = str(config["holdout_start"])
    base_cap = to_decimal(config["base_capital"])
    out: list[DayResult] = []
    try:
        for d_str in batch_dates:
            prior = get_prior_days(
                all_days_ref, d_str, warmup_days=warmup_n, holdout_start=holdout_start
            )
            dr = run_day(
                day=days_subset[d_str],
                prior_days=prior,
                strategy_source=strategy_source,
                config=config,
                track="fixed",
                day_open_balance=base_cap,
                delay_d=delay_d,
                mode=mode,
            )
            from dataclasses import replace as _dc_replace
            out.append(_dc_replace(dr, notrade_counts=dict(dr.notrade_counts)))
        conn.send(("OK", out))
    except Exception as exc:
        conn.send(("ERR", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _stitch_fixed_results(day_results: list[DayResult], base_cap: Decimal) -> TrackRunResult:
    """Stitch per-day Fixed 1-Lot DayResults in chronological order with sequential trade_no."""
    ordered = sorted(day_results, key=lambda r: r.date)
    all_trades = []
    all_skipped = []
    agg_notrade: dict[str, int] = {}
    cap_breach_days: list[str] = []
    total_decide = 0
    total_gaps = 0
    trade_no = 0

    from dataclasses import replace

    for dr in ordered:
        for tr in dr.trades:
            trade_no += 1
            all_trades.append(replace(tr, trade_no=trade_no))
        all_skipped.extend(dr.skipped)
        total_decide += dr.decide_calls
        total_gaps += dr.data_gap_skips
        for k, v in dr.notrade_counts.items():
            agg_notrade[k] = agg_notrade.get(k, 0) + v
        if dr.cap_breach:
            cap_breach_days.append(dr.date)

    final_base = round_paise(base_cap + sum((t.net_base for t in all_trades), ZERO))
    final_stress = round_paise(base_cap + sum((t.net_stress for t in all_trades), ZERO))
    return TrackRunResult(
        track="fixed",
        start_balance=base_cap,
        final_balance_base=final_base,
        final_balance_stress=final_stress,
        trades=tuple(all_trades),
        skipped=tuple(all_skipped),
        day_results=tuple(ordered),
        notrade_counts=MappingProxyType(agg_notrade),
        decide_calls=total_decide,
        data_gap_skips=total_gaps,
        cap_breach_days=tuple(cap_breach_days),
        ruin_flag=False,
        ruin_date=None,
        ruin_balance=None,
    )


def run_fixed_parallel(
    *,
    days: Mapping[str, DayData],
    trading_dates: Sequence[str],
    strategy_source: str,
    config: Mapping[str, Any],
    workers: int,
    delay_d: int = 0,
    mode: str = "fast",
    _pipe_factory: Callable[..., Any] | None = None,
) -> tuple[TrackRunResult, str]:
    """Run Fixed 1-Lot across trading_dates using fork + Pipe (recv BEFORE join; §12.3).

    Returns (TrackRunResult, parallel_status) where parallel_status is
    'serial', 'fork_pipe', or 'serial_fallback' (on OSError; T43).
    """
    dates = sorted(trading_dates)
    base_cap = to_decimal(config["base_capital"])
    if workers <= 1 or len(dates) <= 1:
        res = run_track(
            days=days,
            trading_dates=dates,
            strategy_source=strategy_source,
            config=config,
            track="fixed",
            delay_d=delay_d,
            mode=mode,
        )
        return res, "serial"

    # Respect batch_days memory window (§12.3) when splitting across workers
    batch_sz = max(1, int(config.get("batch_days", 20)))
    if len(dates) > batch_sz:
        stitched_days: list[DayResult] = []
        par_label = "fork_pipe"
        for i0 in range(0, len(dates), batch_sz):
            sub_res, sub_st = run_fixed_parallel(
                days=days,
                trading_dates=dates[i0 : i0 + batch_sz],
                strategy_source=strategy_source,
                config=config,
                workers=workers,
                delay_d=delay_d,
                mode=mode,
                _pipe_factory=_pipe_factory,
            )
            stitched_days.extend(sub_res.day_results)
            if sub_st == "serial_fallback":
                par_label = "serial_fallback"
        return _stitch_fixed_results(stitched_days, base_cap), par_label

    # Split dates deterministically across min(workers, len(dates)) chunks
    n_procs = min(int(workers), len(dates))
    chunks: list[list[str]] = [[] for _ in range(n_procs)]
    for idx, d_str in enumerate(dates):
        chunks[idx % n_procs].append(d_str)

    try:
        ctx = multiprocessing.get_context("fork")
        procs: list[Any] = []
        parent_conns: list[Any] = []
        for chunk in chunks:
            p_conn, c_conn = (
                _pipe_factory(duplex=False)
                if _pipe_factory is not None
                else ctx.Pipe(duplex=False)
            )
            subset = {d: days[d] for d in chunk}
            proc = ctx.Process(
                target=_fixed_worker_target,
                args=(c_conn, subset, chunk, days, strategy_source, config, delay_d, mode),
            )
            proc.start()
            c_conn.close()
            procs.append(proc)
            parent_conns.append(p_conn)

        # CRITICAL (§12.3, T61): drain EVERY pipe with recv() strictly BEFORE join()
        collected: list[DayResult] = []
        worker_errors: list[str] = []
        for p_conn in parent_conns:
            try:
                status, payload = p_conn.recv()
                if status == "OK":
                    collected.extend(payload)
                else:
                    worker_errors.append(str(payload))
            finally:
                p_conn.close()

        for proc in procs:
            proc.join()

        if worker_errors:
            raise RuntimeError(f"Worker failure: {worker_errors[0]}")
        return _stitch_fixed_results(collected, base_cap), "fork_pipe"
    except OSError:
        # Allowed fallback per L09 & T43: record parallel = "serial_fallback" in manifest
        res = run_track(
            days=days,
            trading_dates=dates,
            strategy_source=strategy_source,
            config=config,
            track="fixed",
            delay_d=delay_d,
            mode=mode,
        )
        return res, "serial_fallback"


# ------------------------------------------------------------------------------
# Strategy File Resolution & Fault-Isolated Diagnosis (§3.1, §12.2)
# ------------------------------------------------------------------------------
def _resolve_strategy_path(strategy_arg: str, repo_root: Path, base_dir: Path) -> Path:
    """Resolve strategy file path; hard error if duplicate exists in repo and base_dir (§3.1)."""
    cand = Path(strategy_arg)
    if cand.is_file():
        return cand.resolve()

    fname = strategy_arg if strategy_arg.endswith(".py") else f"{strategy_arg}.py"
    in_repo = repo_root / "strategies" / fname
    in_base = base_dir / "strategies" / fname
    if in_repo.is_file() and in_base.is_file() and in_repo.resolve() != in_base.resolve():
        raise ConfigError(
            f"Duplicate strategy file {fname!r} in both repo/strategies and base_dir/strategies"
        )
    if in_repo.is_file():
        return in_repo.resolve()
    if in_base.is_file():
        return in_base.resolve()
    raise FileNotFoundError(f"Strategy file not found: {strategy_arg!r}")


def _make_output_dir(output_root: Path, strategy_family: str) -> Path:
    """Create unique <YYYY-MM-DD>_<HHMM>_<strategy>[_N] directory under base_dir/output/."""
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(IST_TZ).strftime("%Y-%m-%d_%H%M")
    base_name = f"{stamp}_{strategy_family}"
    cand = output_root / base_name
    if not cand.exists():
        cand.mkdir(parents=True, exist_ok=True)
        return cand
    idx = 2
    while True:
        cand = output_root / f"{base_name}_{idx}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        idx += 1


def _run_diagnosis_isolated(
    *,
    output_dir: Path,
    dual: DualTrackResult,
    days: Mapping[str, DayData],
    summary: Mapping[str, Any],
    manifest: dict[str, Any],
    config: Mapping[str, Any],
    base_dir: Path,
    is_holdout: bool,
    exp_id: str | None,
) -> None:
    """Invoke diagnose.py inside a fault-isolated boundary (including import; §12.2, T45)."""
    try:
        diag_mod = importlib.import_module("diagnose")
        diag_mod.run_diagnosis(
            output_dir=output_dir,
            dual=dual,
            days=days,
            summary=summary,
            manifest=manifest,
            config=config,
            base_dir=base_dir,
            is_holdout=is_holdout,
            exp_id=exp_id,
        )
        manifest["diagnosis"] = "OK"
    except Exception as exc:
        # Allowed broad catch per L09 & §12.2: record failure in manifest, change nothing else
        manifest["diagnosis"] = f"FAILED({type(exc).__name__}: {exc})"
    write_manifest_json(output_dir / "manifest.json", manifest)


# ------------------------------------------------------------------------------
# CLI Argument Parser & Main Orchestrator (§6.3)
# ------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the locked CLI parser for run.py (§6.3)."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=f"BANKNIFTY Backtest Engine ({ENGINE_TAG})",
    )
    parser.add_argument("strategy", help="Strategy name/path, or 'export-csv' subcommand")
    parser.add_argument("--week", dest="week", default=None, help="ISO week 'YYYY-Www' or 'next'")
    parser.add_argument("--design-gate", action="store_true", help="Run full DESIGN gate evaluation")
    parser.add_argument("--holdout", action="store_true", help="Unlock and run Sealed Holdout")
    parser.add_argument("--exp", dest="exp_id", default=None, help="Experiment label (valid with --week/--design-gate only)")
    parser.add_argument("--reset-ruin", action="store_true", help="Acknowledge Dynamic RUIN and reset capital")
    parser.add_argument("--confirm-capital", dest="confirm_capital", default=None, help="New INR capital for --reset-ruin")
    parser.add_argument("--from", dest="from_date", default=None, help="Ad-hoc range start YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", default=None, help="Ad-hoc range end YYYY-MM-DD")
    # Machine Knob overrides (§7)
    parser.add_argument("--base-dir", dest="base_dir", default=None)
    parser.add_argument("--cache-dir", dest="cache_dir", default=None)
    parser.add_argument("--workers", dest="workers", type=int, default=None)
    parser.add_argument("--diag-level", dest="diag_level", choices=("full", "lite"), default=None)
    parser.add_argument("--repo-root", dest="repo_root", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute CLI command and return standard §6.3 exit code (0..8)."""
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_INVALID_CLI_OR_CONFIG

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parent
    )

    # 1. Validate CLI mode combinations (§6.3)
    mode_flags = sum(
        1 for flag in (args.week is not None, args.design_gate, args.holdout, args.reset_ruin) if flag
    )
    if mode_flags > 1:
        print("INVALID_CLI_ARGS: cannot combine --week, --design-gate, --holdout, --reset-ruin", file=sys.stderr)
        return EXIT_INVALID_CLI_OR_CONFIG
    if args.exp_id is not None and (args.holdout or args.reset_ruin):
        print("INVALID_CLI_ARGS: --exp is invalid with --holdout or --reset-ruin", file=sys.stderr)
        return EXIT_INVALID_CLI_OR_CONFIG
    if (args.from_date or args.to_date) and (args.week is not None or args.design_gate or args.holdout or args.reset_ruin):
        print("INVALID_CLI_ARGS: --from/--to cannot combine with --week/--design-gate/--holdout", file=sys.stderr)
        return EXIT_INVALID_CLI_OR_CONFIG

    # 2. Verify LOCK.sha256 & git diff before parsing config (§3.2 -> exit 3)
    git_check = "NA"
    if args.strategy != "export-csv":
        lock_ok, lock_detail, git_check = verify_lock(repo_root)
        if not lock_ok:
            print(f"LOCK_VERIFICATION_FAILED: {lock_detail}", file=sys.stderr)
            return EXIT_LOCK_OR_AST_FAIL

    # 3. Load & validate config.toml (empty base_capital / legacy keys -> exit 2)
    cli_overrides = {
        "base_dir": args.base_dir,
        "cache_dir": args.cache_dir,
        "workers": args.workers,
        "diag_level": args.diag_level,
    }
    try:
        cfg = load_and_validate_config(repo_root / "config.toml", cli_overrides)
    except ConfigError as exc:
        print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID_CLI_OR_CONFIG

    base_dir = Path(os.path.expanduser(str(cfg["base_dir"]))).resolve()
    data_dir = base_dir / "data"
    output_root = base_dir / "output"
    (base_dir / "strategies").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    # Handle export-csv subcommand (§10.1)
    if args.strategy == "export-csv":
        try:
            exported = export_csv_tables(data_dir)
            print(f"export-csv complete: {len(exported)} files")
            return EXIT_OK
        except Exception as exc:
            print(f"export-csv failed: {exc}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR

    # 4. Acquire FUSE-safe O_EXCL lockfile (§6.4 -> exit 4)
    try:
        with EngineLockfile(base_dir, cmd_label=" ".join(sys.argv)):
            return _execute_locked(
                args=args,
                repo_root=repo_root,
                base_dir=base_dir,
                data_dir=data_dir,
                output_root=output_root,
                cfg=cfg,
                git_check=git_check,
            )
    except ConcurrencyLockError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_STATE_OR_CONCURRENCY
    except StateCorruptError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_STATE_OR_CONCURRENCY
    except HoldoutLocked as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_HOLDOUT_REFUSED
    except ConfigError as exc:
        print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID_CLI_OR_CONFIG
    except Exception as exc:
        print(f"RUNTIME_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


def _execute_locked(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    base_dir: Path,
    data_dir: Path,
    output_root: Path,
    cfg: dict[str, Any],
    git_check: str,
) -> int:
    """Core run logic executed while holding .engine.lock."""
    # Verify weekly_state.json chain on every startup (§6.2)
    state = load_and_verify_weekly_state(base_dir)

    # Resolve strategy file & run AST static check (§11.1 -> exit 3)
    strat_path = _resolve_strategy_path(args.strategy, repo_root, base_dir)
    strat_bytes = strat_path.read_bytes()
    strat_source = strat_bytes.decode("utf-8")
    strat_hash = hashlib.sha256(strat_bytes).hexdigest()
    strategy_family = strat_path.stem
    lineage_id = hashlib.sha256(f"{strategy_family}|{ENGINE_TAG}".encode("utf-8")).hexdigest()

    ast_violations = check_strategy_ast(strat_source, filename=strat_path.name)
    if ast_violations:
        for ln, msg in ast_violations:
            print(f"AST_VIOLATION {strat_path.name}:{ln}: {msg}", file=sys.stderr)
        return EXIT_LOCK_OR_AST_FAIL

    prev_rec = _latest_family_record(state, strategy_family)
    base_cap = to_decimal(cfg["base_capital"])
    now_ist = datetime.now(IST_TZ).isoformat(timespec="seconds")

    # Handle --reset-ruin --confirm-capital <INR> (§6.3)
    if args.reset_ruin:
        if prev_rec is None or not bool(prev_rec.get("ruin_flag")):
            print("INVALID_CLI_ARGS: --reset-ruin valid only when latest record has ruin_flag=true", file=sys.stderr)
            return EXIT_INVALID_CLI_OR_CONFIG
        if args.confirm_capital is None:
            print("INVALID_CLI_ARGS: --confirm-capital <INR> required with --reset-ruin", file=sys.stderr)
            return EXIT_INVALID_CLI_OR_CONFIG
        try:
            new_cap = to_decimal(args.confirm_capital)
        except (TypeError, ValueError):
            print("INVALID_CLI_ARGS: --confirm-capital must be positive INR", file=sys.stderr)
            return EXIT_INVALID_CLI_OR_CONFIG
        if new_cap <= ZERO:
            print("INVALID_CLI_ARGS: --confirm-capital must be > 0", file=sys.stderr)
            return EXIT_INVALID_CLI_OR_CONFIG

        append_weekly_record(
            base_dir,
            state,
            {
                "timestamp_ist": now_ist,
                "record_type": "RUIN_RESET",
                "strategy_family": strategy_family,
                "strategy_hash": strat_hash,
                "lineage_id": lineage_id,
                "week_id": None,
                "weeks_completed_in_lineage": list(prev_rec.get("weeks_completed_in_lineage", [])),
                "outcome": "RUIN_RESET",
                "streak": 0,
                "paused_weeks": 0,
                "span_start_week": None,
                "span_end_week": None,
                "span_trades_fixed": 0,
                "span_trades_dynamic": 0,
                "span_stress_net_fixed": "0.00",
                "span_stress_net_dynamic": "0.00",
                "dynamic_closing_balance": str(round_paise(new_cap)),
                "ruin_flag": False,
                "weekly_pass_unlocked": False,
                "design_pass_unlocked": False,
                "holdout_opened": bool(prev_rec.get("holdout_opened", False)),
            },
        )
        return EXIT_OK

    # Gate-order preconditions (§6.3, T58)
    same_hash = prev_rec is not None and prev_rec.get("strategy_hash") == strat_hash
    weekly_unlocked = bool(same_hash and prev_rec.get("weekly_pass_unlocked"))
    design_unlocked = bool(same_hash and prev_rec.get("design_pass_unlocked"))
    holdout_max = int(cfg["holdout_max_opens"])
    holdout_prior = int(cfg["holdout_prior_opens"])
    holdout_opens_used = sum(
        1 for r in state["records"]
        if r.get("strategy_family") == strategy_family and r.get("record_type") == "HOLDOUT"
    )

    if args.design_gate and not weekly_unlocked:
        print("WEEKLY_PASS_REQUIRED", file=sys.stderr)
        return EXIT_GATE_ORDER

    if args.holdout:
        if not weekly_unlocked or not design_unlocked or holdout_opens_used >= holdout_max:
            append_holdout_log(
                base_dir,
                {
                    "action": "REFUSED",
                    "reason": "GATE_OR_MAX_OPENS",
                    "strategy_family": strategy_family,
                    "weekly_pass_unlocked": weekly_unlocked,
                    "design_pass_unlocked": design_unlocked,
                    "holdout_opens_used": holdout_opens_used,
                    "holdout_max_opens": holdout_max,
                },
            )
            print("HOLDOUT_REFUSED", file=sys.stderr)
            return EXIT_HOLDOUT_REFUSED

    # Determine mode & load dataset
    if args.holdout:
        run_mode = "holdout"
        ds = load_dataset(data_dir, cfg, allow_holdout=True, auto_trim_holdout=False, base_dir=base_dir)
        selected_dates = [d for d in ds.holdout_days]
    elif args.design_gate:
        run_mode = "design-gate"
        ds = load_dataset(data_dir, cfg, allow_holdout=False, auto_trim_holdout=True, base_dir=base_dir)
        selected_dates = list(ds.trading_days)
    elif args.week is not None:
        run_mode = "weekly"
        # Load all valid dates first to check if requested week overlaps holdout
        ds_design = load_dataset(data_dir, cfg, allow_holdout=False, auto_trim_holdout=True, base_dir=base_dir)
        if prev_rec is not None and bool(prev_rec.get("ruin_flag")):
            print("DYNAMIC_RUIN_RESET_REQUIRED", file=sys.stderr)
            return EXIT_RUIN_RESET_REQUIRED

        completed_weeks = list(prev_rec.get("weeks_completed_in_lineage", [])) if prev_rec else []
        iso_weeks = group_dates_by_iso_week(ds_design.trading_days)
        window_w = max(1, int(cfg.get("window_weeks", 1)))

        if args.week == "next":
            unseen = [(wk, dts) for wk, dts in iso_weeks if wk not in completed_weeks]
            if not unseen:
                print("NO_UNSEEN_DESIGN_WEEK_AVAILABLE", file=sys.stderr)
                return EXIT_HOLDOUT_REFUSED
            chosen_block = unseen[:window_w]
            target_week_id = chosen_block[-1][0]
            selected_dates = [d for _, dts in chosen_block for d in dts]
        else:
            target_week_id = str(args.week)
            if target_week_id in completed_weeks:
                print("WEEK_ALREADY_EXECUTED_IN_LINEAGE", file=sys.stderr)
                return EXIT_WEEK_REWIND
            wk_map = dict(iso_weeks)
            if target_week_id not in wk_map:
                # Check if requested week lies in holdout (>= holdout_start)
                ds_all = load_dataset(data_dir, cfg, allow_holdout=True, auto_trim_holdout=False, base_dir=None)
                ho_weeks = dict(group_dates_by_iso_week(ds_all.holdout_days))
                if target_week_id in ho_weeks:
                    append_holdout_log(
                        base_dir,
                        {"action": "REFUSED", "reason": "WEEK_TOUCHES_HOLDOUT", "week_id": target_week_id},
                    )
                    print("HOLDOUT_REFUSED: requested week touches holdout", file=sys.stderr)
                    return EXIT_HOLDOUT_REFUSED
                print(f"WEEK_NOT_FOUND: {target_week_id}", file=sys.stderr)
                return EXIT_INVALID_CLI_OR_CONFIG
            # If window_weeks > 1, group consecutive weeks starting from target_week_id
            wk_names = [w for w, _ in iso_weeks]
            idx_w = wk_names.index(target_week_id)
            chosen_block = iso_weeks[idx_w : idx_w + window_w]
            target_week_id = chosen_block[-1][0]
            selected_dates = [d for _, dts in chosen_block for d in dts]
        ds = ds_design
    else:
        run_mode = "range"
        ds = load_dataset(
            data_dir,
            cfg,
            allow_holdout=False,
            auto_trim_holdout=(args.to_date is None),
            base_dir=base_dir,
            start_date=args.from_date,
            end_date=args.to_date,
        )
        selected_dates = list(ds.trading_days)

    # Determine Dynamic starting balance
    if run_mode == "weekly" and same_hash and prev_rec is not None:
        dyn_start_bal = to_decimal(prev_rec["dynamic_closing_balance"])
    else:
        dyn_start_bal = base_cap

    # Execute Fixed 1-Lot (parallel via fork+Pipe) & Dynamic Compounding (strictly serial)
    fixed_res, parallel_status = run_fixed_parallel(
        days=ds.days,
        trading_dates=selected_dates,
        strategy_source=strat_source,
        config=cfg,
        workers=int(cfg["workers"]),
        delay_d=0,
    )
    dynamic_res = run_track(
        days=ds.days,
        trading_dates=selected_dates,
        strategy_source=strat_source,
        config=cfg,
        track="dynamic",
        initial_balance=dyn_start_bal,
        delay_d=0,
    )
    dual = DualTrackResult(fixed=fixed_res, dynamic=dynamic_res)

    # Compute delay-1 and delay-2 for design-gate & holdout
    d1_f = d2_f = d1_d = d2_d = None
    if run_mode in ("design-gate", "holdout") and selected_dates:
        dual_d1 = run_dual_track(days=ds.days, trading_dates=selected_dates, strategy_source=strat_source, config=cfg, delay_d=1)
        dual_d2 = run_dual_track(days=ds.days, trading_dates=selected_dates, strategy_source=strat_source, config=cfg, delay_d=2)
        d1_f = round_paise(sum((t.net_base for t in dual_d1.fixed.trades), ZERO))
        d2_f = round_paise(sum((t.net_base for t in dual_d2.fixed.trades), ZERO))
        d1_d = round_paise(sum((t.net_base for t in dual_d1.dynamic.trades), ZERO))
        d2_d = round_paise(sum((t.net_base for t in dual_d2.dynamic.trades), ZERO))

    m_variants = count_family_variants(base_dir, strategy_family, strat_hash)
    k_opens = holdout_prior + holdout_opens_used + 1

    output_dir = _make_output_dir(output_root, strategy_family)
    # Write exact byte copy of executed strategy (§3.3)
    atomic_write_bytes(output_dir / "strategy_snapshot.py", strat_bytes)

    manifest: dict[str, Any] = {
        "engine_tag": ENGINE_TAG,
        "mode": run_mode,
        "strategy_family": strategy_family,
        "strategy_hash": strat_hash,
        "seed": int(cfg["seed"]),
        "m": m_variants,
        "K": k_opens if run_mode == "holdout" else holdout_prior,
        "git_check": git_check,
        "parallel": parallel_status,
        "peak_memory_mb": int(cfg["peak_memory_mb"]),
        "ruin_flag": dynamic_res.ruin_flag,
        "trading_dates": selected_dates,
        "exp_id": args.exp_id,
        "diagnosis": "PENDING",
    }

    summary = generate_reports(
        output_dir=output_dir,
        dual=dual,
        days=ds.days,
        trading_dates=selected_dates,
        config=cfg,
        manifest=manifest,
        dataset=ds,
        mode=run_mode,
        m_variants=m_variants,
        k_opens=k_opens,
        is_holdout=(run_mode == "holdout"),
        weekly_pass=weekly_unlocked,
        design_pass=design_unlocked,
        delay1_net_fixed=d1_f,
        delay2_net_fixed=d2_f,
        delay1_net_dynamic=d1_d,
        delay2_net_dynamic=d2_d,
    )

    # Update weekly_state.json for weekly / design-gate / holdout modes (§6.2)
    if run_mode == "weekly":
        wv = evaluate_weekly_verdict(summary["fixed"], summary["dynamic"], cfg)
        outcome = wv["outcome"]
        completed_list = list(prev_rec.get("weeks_completed_in_lineage", [])) if prev_rec else []
        if target_week_id not in completed_list:
            completed_list.append(target_week_id)

        if same_hash and prev_rec is not None:
            streak = int(prev_rec.get("streak", 0))
            paused = int(prev_rec.get("paused_weeks", 0))
            span_start = prev_rec.get("span_start_week")
            span_tf = int(prev_rec.get("span_trades_fixed", 0))
            span_td = int(prev_rec.get("span_trades_dynamic", 0))
            span_sf = to_decimal(prev_rec.get("span_stress_net_fixed", "0.00"))
            span_sd = to_decimal(prev_rec.get("span_stress_net_dynamic", "0.00"))
        else:
            streak = 0
            paused = 0
            span_start = None
            span_tf = 0
            span_td = 0
            span_sf = ZERO
            span_sd = ZERO

        wk_tf = int(summary["fixed"]["trades_count"])
        wk_td = int(summary["dynamic"]["trades_count"])
        wk_sf = to_decimal(summary["fixed"]["net_stress"])
        wk_sd = to_decimal(summary["dynamic"]["net_stress"])

        if outcome == "FAIL":
            streak = 0
            paused = 0
            span_start = None
            span_tf = 0
            span_td = 0
            span_sf = ZERO
            span_sd = ZERO
        elif outcome == "LOW_SAMPLE":
            paused += 1
            if paused > int(cfg["max_paused_weeks"]):
                streak = 0
                paused = 0
                span_start = None
                span_tf = 0
                span_td = 0
                span_sf = ZERO
                span_sd = ZERO
            elif span_start is not None:
                span_tf += wk_tf
                span_td += wk_td
                span_sf = round_paise(span_sf + wk_sf)
                span_sd = round_paise(span_sd + wk_sd)
        else:  # PASS
            if span_start is None:
                span_start = target_week_id
            streak += 1
            paused = 0
            span_tf += wk_tf
            span_td += wk_td
            span_sf = round_paise(span_sf + wk_sf)
            span_sd = round_paise(span_sd + wk_sd)

        new_weekly_pass = (
            streak >= int(cfg["final_streak_weeks"])
            and span_tf >= int(cfg["final_min_trades"])
            and span_td >= int(cfg["final_min_trades"])
            and span_sf > ZERO
            and span_sd > ZERO
        )
        append_weekly_record(
            base_dir,
            state,
            {
                "timestamp_ist": now_ist,
                "record_type": "WEEK_RUN",
                "strategy_family": strategy_family,
                "strategy_hash": strat_hash,
                "lineage_id": lineage_id,
                "week_id": target_week_id,
                "weeks_completed_in_lineage": completed_list,
                "outcome": outcome,
                "streak": streak,
                "paused_weeks": paused,
                "span_start_week": span_start,
                "span_end_week": target_week_id if span_start is not None else None,
                "span_trades_fixed": span_tf,
                "span_trades_dynamic": span_td,
                "span_stress_net_fixed": str(span_sf),
                "span_stress_net_dynamic": str(span_sd),
                "dynamic_closing_balance": summary["dynamic"]["final_balance_base"],
                "ruin_flag": dynamic_res.ruin_flag,
                "weekly_pass_unlocked": new_weekly_pass,
                "design_pass_unlocked": False,
                "holdout_opened": bool(prev_rec.get("holdout_opened", False)) if prev_rec else False,
            },
        )
    elif run_mode == "design-gate":
        dg_pass = summary["verdict"] == "PASS"
        append_weekly_record(
            base_dir,
            state,
            {
                "timestamp_ist": now_ist,
                "record_type": "DESIGN_GATE",
                "strategy_family": strategy_family,
                "strategy_hash": strat_hash,
                "lineage_id": lineage_id,
                "week_id": None,
                "weeks_completed_in_lineage": list(prev_rec.get("weeks_completed_in_lineage", [])) if prev_rec else [],
                "outcome": summary["verdict"],
                "streak": int(prev_rec.get("streak", 0)) if prev_rec else 0,
                "paused_weeks": int(prev_rec.get("paused_weeks", 0)) if prev_rec else 0,
                "span_start_week": prev_rec.get("span_start_week") if prev_rec else None,
                "span_end_week": prev_rec.get("span_end_week") if prev_rec else None,
                "span_trades_fixed": int(prev_rec.get("span_trades_fixed", 0)) if prev_rec else 0,
                "span_trades_dynamic": int(prev_rec.get("span_trades_dynamic", 0)) if prev_rec else 0,
                "span_stress_net_fixed": str(prev_rec.get("span_stress_net_fixed", "0.00")) if prev_rec else "0.00",
                "span_stress_net_dynamic": str(prev_rec.get("span_stress_net_dynamic", "0.00")) if prev_rec else "0.00",
                "dynamic_closing_balance": summary["dynamic"]["final_balance_base"],
                "ruin_flag": dynamic_res.ruin_flag,
                "weekly_pass_unlocked": True,
                "design_pass_unlocked": dg_pass,
                "holdout_opened": bool(prev_rec.get("holdout_opened", False)) if prev_rec else False,
            },
        )
    elif run_mode == "holdout":
        append_weekly_record(
            base_dir,
            state,
            {
                "timestamp_ist": now_ist,
                "record_type": "HOLDOUT",
                "strategy_family": strategy_family,
                "strategy_hash": strat_hash,
                "lineage_id": lineage_id,
                "week_id": None,
                "weeks_completed_in_lineage": list(prev_rec.get("weeks_completed_in_lineage", [])) if prev_rec else [],
                "outcome": summary["verdict"],
                "streak": int(prev_rec.get("streak", 0)) if prev_rec else 0,
                "paused_weeks": 0,
                "span_start_week": prev_rec.get("span_start_week") if prev_rec else None,
                "span_end_week": prev_rec.get("span_end_week") if prev_rec else None,
                "span_trades_fixed": int(prev_rec.get("span_trades_fixed", 0)) if prev_rec else 0,
                "span_trades_dynamic": int(prev_rec.get("span_trades_dynamic", 0)) if prev_rec else 0,
                "span_stress_net_fixed": str(prev_rec.get("span_stress_net_fixed", "0.00")) if prev_rec else "0.00",
                "span_stress_net_dynamic": str(prev_rec.get("span_stress_net_dynamic", "0.00")) if prev_rec else "0.00",
                "dynamic_closing_balance": summary["dynamic"]["final_balance_base"],
                "ruin_flag": dynamic_res.ruin_flag,
                "weekly_pass_unlocked": True,
                "design_pass_unlocked": True,
                "holdout_opened": True,
            },
        )

    # Record trial in trials.jsonl (parent-only, flush+fsync; §3.1)
    append_trial_record(
        base_dir,
        {
            "timestamp_ist": now_ist,
            "engine_tag": ENGINE_TAG,
            "mode": run_mode,
            "strategy_family": strategy_family,
            "strategy_hash": strat_hash,
            "m": m_variants,
            "K": manifest["K"],
            "output_dir": output_dir.name,
            "verdict": summary["verdict"],
            "fixed_net_base": summary["fixed"]["net_base"],
            "dynamic_net_base": summary["dynamic"]["net_base"],
            "exp_id": args.exp_id,
        },
    )

    # Fault-isolated diagnosis invocation (§12.2, T45)
    _run_diagnosis_isolated(
        output_dir=output_dir,
        dual=dual,
        days=ds.days,
        summary=summary,
        manifest=manifest,
        config=cfg,
        base_dir=base_dir,
        is_holdout=(run_mode == "holdout"),
        exp_id=args.exp_id,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
