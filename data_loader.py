"""updl dual-layout market data loader, day validator, and time-gated accessor.

Purpose:
    Loads BANKNIFTY 1-minute IST bars from updl week_* folders in Parquet or
    CSV format (with automatic layout selection and _from_ split CSV merging),
    enforces Niyam 1/2/11/12 (blind replay, bar-close visibility, stale->NaN,
    holdout guard), computes contract-driven expiry/DTE/lot_size per trading
    day, and exposes read-only time-gated numpy arrays.

PRD Sections:
    §2 (Niyam 1, 2, 11, 12), §3.1, §6.1, §10.1, §10.2, §12.3

Niyam Covered:
    - Niyam 1 : Blind replay (TimeGatedView slices strictly < t; i >= t raises LookaheadError)
    - Niyam 2 : Bar close semantics (avail_minute = ceil_minute(max(ts_open+1, available_from_ts)))
    - Niyam 11: Zero forward-fill (v==0, is_stale, is_filled -> NaN OHLC/OI, live=False)
    - Niyam 12: Holdout guard keyed on holdout_start with flush+fsync audit log

Guarding Tests:
    T01, T17, T25, T28, T29, T33, T37, T38, T39, T40, T41, T62

PUBLIC API:
    Constants:
        HAS_PYARROW, IST_TZ, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
        STANDARD_INDEX_MINUTES
    Dataclasses (frozen=True, slots=True):
        BarSeries, DayData, ExcludedDay, DatasetResult
    Classes:
        TimeGatedView
    Functions:
        parse_ist_timestamp, minute_to_hhmm, hhmm_to_minute, ceil_avail_minute,
        atomic_write_bytes, append_holdout_log, check_holdout_guard,
        export_csv_tables, save_day_cache, load_day_cache, get_prior_days,
        load_dataset
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date as dt_date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from contracts import DataMissingError, HoldoutLocked, LookaheadError

# Optional guarded pyarrow import (L06)
try:
    import pyarrow.parquet as _pq  # type: ignore[import-not-found]

    HAS_PYARROW: bool = True
except ImportError:
    _pq = None
    HAS_PYARROW = False

# ------------------------------------------------------------------------------
# Structural Constants (no tunable magic numbers; L12)
# ------------------------------------------------------------------------------
IST_TZ: timezone = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 15
MARKET_OPEN_TOTAL_MIN: int = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
# NSE cash index session spans 09:15..15:29 IST (inclusive); derived from clock
# boundaries rather than a hardcoded bar count. F&O grid length is always
# data-driven from actual F&O bars (§10.1, T38).
INDEX_CLOSE_HOUR: int = 15
INDEX_CLOSE_MINUTE: int = 29
STANDARD_INDEX_MINUTES: int = (INDEX_CLOSE_HOUR * 60 + INDEX_CLOSE_MINUTE) - MARKET_OPEN_TOTAL_MIN + 1
_WEEKDAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_OPT_CSV_RE: re.Pattern[str] = re.compile(
    r"^(?P<expiry>\d{4}-\d{2}-\d{2})(?:_from_\d{4}-\d{2}-\d{2})?\.csv(?:\.gz)?$"
)


# ------------------------------------------------------------------------------
# IST Time Helpers (§10.1, AUD-015)
# ------------------------------------------------------------------------------
def hhmm_to_minute(hhmm: str) -> int:
    """Convert 'HH:MM' IST string to integer minute offset from 09:15 IST."""
    parts = hhmm.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time string: {hhmm!r}")
    h, m = int(parts[0]), int(parts[1])
    return (h * 60 + m) - MARKET_OPEN_TOTAL_MIN


def minute_to_hhmm(minute_offset: int) -> str:
    """Convert integer minute offset from 09:15 IST to 'HH:MM' string."""
    total = MARKET_OPEN_TOTAL_MIN + int(minute_offset)
    h, m = divmod(total, 60)
    return f"{h:02d}:{m:02d}"


def _to_ist_datetime(raw: Any) -> datetime:
    """Parse ISO string or datetime into timezone-aware IST datetime."""
    if isinstance(raw, datetime):
        dt = raw
    elif hasattr(raw, "to_pydatetime"):
        dt = raw.to_pydatetime()
    else:
        s = str(raw).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST_TZ)
    return dt.astimezone(IST_TZ)


def parse_ist_timestamp(raw: Any) -> tuple[str, int]:
    """Normalize timestamp to (YYYY-MM-DD, integer minute offset from 09:15 IST)."""
    dt = _to_ist_datetime(raw)
    date_str = dt.strftime("%Y-%m-%d")
    minute_offset = (dt.hour * 60 + dt.minute) - MARKET_OPEN_TOTAL_MIN
    return date_str, minute_offset


def ceil_avail_minute(ts_open_min: int, available_from_ts: Any, day_date: str) -> int:
    """Compute Niyam 2 availability minute: ceil_minute(max(ts_open+1, available_from_ts))."""
    base_avail = ts_open_min + 1
    if available_from_ts is None:
        return base_avail
    if isinstance(available_from_ts, float) and not math.isfinite(available_from_ts):
        return base_avail
    s = str(available_from_ts).strip()
    if s == "" or s.lower() in ("nan", "nat", "none", "null"):
        return base_avail
    dt = _to_ist_datetime(available_from_ts)
    if dt.strftime("%Y-%m-%d") > day_date:
        # Available on a future calendar date -> never visible on day_date
        return 10**6
    if dt.strftime("%Y-%m-%d") < day_date:
        return base_avail
    avail_min = (dt.hour * 60 + dt.minute) - MARKET_OPEN_TOTAL_MIN
    if dt.second > 0 or dt.microsecond > 0:
        avail_min += 1
    return max(base_avail, avail_min)


# ------------------------------------------------------------------------------
# Atomic I/O & Holdout Audit Log (§10.1, §12.3, Niyam 12)
# ------------------------------------------------------------------------------
def atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Write bytes atomically via same-directory .tmp.<pid>.<name> + fsync + replace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.parent / f".tmp.{os.getpid()}.{dest.name}"
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def append_holdout_log(base_dir: Path, event: Mapping[str, Any]) -> None:
    """Append a single JSON audit line to <base_dir>/holdout_access.log with flush+fsync."""
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "holdout_access.log"
    line = json.dumps(dict(event), sort_keys=True, separators=(",", ":")) + "\n"
    with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def check_holdout_guard(
    *,
    dates: list[str],
    holdout_start: str,
    allow_holdout: bool,
    base_dir: Path | None,
    reason: str = "load_dataset",
    auto_trim: bool = True,
) -> list[str]:
    """Enforce Niyam 12 holdout boundary keyed strictly on holdout_start.

    When allow_holdout is False:
      - If auto_trim is False and any date >= holdout_start: logs refusal and
        raises HoldoutLocked.
      - If auto_trim is True: silently returns only dates < holdout_start.
    When allow_holdout is True:
      - Logs authorized holdout access if any date >= holdout_start.
    """
    touching = [d for d in dates if d >= holdout_start]
    if not touching:
        return list(dates)

    if not allow_holdout:
        if auto_trim:
            return [d for d in dates if d < holdout_start]
        if base_dir is not None:
            append_holdout_log(
                base_dir,
                {
                    "action": "REFUSED",
                    "reason": reason,
                    "holdout_start": holdout_start,
                    "first_touched": touching[0],
                    "last_touched": touching[-1],
                    "count": len(touching),
                },
            )
        raise HoldoutLocked(
            f"Holdout data (>= {holdout_start}) is locked; requested {touching[0]}..{touching[-1]}"
        )

    if base_dir is not None:
        append_holdout_log(
            base_dir,
            {
                "action": "ALLOWED",
                "reason": reason,
                "holdout_start": holdout_start,
                "first_touched": touching[0],
                "last_touched": touching[-1],
                "count": len(touching),
            },
        )
    return list(dates)


# ------------------------------------------------------------------------------
# Read-Only Array Helpers & Dataclasses (§10.1, §11.1)
# ------------------------------------------------------------------------------
def _ro_array(arr: np.ndarray) -> np.ndarray:
    """Mark numpy array read-only (writeable=False) and return it."""
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, slots=True)
class BarSeries:
    """Read-only 1-minute bar arrays for a single instrument on one trading day."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    oi: np.ndarray
    live: np.ndarray
    avail_minute: np.ndarray
    half_spread_proxy: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.close.shape[0])

    def slice_visible(self, t: int) -> BarSeries:
        """Return read-only BarSeries visible strictly before decision minute t (Niyam 1, 2)."""
        n = min(max(int(t), 0), len(self))
        av = self.avail_minute[:n]
        if n == 0 or bool(np.all(av <= t)):
            return BarSeries(
                open=_ro_array(self.open[:n]),
                high=_ro_array(self.high[:n]),
                low=_ro_array(self.low[:n]),
                close=_ro_array(self.close[:n]),
                volume=_ro_array(self.volume[:n]),
                oi=_ro_array(self.oi[:n]),
                live=_ro_array(self.live[:n]),
                avail_minute=_ro_array(av),
                half_spread_proxy=(
                    _ro_array(self.half_spread_proxy[:n])
                    if self.half_spread_proxy is not None
                    else None
                ),
            )
        # Mask bars whose avail_minute > t (Niyam 2)
        hidden = av > t
        o = self.open[:n].copy()
        h = self.high[:n].copy()
        l = self.low[:n].copy()
        c = self.close[:n].copy()
        v = self.volume[:n].copy()
        oi = self.oi[:n].copy()
        lv = self.live[:n].copy()
        o[hidden] = np.nan
        h[hidden] = np.nan
        l[hidden] = np.nan
        c[hidden] = np.nan
        v[hidden] = np.nan
        oi[hidden] = np.nan
        lv[hidden] = False
        hsp = None
        if self.half_spread_proxy is not None:
            hsp = self.half_spread_proxy[:n].copy()
            hsp[hidden] = np.nan
            hsp = _ro_array(hsp)
        return BarSeries(
            open=_ro_array(o),
            high=_ro_array(h),
            low=_ro_array(l),
            close=_ro_array(c),
            volume=_ro_array(v),
            oi=_ro_array(oi),
            live=_ro_array(lv),
            avail_minute=_ro_array(av.copy()),
            half_spread_proxy=hsp,
        )


@dataclass(frozen=True, slots=True)
class ExcludedDay:
    """Record of a market day excluded by §10.2 validation rules."""

    date: str
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DayData:
    """Validated, read-only single trading day market data."""

    date: str
    weekday: str
    grid_len: int
    expiry: str
    dte: int
    dte_cal: int
    dte_source: str
    lot_size: int
    is_expiry_day: bool
    ref_spot: float
    index: BarSeries
    vix: BarSeries | None
    futures: BarSeries | None
    options: Mapping[tuple[int, str], BarSeries]
    stocks: Mapping[str, BarSeries]
    strikes: tuple[int, ...]
    has_vix: bool
    has_futures: bool
    has_greeks: bool
    has_stocks: bool


@dataclass(frozen=True, slots=True)
class DatasetResult:
    """Loaded dataset with full day taxonomy (§10.1, §10.2)."""

    days: Mapping[str, DayData]
    calendar_days: tuple[str, ...]
    market_days: tuple[str, ...]
    excluded_days: tuple[ExcludedDay, ...]
    warmup_days: tuple[str, ...]
    trading_days: tuple[str, ...]
    holdout_days: tuple[str, ...]


# ------------------------------------------------------------------------------
# Time-Gated Read-Only Accessor (§10.1, §11.1, Niyam 1)
# ------------------------------------------------------------------------------
class TimeGatedView:
    """Read-only MarketView enforcing Niyam 1 (strictly < t) and Niyam 2 (avail_minute <= t)."""

    __slots__ = ("_day", "_t", "_chain_band_pct", "_cache_band_pct", "ind")

    def __init__(
        self,
        day: DayData,
        t: int,
        chain_band_pct: float,
        cache_band_pct: float,
        ind: Any = None,
    ) -> None:
        self._day = day
        self._t = int(t)
        self._chain_band_pct = float(chain_band_pct)
        self._cache_band_pct = float(cache_band_pct)
        self.ind = ind

    def index(self) -> BarSeries:
        """Return read-only index bars strictly before t."""
        return self._day.index.slice_visible(self._t)

    def vix(self) -> BarSeries:
        """Return read-only VIX bars strictly before t, or raise DataMissingError."""
        if self._day.vix is None:
            raise DataMissingError("data_missing: vix_1m")
        return self._day.vix.slice_visible(self._t)

    def futures(self) -> BarSeries:
        """Return read-only futures bars strictly before t, or raise DataMissingError."""
        if self._day.futures is None:
            raise DataMissingError("data_missing: fut_1m")
        return self._day.futures.slice_visible(self._t)

    def option(self, strike: int, side: str) -> BarSeries:
        """Return read-only option bars strictly before t for (strike, side)."""
        key = (int(strike), str(side))
        if key not in self._day.options:
            raise KeyError(f"Option ({strike}, {side}) not in cache band for {self._day.date}")
        return self._day.options[key].slice_visible(self._t)

    def stocks(self, symbol: str) -> BarSeries:
        """Return read-only constituent stock bars strictly before t."""
        if not self._day.has_stocks:
            raise DataMissingError("data_missing: stock_1m")
        sym = str(symbol)
        if sym not in self._day.stocks:
            raise DataMissingError(f"data_missing: stock_1m ({sym})")
        return self._day.stocks[sym].slice_visible(self._t)

    def get_chain(
        self,
        i: int,
        band_pct: float | None = None,
        asof: bool = False,
    ) -> dict[tuple[int, str], dict[str, Any]]:
        """Return option chain snapshot at bar i (< t), optionally carrying last-live asof."""
        # Niyam 1: forbid i >= t or negative index
        if isinstance(i, bool) or not isinstance(i, int):
            raise TypeError(f"Bar index i must be int, got {type(i).__name__}")
        if i >= self._t or i < 0:
            raise LookaheadError(
                f"LookaheadError: get_chain(i={i}) forbidden at decision minute t={self._t}"
            )
        if i >= self._day.grid_len:
            raise LookaheadError(
                f"LookaheadError: bar index {i} exceeds grid_len {self._day.grid_len}"
            )

        bp = self._chain_band_pct if band_pct is None else float(band_pct)
        if not math.isfinite(bp) or bp <= 0.0 or bp > self._cache_band_pct:
            raise ValueError(
                f"band_pct ({bp}) must be in (0, cache_band_pct={self._cache_band_pct}]"
            )

        # Reference spot strictly <= i (Niyam 1)
        idx_vis = self._day.index.slice_visible(self._t)
        spot_ref = self._day.ref_spot
        for j in range(min(i, len(idx_vis) - 1), -1, -1):
            if bool(idx_vis.live[j]) and math.isfinite(float(idx_vis.close[j])):
                spot_ref = float(idx_vis.close[j])
                break

        lo_strike = spot_ref * (1.0 - bp / 100.0)
        hi_strike = spot_ref * (1.0 + bp / 100.0)

        out: dict[tuple[int, str], dict[str, Any]] = {}
        for strike in self._day.strikes:
            if strike < lo_strike or strike > hi_strike:
                continue
            for side in ("CE", "PE"):
                key = (strike, side)
                series = self._day.options.get(key)
                if series is None:
                    continue
                if bool(series.live[i]) and int(series.avail_minute[i]) <= self._t:
                    out[key] = {
                        "open": float(series.open[i]),
                        "high": float(series.high[i]),
                        "low": float(series.low[i]),
                        "close": float(series.close[i]),
                        "volume": float(series.volume[i]),
                        "oi": float(series.oi[i]),
                        "live": True,
                        "age_min": 0,
                    }
                elif asof:
                    found_j: int | None = None
                    for j in range(i - 1, -1, -1):
                        if bool(series.live[j]) and int(series.avail_minute[j]) <= self._t:
                            found_j = j
                            break
                    if found_j is not None:
                        out[key] = {
                            "open": float(series.open[found_j]),
                            "high": float(series.high[found_j]),
                            "low": float(series.low[found_j]),
                            "close": float(series.close[found_j]),
                            "volume": float(series.volume[found_j]),
                            "oi": float(series.oi[found_j]),
                            "live": False,
                            "age_min": int(i - found_j),
                        }
                    else:
                        out[key] = {
                            "open": float("nan"),
                            "high": float("nan"),
                            "low": float("nan"),
                            "close": float("nan"),
                            "volume": float("nan"),
                            "oi": float("nan"),
                            "live": False,
                            "age_min": None,
                        }
                else:
                    out[key] = {
                        "open": float("nan"),
                        "high": float("nan"),
                        "low": float("nan"),
                        "close": float("nan"),
                        "volume": float("nan"),
                        "oi": float("nan"),
                        "live": False,
                        "age_min": None,
                    }
        return out


# ------------------------------------------------------------------------------
# prior_days Isolation Helper (§10.1, T17)
# ------------------------------------------------------------------------------
def get_prior_days(
    days: Mapping[str, DayData],
    today: str,
    warmup_days: int,
    holdout_start: str | None = None,
) -> tuple[DayData, ...]:
    """Return strictly the preceding warmup_days valid DayData objects (< today).

    Excluded days are absent from `days`. Design days (< holdout_start) never
    receive holdout days; holdout days may read design days.
    """
    is_design = holdout_start is not None and today < holdout_start
    eligible = [
        d
        for d in sorted(days.keys())
        if d < today and (not is_design or d < holdout_start)
    ]
    selected = eligible[-warmup_days:] if warmup_days > 0 else []
    return tuple(days[d] for d in selected)


# ------------------------------------------------------------------------------
# CSV / Parquet Low-Level Readers & export-csv Helper (§10.1)
# ------------------------------------------------------------------------------
def _open_text_auto(path: Path):
    """Open plain .csv or .csv.gz as UTF-8 text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    """Read a CSV (.csv/.csv.gz) or Parquet file into list of row dicts."""
    if path.suffix == ".parquet":
        if not HAS_PYARROW or _pq is None:
            raise RuntimeError("pyarrow is required to read Parquet files")
        table = _pq.read_table(path)
        return table.to_pylist()
    with _open_text_auto(path) as fh:
        return list(csv.DictReader(fh))


def _resolve_col(row: Mapping[str, Any], canonical: str, col_map: Mapping[str, str]) -> Any:
    """Fetch field using column_map alias first, then canonical name."""
    mapped = col_map.get(canonical, canonical)
    if mapped in row:
        return row[mapped]
    return row.get(canonical)


def _to_float(val: Any) -> float:
    """Convert raw scalar to plain Python float (NaN on empty/None/invalid)."""
    if val is None:
        return float("nan")
    if isinstance(val, bool):
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "na"):
        return float("nan")
    return float(s)


def _to_bool(val: Any) -> bool:
    """Convert raw CSV/Parquet flag to plain Python bool."""
    if val is None:
        return False
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "t")


def export_csv_tables(data_dir: Path) -> list[Path]:
    """Convert Parquet vix_1m, iv_greeks_1m, stock_1m to deterministic .csv.gz (§10.1)."""
    if not HAS_PYARROW or _pq is None:
        raise RuntimeError("pyarrow is required for export-csv")
    exported: list[Path] = []
    folders = _discover_week_folders(data_dir)
    for folder in folders:
        store = folder / "store"
        if not store.is_dir():
            continue
        out_dir = folder / "BANKNIFTY"
        out_dir.mkdir(parents=True, exist_ok=True)
        for tbl in ("vix_1m", "iv_greeks_1m", "stock_1m"):
            pq_path = store / f"{tbl}.parquet"
            if not pq_path.is_file():
                continue
            rows = _pq.read_table(pq_path).to_pylist()
            cols = list(rows[0].keys()) if rows else []
            buf_lines: list[str] = []
            if cols:
                from io import StringIO

                sio = StringIO()
                writer = csv.DictWriter(sio, fieldnames=cols, lineterminator="\n")
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: ("" if r[k] is None else str(r[k])) for k in cols})
                buf_lines.append(sio.getvalue())
            raw_csv = "".join(buf_lines).encode("utf-8")
            gz_bytes = gzip.compress(raw_csv, mtime=0)
            dest = out_dir / f"{tbl}.csv.gz"
            atomic_write_bytes(dest, gz_bytes)
            exported.append(dest)
    return exported


# ------------------------------------------------------------------------------
# Atomic Cache Helpers (§3.1, §12.3)
# ------------------------------------------------------------------------------
def _cache_key(day: DayData) -> str:
    """Deterministic cache filename stem for a DayData."""
    h = hashlib.sha256(f"{day.date}|{day.expiry}|{day.lot_size}|{day.grid_len}".encode("utf-8")).hexdigest()[:12]
    return f"day_{day.date}_{h}"


def save_day_cache(cache_dir: Path, day: DayData) -> Path:
    """Atomically persist DayData index/option arrays and metadata in cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = _cache_key(day)
    meta = {
        "date": day.date,
        "weekday": day.weekday,
        "grid_len": day.grid_len,
        "expiry": day.expiry,
        "dte": day.dte,
        "dte_cal": day.dte_cal,
        "dte_source": day.dte_source,
        "lot_size": day.lot_size,
        "is_expiry_day": day.is_expiry_day,
        "ref_spot": day.ref_spot,
        "strikes": list(day.strikes),
        "has_vix": day.has_vix,
        "has_futures": day.has_futures,
        "has_greeks": day.has_greeks,
        "has_stocks": day.has_stocks,
        "option_keys": [[k[0], k[1]] for k in sorted(day.options.keys())],
        "stock_keys": sorted(day.stocks.keys()),
    }
    meta_path = cache_dir / f"{stem}.json"
    atomic_write_bytes(
        meta_path,
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    arrays: dict[str, np.ndarray] = {}

    def _pack_series(prefix: str, s: BarSeries) -> None:
        arrays[f"{prefix}__open"] = s.open
        arrays[f"{prefix}__high"] = s.high
        arrays[f"{prefix}__low"] = s.low
        arrays[f"{prefix}__close"] = s.close
        arrays[f"{prefix}__volume"] = s.volume
        arrays[f"{prefix}__oi"] = s.oi
        arrays[f"{prefix}__live"] = s.live
        arrays[f"{prefix}__avail"] = s.avail_minute
        if s.half_spread_proxy is not None:
            arrays[f"{prefix}__hsp"] = s.half_spread_proxy

    _pack_series("index", day.index)
    if day.vix is not None:
        _pack_series("vix", day.vix)
    if day.futures is not None:
        _pack_series("fut", day.futures)
    for strike, side in sorted(day.options.keys()):
        _pack_series(f"opt_{strike}_{side}", day.options[(strike, side)])
    for sym in sorted(day.stocks.keys()):
        _pack_series(f"stk_{sym}", day.stocks[sym])

    from io import BytesIO

    bio = BytesIO()
    np.savez(bio, **arrays)
    npz_path = cache_dir / f"{stem}.npz"
    atomic_write_bytes(npz_path, bio.getvalue())
    return npz_path


def load_day_cache(meta_path: Path) -> DayData:
    """Load DayData from cache files with all numpy arrays marked writeable=False."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    npz_path = meta_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as z:

        def _unpack(prefix: str) -> BarSeries:
            hsp_key = f"{prefix}__hsp"
            return BarSeries(
                open=_ro_array(z[f"{prefix}__open"].copy()),
                high=_ro_array(z[f"{prefix}__high"].copy()),
                low=_ro_array(z[f"{prefix}__low"].copy()),
                close=_ro_array(z[f"{prefix}__close"].copy()),
                volume=_ro_array(z[f"{prefix}__volume"].copy()),
                oi=_ro_array(z[f"{prefix}__oi"].copy()),
                live=_ro_array(z[f"{prefix}__live"].copy()),
                avail_minute=_ro_array(z[f"{prefix}__avail"].copy()),
                half_spread_proxy=_ro_array(z[hsp_key].copy()) if hsp_key in z else None,
            )

        idx = _unpack("index")
        vix = _unpack("vix") if meta["has_vix"] else None
        fut = _unpack("fut") if meta["has_futures"] else None
        opts: dict[tuple[int, str], BarSeries] = {}
        for strike, side in meta["option_keys"]:
            opts[(int(strike), str(side))] = _unpack(f"opt_{strike}_{side}")
        stks: dict[str, BarSeries] = {}
        for sym in meta["stock_keys"]:
            stks[str(sym)] = _unpack(f"stk_{sym}")

    return DayData(
        date=str(meta["date"]),
        weekday=str(meta["weekday"]),
        grid_len=int(meta["grid_len"]),
        expiry=str(meta["expiry"]),
        dte=int(meta["dte"]),
        dte_cal=int(meta["dte_cal"]),
        dte_source=str(meta["dte_source"]),
        lot_size=int(meta["lot_size"]),
        is_expiry_day=bool(meta["is_expiry_day"]),
        ref_spot=float(meta["ref_spot"]),
        index=idx,
        vix=vix,
        futures=fut,
        options=MappingProxyType(opts),
        stocks=MappingProxyType(stks),
        strikes=tuple(int(s) for s in meta["strikes"]),
        has_vix=bool(meta["has_vix"]),
        has_futures=bool(meta["has_futures"]),
        has_greeks=bool(meta["has_greeks"]),
        has_stocks=bool(meta["has_stocks"]),
    )


# ------------------------------------------------------------------------------
# Folder Discovery, Row Merging & Bar Construction (§10.1, §10.2)
# ------------------------------------------------------------------------------
def _discover_week_folders(data_dir: Path) -> list[Path]:
    """Return sorted updl week_* folders under data_dir (or [data_dir] if direct)."""
    if not data_dir.exists():
        return []
    weeks = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("week_")
    )
    if weeks:
        return weeks
    if (data_dir / "store").is_dir() or (data_dir / "BANKNIFTY").is_dir():
        return [data_dir]
    return []


def _choose_layout(folder: Path, data_layout: str) -> str:
    """Resolve effective layout ('parquet' or 'csv') for one folder."""
    has_store = (folder / "store").is_dir()
    has_csv = (folder / "BANKNIFTY").is_dir()
    if data_layout == "parquet":
        if not HAS_PYARROW:
            raise RuntimeError("data_layout='parquet' requested but pyarrow is not installed")
        if not has_store:
            raise FileNotFoundError(f"Parquet store/ missing in {folder}")
        return "parquet"
    if data_layout == "csv":
        if not has_csv:
            raise FileNotFoundError(f"CSV BANKNIFTY/ missing in {folder}")
        return "csv"
    # "auto": prefer Parquet if pyarrow installed and store/ exists
    if HAS_PYARROW and has_store:
        return "parquet"
    if has_csv:
        return "csv"
    if has_store:
        raise RuntimeError(f"Found store/ in {folder} but pyarrow is not installed and BANKNIFTY/ CSV is absent")
    return "csv"


def _row_values_equal(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    """Compare two normalized row tuples with NaN == NaN."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, float) and isinstance(y, float):
            if math.isnan(x) and math.isnan(y):
                continue
            if abs(x - y) > 1e-9:
                return False
        elif x != y:
            return False
    return True


def _normalize_bar_row(
    row: Mapping[str, Any],
    col_map: Mapping[str, str],
) -> tuple[str, int, float, float, float, float, float, float, bool, bool, int, str]:
    """Normalize a single OHLCV row into a canonical tuple."""
    ts_raw = _resolve_col(row, "ts_open", col_map)
    day_str, m_off = parse_ist_timestamp(ts_raw)
    o = _to_float(_resolve_col(row, "open", col_map))
    h = _to_float(_resolve_col(row, "high", col_map))
    l = _to_float(_resolve_col(row, "low", col_map))
    c = _to_float(_resolve_col(row, "close", col_map))
    v = _to_float(_resolve_col(row, "volume", col_map))
    oi_raw = _resolve_col(row, "oi", col_map)
    oi = _to_float(oi_raw) if oi_raw is not None else float("nan")
    is_stale = _to_bool(_resolve_col(row, "is_stale", col_map))
    is_filled = _to_bool(_resolve_col(row, "is_filled", col_map))
    avail_min = ceil_avail_minute(m_off, _resolve_col(row, "available_from_ts", col_map), day_str)
    sess_raw = _resolve_col(row, "session_type", col_map)
    session_type = "NORMAL" if sess_raw is None or str(sess_raw).strip() == "" else str(sess_raw).strip().upper()
    return (day_str, m_off, o, h, l, c, v, oi, is_stale, is_filled, avail_min, session_type)


def _ingest_series_rows(
    raw_rows: list[dict[str, Any]],
    col_map: Mapping[str, str],
    store: dict[str, dict[int, tuple[Any, ...]]],
    bad_days: dict[str, tuple[str, str]],
) -> None:
    """Ingest rows from one file into date -> minute -> tuple map with intra-file order & conflict checks."""
    # Track intra-file timestamp order per date (§10.2 Rule 1)
    last_min_in_file: dict[str, int] = {}
    seen_in_file: set[tuple[str, int]] = set()
    for r in raw_rows:
        norm = _normalize_bar_row(r, col_map)
        day_str, m_off = norm[0], norm[1]
        if (day_str, m_off) in seen_in_file:
            bad_days.setdefault(day_str, ("duplicate_or_unsorted_ts", f"duplicate minute {m_off}"))
            continue
        prev_m = last_min_in_file.get(day_str)
        if prev_m is not None and m_off <= prev_m:
            bad_days.setdefault(day_str, ("duplicate_or_unsorted_ts", f"out-of-order minute {m_off} <= {prev_m}"))
        last_min_in_file[day_str] = m_off
        seen_in_file.add((day_str, m_off))

        # Check session_type != NORMAL (§10.2 Rule 4)
        if norm[11] != "NORMAL":
            bad_days.setdefault(day_str, ("special_session", f"session_type={norm[11]}"))

        # Merge across folders: exact duplicate dedupes, conflicting values exclude day
        day_map = store.setdefault(day_str, {})
        payload = norm[2:11]
        if m_off in day_map:
            if not _row_values_equal(day_map[m_off], payload):
                bad_days.setdefault(day_str, ("data_conflict", f"conflicting bar at minute {m_off}"))
        else:
            day_map[m_off] = payload


def _build_bar_series(
    minute_map: Mapping[int, tuple[Any, ...]],
    grid_len: int,
    *,
    is_index_or_vix: bool = False,
    greeks_map: Mapping[int, tuple[float, int]] | None = None,
) -> tuple[BarSeries, str | None]:
    """Convert minute_map into read-only BarSeries (Niyam 11: stale/filled -> NaN).

    Returns (BarSeries, ohlc_error_or_None).
    """
    o = np.full(grid_len, np.nan, dtype=np.float64)
    h = np.full(grid_len, np.nan, dtype=np.float64)
    l = np.full(grid_len, np.nan, dtype=np.float64)
    c = np.full(grid_len, np.nan, dtype=np.float64)
    v = np.full(grid_len, np.nan, dtype=np.float64)
    oi = np.full(grid_len, np.nan, dtype=np.float64)
    live = np.zeros(grid_len, dtype=np.bool_)
    avail = np.arange(1, grid_len + 1, dtype=np.int32)
    hsp = np.full(grid_len, np.nan, dtype=np.float64) if greeks_map is not None else None

    ohlc_err: str | None = None
    for m_off, vals in minute_map.items():
        if m_off < 0 or m_off >= grid_len:
            continue
        # Index and VIX bars after 15:29 are NaN (§10.1)
        if is_index_or_vix and m_off >= STANDARD_INDEX_MINUTES:
            continue
        ro, rh, rl, rc, rv, roi, is_stale, is_filled, av_min = vals
        avail[m_off] = int(av_min)

        # Check live-bar definition (§10.1, Niyam 11)
        # Note: index/VIX cash bars may have volume==0 or NaN in some feeds;
        # for index/VIX, non-stale & non-filled & finite positive OHLC is live.
        vol_ok = True if is_index_or_vix else (math.isfinite(rv) and rv > 0.0)
        if is_stale or is_filled or not vol_ok:
            # Niyam 11: overwrite OHLC & OI with NaN, live = False
            v[m_off] = 0.0 if (math.isfinite(rv) and rv == 0.0) else np.nan
            continue

        # Validate positive OHLC & h >= l & o,c in [l, h] (§10.2 Rule 2)
        if not (
            math.isfinite(ro)
            and math.isfinite(rh)
            and math.isfinite(rl)
            and math.isfinite(rc)
        ):
            ohlc_err = f"non-finite OHLC at minute {m_off}"
            continue
        if ro <= 0.0 or rh <= 0.0 or rl <= 0.0 or rc <= 0.0:
            ohlc_err = f"non-positive OHLC at minute {m_off}"
            continue
        if rh < rl or ro < rl or ro > rh or rc < rl or rc > rh:
            ohlc_err = f"OHLC invariant violated at minute {m_off}"
            continue

        o[m_off] = ro
        h[m_off] = rh
        l[m_off] = rl
        c[m_off] = rc
        v[m_off] = rv if math.isfinite(rv) else 0.0
        oi[m_off] = roi
        live[m_off] = True

    if hsp is not None and greeks_map is not None:
        for m_off, (proxy_val, g_avail) in greeks_map.items():
            if 0 <= m_off < grid_len and math.isfinite(proxy_val):
                hsp[m_off] = proxy_val
                if g_avail > int(avail[m_off]):
                    avail[m_off] = int(g_avail)

    series = BarSeries(
        open=_ro_array(o),
        high=_ro_array(h),
        low=_ro_array(l),
        close=_ro_array(c),
        volume=_ro_array(v),
        oi=_ro_array(oi),
        live=_ro_array(live),
        avail_minute=_ro_array(avail),
        half_spread_proxy=_ro_array(hsp) if hsp is not None else None,
    )
    return series, ohlc_err


def _resolve_contract_meta(
    day_str: str,
    contracts: Mapping[str, int | None],
    lot_size_fallback: list[dict[str, Any]],
    all_valid_market_dates: list[str],
    nse_holidays: set[str],
) -> tuple[str | None, int | None, int | None, str, int | None]:
    """Resolve (expiry, dte, dte_cal, dte_source, lot_size) for day_str (§10.1, T39, T62)."""
    expiries = sorted(e for e in contracts.keys() if e >= day_str)
    if not expiries:
        return None, None, None, "missing", None
    expiry = expiries[0]

    # Resolve lot_size: contracts.csv primary, dated config fallback secondary
    lot_sz = contracts.get(expiry)
    if lot_sz is None or lot_sz <= 0:
        lot_sz = None
        for entry in sorted(lot_size_fallback, key=lambda x: str(x.get("effective_from", ""))):
            if str(entry.get("effective_from", "")) <= day_str:
                cand = int(entry["lot_size"])
                if cand > 0:
                    lot_sz = cand

    d_today = dt_date.fromisoformat(day_str)
    d_exp = dt_date.fromisoformat(expiry)
    dte_cal = (d_exp - d_today).days

    if expiry == day_str:
        return expiry, 0, 0, "market_days", lot_sz

    if all_valid_market_dates and expiry <= all_valid_market_dates[-1]:
        dte = sum(1 for d in all_valid_market_dates if day_str < d <= expiry)
        return expiry, dte, dte_cal, "market_days", lot_sz

    # Count Mon..Fri weekdays strictly > day_str and <= expiry minus nse_holidays
    count = 0
    cur = d_today + timedelta(days=1)
    while cur <= d_exp:
        if cur.weekday() < 5 and cur.isoformat() not in nse_holidays:
            count += 1
        cur += timedelta(days=1)
    dte_source = "nse_holidays" if nse_holidays else "weekday_estimate"
    return expiry, count, dte_cal, dte_source, lot_sz


# ------------------------------------------------------------------------------
# Main Dataset Loader (§10.1, §10.2)
# ------------------------------------------------------------------------------
def load_dataset(
    data_dir: Path,
    config: Mapping[str, Any],
    *,
    allow_holdout: bool = False,
    auto_trim_holdout: bool = True,
    base_dir: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> DatasetResult:
    """Load and validate all updl week_* folders under data_dir into a DatasetResult."""
    col_map: Mapping[str, str] = config.get("column_map", {})
    data_layout = str(config.get("data_layout", "auto"))
    coverage_min = float(config.get("coverage_min", 0.95))
    cache_band_pct = float(config.get("cache_band_pct", 6.0))
    warmup_n = int(config.get("warmup_days", 20))
    holdout_start = str(config.get("holdout_start", "2099-12-31"))
    nse_holidays = set(str(x) for x in config.get("nse_holidays", []))
    lot_size_fallback: list[dict[str, Any]] = list(config.get("lot_size", []))

    folders = _discover_week_folders(data_dir)

    # Raw merged stores keyed by date -> minute -> payload
    spot_store: dict[str, dict[int, tuple[Any, ...]]] = {}
    vix_store: dict[str, dict[int, tuple[Any, ...]]] = {}
    fut_store: dict[str, dict[int, tuple[Any, ...]]] = {}
    # options: expiry -> (strike, side) -> date -> minute -> payload
    opt_store: dict[str, dict[tuple[int, str], dict[str, dict[int, tuple[Any, ...]]]]] = {}
    # greeks: expiry -> (strike, side) -> date -> minute -> (half_spread_proxy, avail_min)
    greeks_store: dict[str, dict[tuple[int, str], dict[str, dict[int, tuple[float, int]]]]] = {}
    # stocks: symbol -> date -> minute -> payload
    stock_store: dict[str, dict[str, dict[int, tuple[Any, ...]]]] = {}
    # contracts: expiry -> lot_size (None if missing)
    contracts: dict[str, int | None] = {}
    bad_days: dict[str, tuple[str, str]] = {}

    for folder in folders:
        layout = _choose_layout(folder, data_layout)
        if layout == "parquet":
            base = folder / "store"
            spot_path = base / "spot_1m.parquet"
            if spot_path.is_file():
                _ingest_series_rows(_read_table_rows(spot_path), col_map, spot_store, bad_days)
            vix_path = base / "vix_1m.parquet"
            if vix_path.is_file():
                _ingest_series_rows(_read_table_rows(vix_path), col_map, vix_store, bad_days)
            fut_path = base / "fut_1m.parquet"
            if fut_path.is_file():
                _ingest_series_rows(_read_table_rows(fut_path), col_map, fut_store, bad_days)
            for opt_name in ("opt_1m.parquet", "opt_wide_1m.parquet"):
                opt_path = base / opt_name
                if opt_path.is_file():
                    rows = _read_table_rows(opt_path)
                    # Group by (expiry, strike, side) preserving intra-file order
                    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
                    for r in rows:
                        exp = str(_resolve_col(r, "expiry", col_map)).strip()
                        strike = int(float(_resolve_col(r, "strike", col_map)))
                        side = str(_resolve_col(r, "option_type", col_map)).strip().upper()
                        grouped.setdefault((exp, strike, side), []).append(r)
                    for (exp, strike, side), grp_rows in grouped.items():
                        per_inst = opt_store.setdefault(exp, {}).setdefault((strike, side), {})
                        _ingest_series_rows(grp_rows, col_map, per_inst, bad_days)
                    break
            greeks_path = base / "iv_greeks_1m.parquet"
            if greeks_path.is_file():
                _ingest_greeks_rows(_read_table_rows(greeks_path), col_map, greeks_store)
            stock_path = base / "stock_1m.parquet"
            if stock_path.is_file():
                _ingest_stock_rows(_read_table_rows(stock_path), col_map, stock_store)
            contracts_path = base / "contracts.csv"
            if not contracts_path.is_file():
                contracts_path = folder / "BANKNIFTY" / "contracts.csv"
            if contracts_path.is_file():
                _ingest_contracts(contracts_path, contracts)
        else:
            base = folder / "BANKNIFTY"
            spot_path = base / "spot_1m.csv"
            if spot_path.is_file():
                _ingest_series_rows(_read_table_rows(spot_path), col_map, spot_store, bad_days)
            for vname in ("vix_1m.csv", "vix_1m.csv.gz"):
                vp = base / vname
                if vp.is_file():
                    _ingest_series_rows(_read_table_rows(vp), col_map, vix_store, bad_days)
                    break
            fut_path = base / "futures_1m.csv"
            if fut_path.is_file():
                _ingest_series_rows(_read_table_rows(fut_path), col_map, fut_store, bad_days)
            opt_dir = base / "options_1m"
            if opt_dir.is_dir():
                for fp in sorted(opt_dir.iterdir()):
                    m = _OPT_CSV_RE.match(fp.name)
                    if not m:
                        continue
                    exp_from_name = m.group("expiry")
                    rows = _read_table_rows(fp)
                    grouped_csv: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
                    for r in rows:
                        exp_raw = _resolve_col(r, "expiry", col_map)
                        exp = exp_from_name if exp_raw is None or str(exp_raw).strip() == "" else str(exp_raw).strip()
                        strike = int(float(_resolve_col(r, "strike", col_map)))
                        side = str(_resolve_col(r, "option_type", col_map)).strip().upper()
                        grouped_csv.setdefault((exp, strike, side), []).append(r)
                    for (exp, strike, side), grp_rows in grouped_csv.items():
                        per_inst = opt_store.setdefault(exp, {}).setdefault((strike, side), {})
                        _ingest_series_rows(grp_rows, col_map, per_inst, bad_days)
            for gname in ("iv_greeks_1m.csv", "iv_greeks_1m.csv.gz"):
                gp = base / gname
                if gp.is_file():
                    _ingest_greeks_rows(_read_table_rows(gp), col_map, greeks_store)
                    break
            for sname in ("stock_1m.csv", "stock_1m.csv.gz"):
                sp = base / sname
                if sp.is_file():
                    _ingest_stock_rows(_read_table_rows(sp), col_map, stock_store)
                    break
            contracts_path = base / "contracts.csv"
            if contracts_path.is_file():
                _ingest_contracts(contracts_path, contracts)

    # All market dates observed in spot_1m
    all_market_dates = sorted(spot_store.keys())
    if start_date is not None:
        all_market_dates = [d for d in all_market_dates if d >= start_date]
    if end_date is not None:
        all_market_dates = [d for d in all_market_dates if d <= end_date]

    # Holdout guard (Niyam 12, T28)
    # Explicit end_date >= holdout_start without allow_holdout refuses + logs.
    effective_auto_trim = auto_trim_holdout and (end_date is None)
    all_market_dates = check_holdout_guard(
        dates=all_market_dates,
        holdout_start=holdout_start,
        allow_holdout=allow_holdout,
        base_dir=base_dir,
        reason="load_dataset",
        auto_trim=effective_auto_trim,
    )

    if not all_market_dates:
        return DatasetResult(
            days=MappingProxyType({}),
            calendar_days=(),
            market_days=(),
            excluded_days=(),
            warmup_days=(),
            trading_days=(),
            holdout_days=(),
        )

    # Build calendar day sequence from first to last market date
    d0 = dt_date.fromisoformat(all_market_dates[0])
    d1 = dt_date.fromisoformat(all_market_dates[-1])
    calendar_days: list[str] = []
    cur = d0
    while cur <= d1:
        calendar_days.append(cur.isoformat())
        cur += timedelta(days=1)

    # First pass: determine candidate valid market dates for DTE calculation
    prelim_valid = [d for d in all_market_dates if d not in bad_days]

    # Second pass: build validated DayData objects
    valid_days: dict[str, DayData] = {}
    excluded_list: list[ExcludedDay] = []
    prev_day_close: float | None = None

    for day_str in all_market_dates:
        if day_str in bad_days:
            reason, detail = bad_days[day_str]
            excluded_list.append(ExcludedDay(date=day_str, reason=reason, detail=detail))
            continue

        spot_min_map = spot_store[day_str]
        # Resolve contract meta (expiry, dte, lot_size)
        expiry, dte, dte_cal, dte_source, lot_sz = _resolve_contract_meta(
            day_str, contracts, lot_size_fallback, prelim_valid, nse_holidays
        )
        if expiry is None:
            excluded_list.append(ExcludedDay(date=day_str, reason="missing_options", detail="no active expiry in contracts.csv"))
            continue
        if lot_sz is None:
            excluded_list.append(ExcludedDay(date=day_str, reason="missing_lot_size", detail=f"expiry {expiry} missing lot_size"))
            continue

        # Check option data exists for active expiry on day_str (§10.2 Rule 5)
        exp_opts = opt_store.get(expiry, {})
        day_opt_insts = {
            k: v[day_str] for k, v in exp_opts.items() if day_str in v and len(v[day_str]) > 0
        }
        if not day_opt_insts:
            excluded_list.append(ExcludedDay(date=day_str, reason="missing_options", detail=f"no option bars for {expiry}"))
            continue

        # Dynamic F&O grid length from actual data (§10.1, T38 — never hardcoded!)
        max_min = max(spot_min_map.keys())
        if day_str in fut_store and fut_store[day_str]:
            max_min = max(max_min, max(fut_store[day_str].keys()))
        for om in day_opt_insts.values():
            max_min = max(max_min, max(om.keys()))
        grid_len = int(max_min) + 1

        # Build index BarSeries & validate OHLC + coverage (§10.2 Rules 2 & 3)
        idx_series, idx_err = _build_bar_series(spot_min_map, grid_len, is_index_or_vix=True)
        if idx_err is not None:
            excluded_list.append(ExcludedDay(date=day_str, reason="invalid_ohlc", detail=f"index: {idx_err}"))
            continue

        live_idx_count = int(np.sum(idx_series.live[: min(grid_len, STANDARD_INDEX_MINUTES)]))
        coverage = live_idx_count / float(STANDARD_INDEX_MINUTES)
        if coverage < coverage_min:
            excluded_list.append(
                ExcludedDay(
                    date=day_str,
                    reason="low_coverage",
                    detail=f"index coverage {coverage:.4f} < {coverage_min}",
                )
            )
            continue

        # Determine ref_spot for cache strike band (prior close, or Day 1 09:15 open — NEVER intraday H/L)
        if prev_day_close is not None:
            ref_spot = prev_day_close
        else:
            live_indices = np.flatnonzero(idx_series.live)
            ref_spot = float(idx_series.open[int(live_indices[0])])

        lo_cache = ref_spot * (1.0 - cache_band_pct / 100.0)
        hi_cache = ref_spot * (1.0 + cache_band_pct / 100.0)

        # Build option BarSeries within cache_band_pct
        opt_series_map: dict[tuple[int, str], BarSeries] = {}
        opt_ohlc_err: str | None = None
        has_greeks = False
        exp_greeks = greeks_store.get(expiry, {})
        for (strike, side), om in sorted(day_opt_insts.items()):
            if strike < lo_cache or strike > hi_cache:
                continue
            g_day = exp_greeks.get((strike, side), {}).get(day_str)
            if g_day is not None:
                has_greeks = True
            oseries, oerr = _build_bar_series(om, grid_len, is_index_or_vix=False, greeks_map=g_day)
            if oerr is not None:
                opt_ohlc_err = f"option ({strike},{side}): {oerr}"
                break
            opt_series_map[(strike, side)] = oseries

        if opt_ohlc_err is not None:
            excluded_list.append(ExcludedDay(date=day_str, reason="invalid_ohlc", detail=opt_ohlc_err))
            continue

        if not opt_series_map:
            excluded_list.append(
                ExcludedDay(date=day_str, reason="missing_options", detail="no strikes within cache_band_pct")
            )
            continue

        # Optional tables: VIX, futures, stocks (never exclude day if missing; §10.2, T41)
        vix_series: BarSeries | None = None
        if day_str in vix_store and vix_store[day_str]:
            vix_series, _ = _build_bar_series(vix_store[day_str], grid_len, is_index_or_vix=True)

        fut_series: BarSeries | None = None
        if day_str in fut_store and fut_store[day_str]:
            fut_series, _ = _build_bar_series(fut_store[day_str], grid_len, is_index_or_vix=False)

        day_stocks: dict[str, BarSeries] = {}
        for sym in sorted(stock_store.keys()):
            if day_str in stock_store[sym] and stock_store[sym][day_str]:
                s_ser, _ = _build_bar_series(stock_store[sym][day_str], grid_len, is_index_or_vix=False)
                day_stocks[sym] = s_ser

        strikes = tuple(sorted({k[0] for k in opt_series_map.keys()}))
        weekday = _WEEKDAYS[dt_date.fromisoformat(day_str).weekday()]

        # Update prev_day_close from last live index bar
        live_idx_pos = np.flatnonzero(idx_series.live)
        prev_day_close = float(idx_series.close[int(live_idx_pos[-1])])

        valid_days[day_str] = DayData(
            date=day_str,
            weekday=weekday,
            grid_len=grid_len,
            expiry=expiry,
            dte=int(dte),
            dte_cal=int(dte_cal),
            dte_source=dte_source,
            lot_size=int(lot_sz),
            is_expiry_day=(expiry == day_str),
            ref_spot=ref_spot,
            index=idx_series,
            vix=vix_series,
            futures=fut_series,
            options=MappingProxyType(opt_series_map),
            stocks=MappingProxyType(day_stocks),
            strikes=strikes,
            has_vix=(vix_series is not None),
            has_futures=(fut_series is not None),
            has_greeks=has_greeks,
            has_stocks=(len(day_stocks) > 0),
        )

    # Taxonomy partitioning (§10.1)
    valid_sorted = sorted(valid_days.keys())
    warmup_list = valid_sorted[:warmup_n]
    post_warmup = valid_sorted[warmup_n:]
    trading_list = [d for d in post_warmup if d < holdout_start]
    holdout_list = [d for d in valid_sorted if d >= holdout_start]

    return DatasetResult(
        days=MappingProxyType(valid_days),
        calendar_days=tuple(calendar_days),
        market_days=tuple(all_market_dates),
        excluded_days=tuple(excluded_list),
        warmup_days=tuple(warmup_list),
        trading_days=tuple(trading_list),
        holdout_days=tuple(holdout_list),
    )


def _ingest_contracts(path: Path, contracts: dict[str, int | None]) -> None:
    """Read contracts.csv and populate expiry -> lot_size map."""
    for r in _read_table_rows(path):
        exp = str(r.get("expiry", "")).strip()
        if not exp:
            continue
        ls_raw = r.get("lot_size")
        if ls_raw is not None and str(ls_raw).strip() != "":
            contracts[exp] = int(float(ls_raw))
        else:
            contracts.setdefault(exp, None)


def _ingest_greeks_rows(
    rows: list[dict[str, Any]],
    col_map: Mapping[str, str],
    greeks_store: dict[str, dict[tuple[int, str], dict[str, dict[int, tuple[float, int]]]]],
) -> None:
    """Ingest iv_greeks_1m rows, masking non-OK iv_status to NaN (§10.1)."""
    for r in rows:
        exp = str(_resolve_col(r, "expiry", col_map)).strip()
        strike = int(float(_resolve_col(r, "strike", col_map)))
        side = str(_resolve_col(r, "option_type", col_map)).strip().upper()
        day_str, m_off = parse_ist_timestamp(_resolve_col(r, "ts_open", col_map))
        iv_status = _resolve_col(r, "iv_status", col_map)
        av_min = ceil_avail_minute(m_off, _resolve_col(r, "available_from_ts", col_map), day_str)
        if iv_status is not None and str(iv_status).strip().upper() != "OK":
            proxy = float("nan")
        else:
            proxy = _to_float(_resolve_col(r, "half_spread_proxy", col_map))
        greeks_store.setdefault(exp, {}).setdefault((strike, side), {}).setdefault(day_str, {})[m_off] = (
            proxy,
            av_min,
        )


def _ingest_stock_rows(
    rows: list[dict[str, Any]],
    col_map: Mapping[str, str],
    stock_store: dict[str, dict[str, dict[int, tuple[Any, ...]]]],
) -> None:
    """Ingest optional constituent stock_1m rows grouped by symbol."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        sym = str(r.get("symbol", "")).strip()
        if sym:
            grouped.setdefault(sym, []).append(r)
    for sym, grp in grouped.items():
        per_sym = stock_store.setdefault(sym, {})
        # Missing or bad stock bars never exclude the whole market day (§10.2)
        local_bad: dict[str, tuple[str, str]] = {}
        _ingest_series_rows(grp, col_map, per_sym, local_bad)
