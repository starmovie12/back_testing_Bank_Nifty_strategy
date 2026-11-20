"""Causal technical indicators and time-gated indicator accessor.

Purpose:
    Computes strictly causal technical indicators (EMA, ATR, RSI, rolling_high,
    rolling_low) where output index i depends ONLY on inputs <= i, handles NaN
    bars without lookahead or forward-fill, caches per-day computations, and
    exposes read-only value() and series() accessors capped strictly at t-1.

PRD Sections:
    §2 (Niyam 1, Niyam 2, Niyam 11), §11.1

Niyam Covered:
    - Niyam 1 : Blind replay (all indicators causal; accessor caps strictly at t-1)
    - Niyam 2 : Bar close / avail_minute visibility respected by IndicatorAccessor
    - Niyam 11: NaN bars stay NaN and are never forward-filled

Guarding Tests:
    T13

PUBLIC API:
    Functions:
        ema, atr, rsi, rolling_high, rolling_low
    Classes:
        IndicatorCache, IndicatorAccessor
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from contracts import LookaheadError
from data_loader import BarSeries


def _ro(arr: np.ndarray) -> np.ndarray:
    """Mark numpy array read-only and return it."""
    arr.flags.writeable = False
    return arr


def _validate_period(period: int, name: str) -> int:
    """Validate indicator period/window is a plain positive int (not bool)."""
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise ValueError(f"{name} period must be int >= 1, got {period!r}")
    return period


# ------------------------------------------------------------------------------
# Causal Indicators (Niyam 1 & Niyam 11)
# ------------------------------------------------------------------------------
def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Causal Exponential Moving Average (alpha = 2 / (period + 1)).

    Element i depends strictly on values[:i+1]. NaN inputs emit NaN without
    corrupting or forward-filling the internal state.
    """
    p = _validate_period(period, "ema")
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return _ro(out)

    alpha = 2.0 / (p + 1.0)
    one_minus = 1.0 - alpha
    prev: float | None = None
    for i in range(n):
        x = float(arr[i])
        if not math.isfinite(x):
            # Niyam 11: NaN bar emits NaN (no forward-fill)
            continue
        if prev is None:
            prev = x
        else:
            prev = alpha * x + one_minus * prev
        out[i] = prev
    return _ro(out)


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Causal Wilder Average True Range over (high, low, close).

    Element i depends strictly on bars <= i. NaN bars emit NaN without
    forward-filling; previous finite close is used for True Range only on
    subsequent finite bars.
    """
    p = _validate_period(period, "atr")
    h_arr = np.asarray(high, dtype=np.float64)
    l_arr = np.asarray(low, dtype=np.float64)
    c_arr = np.asarray(close, dtype=np.float64)
    n = c_arr.shape[0]
    if h_arr.shape[0] != n or l_arr.shape[0] != n:
        raise ValueError("high, low, close arrays must have equal length")

    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return _ro(out)

    prev_close: float | None = None
    tr_seed: list[float] = []
    cur_atr: float | None = None

    for i in range(n):
        hi = float(h_arr[i])
        lo = float(l_arr[i])
        cl = float(c_arr[i])
        if not (math.isfinite(hi) and math.isfinite(lo) and math.isfinite(cl)):
            continue
        if prev_close is None:
            tr = hi - lo
        else:
            tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        prev_close = cl

        if cur_atr is None:
            tr_seed.append(tr)
            if len(tr_seed) == p:
                cur_atr = sum(tr_seed) / float(p)
                out[i] = cur_atr
        else:
            cur_atr = ((cur_atr * (p - 1)) + tr) / float(p)
            out[i] = cur_atr
    return _ro(out)


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Causal Wilder Relative Strength Index in [0.0, 100.0].

    Element i depends strictly on close[:i+1]. NaN inputs emit NaN without
    forward-filling.
    """
    p = _validate_period(period, "rsi")
    c_arr = np.asarray(close, dtype=np.float64)
    n = c_arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return _ro(out)

    prev_close: float | None = None
    gains: list[float] = []
    losses: list[float] = []
    avg_gain: float | None = None
    avg_loss: float | None = None

    for i in range(n):
        cl = float(c_arr[i])
        if not math.isfinite(cl):
            continue
        if prev_close is None:
            prev_close = cl
            continue
        diff = cl - prev_close
        prev_close = cl
        gain = diff if diff > 0.0 else 0.0
        loss = -diff if diff < 0.0 else 0.0

        if avg_gain is None:
            gains.append(gain)
            losses.append(loss)
            if len(gains) == p:
                avg_gain = sum(gains) / float(p)
                avg_loss = sum(losses) / float(p)
                out[i] = _rsi_from_avgs(avg_gain, avg_loss)
        else:
            assert avg_loss is not None
            avg_gain = ((avg_gain * (p - 1)) + gain) / float(p)
            avg_loss = ((avg_loss * (p - 1)) + loss) / float(p)
            out[i] = _rsi_from_avgs(avg_gain, avg_loss)
    return _ro(out)


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    """Convert Wilder average gain/loss to bounded [0, 100] RSI."""
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rolling_high(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling maximum over values[max(0, i-window+1) : i+1].

    NaN bar at i emits NaN (no forward-fill); finite bars take max of finite
    values within the trailing window ending at i.
    """
    w = _validate_period(window, "rolling_high")
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not math.isfinite(float(arr[i])):
            continue
        start = 0 if i + 1 < w else i + 1 - w
        win = arr[start : i + 1]
        finite = win[np.isfinite(win)]
        if finite.size > 0:
            out[i] = float(np.max(finite))
    return _ro(out)


def rolling_low(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling minimum over values[max(0, i-window+1) : i+1].

    NaN bar at i emits NaN (no forward-fill); finite bars take min of finite
    values within the trailing window ending at i.
    """
    w = _validate_period(window, "rolling_low")
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not math.isfinite(float(arr[i])):
            continue
        start = 0 if i + 1 < w else i + 1 - w
        win = arr[start : i + 1]
        finite = win[np.isfinite(win)]
        if finite.size > 0:
            out[i] = float(np.min(finite))
    return _ro(out)


# ------------------------------------------------------------------------------
# Per-Day Cache & Time-Gated Indicator Accessor (§11.1, Niyam 1 & 2)
# ------------------------------------------------------------------------------
class IndicatorCache:
    """Computes and caches causal indicator arrays once per trading day."""

    __slots__ = ("_sources", "_cache")

    def __init__(self, sources: Mapping[str, BarSeries]) -> None:
        self._sources: dict[str, BarSeries] = dict(sources)
        self._cache: dict[tuple[str, str, int], np.ndarray] = {}

    def get_full(self, name: str, period: int, source: str = "index") -> tuple[np.ndarray, BarSeries]:
        """Return cached full-day causal indicator array and its backing BarSeries."""
        if source not in self._sources:
            raise KeyError(f"Unknown indicator source: {source!r}")
        bar_series = self._sources[source]
        key = (source, name.lower(), _validate_period(period, name))
        arr = self._cache.get(key)
        if arr is None:
            lname = key[1]
            if lname == "ema":
                arr = ema(bar_series.close, key[2])
            elif lname == "atr":
                arr = atr(bar_series.high, bar_series.low, bar_series.close, key[2])
            elif lname == "rsi":
                arr = rsi(bar_series.close, key[2])
            elif lname == "rolling_high":
                arr = rolling_high(bar_series.high, key[2])
            elif lname == "rolling_low":
                arr = rolling_low(bar_series.low, key[2])
            else:
                raise ValueError(f"Unsupported indicator: {name!r}")
            self._cache[key] = arr
        return arr, bar_series


class IndicatorAccessor:
    """Time-gated strategy-facing indicator accessor (state.view.ind; §11.1)."""

    __slots__ = ("_cache", "_t")

    def __init__(self, cache: IndicatorCache, t: int) -> None:
        self._cache = cache
        self._t = int(t)

    def series(self, name: str, period: int, *, source: str = "index") -> np.ndarray:
        """Return read-only indicator array strictly capped at [:t] (Niyam 1, 2)."""
        full_arr, bar_series = self._cache.get_full(name, period, source=source)
        n = min(max(self._t, 0), full_arr.shape[0])
        av = bar_series.avail_minute[:n]
        if n == 0 or bool(np.all(av <= self._t)):
            return _ro(full_arr[:n])
        # Mask any bar with avail_minute > t (Niyam 2)
        masked = full_arr[:n].copy()
        masked[av > self._t] = np.nan
        return _ro(masked)

    def value(
        self,
        name: str,
        period: int,
        i: int | None = None,
        *,
        source: str = "index",
    ) -> float:
        """Return plain Python float indicator value at bar i (default t-1; Niyam 1)."""
        idx = (self._t - 1) if i is None else i
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError(f"Bar index i must be int, got {type(idx).__name__}")
        # Niyam 1: strictly forbid i >= t
        if idx >= self._t or idx < 0:
            raise LookaheadError(
                f"LookaheadError: ind.value({name!r}, i={idx}) forbidden at t={self._t}"
            )
        arr = self.series(name, period, source=source)
        if idx >= arr.shape[0]:
            raise LookaheadError(
                f"LookaheadError: bar index {idx} out of range for t={self._t}"
            )
        return float(arr[idx])
