"""Acceptance tests for indicators.py (PRD §13: T13).

Purpose:
    Verifies strict prefix causality, NaN non-forward-fill behavior, read-only
    output arrays, avail_minute masking, and LookaheadError enforcement across
    all indicators (EMA, ATR, RSI, rolling_high, rolling_low) and the
    IndicatorAccessor surface.

PRD Sections:
    §2 (Niyam 1, 2, 11), §11.1, §13, §14 (Step 4)

Guarding Tests (1):
    T13

PUBLIC API:
    test_T13_indicator_causality
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import data_loader as DL
import indicators as IND
from contracts import LookaheadError
from conftest import REPO_ROOT, UpdlDaySpec, build_updl_week, load_test_config, make_base_dir


def test_T13_indicator_causality(tmp_path: Path) -> None:
    """T13: Prefix truncation & future-poison causality on EMA, ATR, RSI, rolling_high/low."""
    rng = np.random.default_rng(42)
    n = 150
    close = 52000.0 + np.cumsum(rng.normal(0.0, 18.0, size=n))
    high = close + rng.uniform(4.0, 22.0, size=n)
    low = close - rng.uniform(4.0, 22.0, size=n)

    # Inject NaN bars (Niyam 11: must remain NaN, never forward-filled)
    nan_positions = (0, 7, 25, 60, 110, 149)
    for pos in nan_positions:
        close[pos] = np.nan
        high[pos] = np.nan
        low[pos] = np.nan

    full_ema = IND.ema(close, 9)
    full_atr = IND.atr(high, low, close, 14)
    full_rsi = IND.rsi(close, 14)
    full_rh = IND.rolling_high(high, 10)
    full_rl = IND.rolling_low(low, 10)

    # 1. Read-only & zero forward-fill on NaN bars
    for arr in (full_ema, full_atr, full_rsi, full_rh, full_rl):
        assert arr.flags.writeable is False
        for pos in nan_positions:
            assert np.isnan(arr[pos]), f"NaN bar {pos} was forward-filled"

    # 2. Prefix truncation: indicator(data[:k]) == indicator(data[:n])[:k]
    for k in (1, 5, 14, 26, 45, 75, 120, 149):
        assert np.array_equal(IND.ema(close[:k], 9), full_ema[:k], equal_nan=True)
        assert np.array_equal(IND.atr(high[:k], low[:k], close[:k], 14), full_atr[:k], equal_nan=True)
        assert np.array_equal(IND.rsi(close[:k], 14), full_rsi[:k], equal_nan=True)
        assert np.array_equal(IND.rolling_high(high[:k], 10), full_rh[:k], equal_nan=True)
        assert np.array_equal(IND.rolling_low(low[:k], 10), full_rl[:k], equal_nan=True)

    # 3. Future-poison test on a loaded updl DayData via IndicatorCache + IndicatorAccessor
    dirs = make_base_dir(tmp_path)
    build_updl_week(
        dirs["data_dir"] / "week_01",
        [UpdlDaySpec(date="2026-11-20", expiry="2026-11-23", lot_size=30)],
        write_csv=True,
    )
    cfg = load_test_config(
        REPO_ROOT,
        overrides={"data_layout": "csv", "warmup_days": 0, "holdout_start": "2026-12-31"},
    )
    day = DL.load_dataset(dirs["data_dir"], cfg, base_dir=dirs["base_dir"]).days["2026-11-20"]

    t = 40
    # Create a future-poisoned copy of day.index where all bars >= t are overwritten
    po = day.index.open.copy()
    ph = day.index.high.copy()
    pl = day.index.low.copy()
    pc = day.index.close.copy()
    po[t:] = 999999.0
    ph[t:] = 999999.0
    pl[t:] = 1.0
    pc[t:] = 999999.0
    av = day.index.avail_minute.copy()
    av[t - 2] = t + 5  # bar t-2 not yet available at decision minute t (Niyam 2)

    poisoned_series = DL.BarSeries(
        open=po,
        high=ph,
        low=pl,
        close=pc,
        volume=day.index.volume.copy(),
        oi=day.index.oi.copy(),
        live=day.index.live.copy(),
        avail_minute=av,
    )

    acc_clean = IND.IndicatorAccessor(IND.IndicatorCache({"index": day.index}), t=t)
    acc_poison = IND.IndicatorAccessor(IND.IndicatorCache({"index": poisoned_series}), t=t)

    for ind_name, period in (("ema", 9), ("atr", 14), ("rsi", 14), ("rolling_high", 10), ("rolling_low", 10)):
        v_clean = acc_clean.value(ind_name, period, i=t - 1)
        v_poison = acc_poison.value(ind_name, period, i=t - 1)
        assert type(v_clean) is float and type(v_poison) is float
        assert math.isclose(v_clean, v_poison, rel_tol=0.0, abs_tol=1e-12)

        s_poison = acc_poison.series(ind_name, period)
        assert s_poison.shape == (t,)
        assert s_poison.flags.writeable is False
        assert np.isnan(s_poison[t - 2])  # Niyam 2 avail_minute masking

    # Niyam 1: accessing bar >= t raises LookaheadError
    for bad_i in (t, t + 1, -1):
        try:
            acc_clean.value("ema", 9, i=bad_i)
            raise AssertionError(f"Expected LookaheadError for i={bad_i}")
        except LookaheadError:
            pass
