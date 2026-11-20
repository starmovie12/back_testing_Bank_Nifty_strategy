"""Core minute-by-minute backtest replay engine, Dual-Track risk shell, and RUIN.

Purpose:
    Executes deterministic, zero-lookahead 1-minute replay of BANKNIFTY index
    option strategies across Fixed 1-Lot (parallelizable, daily reset) and
    Dynamic Compounding (strictly serial, balance carryover, deterministic RUIN)
    tracks. Enforces all 14 Sakht Niyam, exact Decimal sizing and accounting,
    strict exclusive cutoffs, bar lifecycle steps 0/1/2A/2B/2C, and per-day
    fresh strategy namespaces. Performs zero file I/O and zero network access.

PRD Sections:
    §2 (Niyam 1..14), §3.3, §4, §5, §8, §9, §10.4, §11.2, §11.3, §11.4

Niyam Covered:
    All 14 Niyam (Niyam 1 through Niyam 14)

Guarding Tests:
    T01, T02, T03, T04, T05, T06, T09, T11, T18, T19, T22, T23, T24, T26,
    T30, T31, T34, T35, T36, T53, T54, T59, T62

PUBLIC API:
    Dataclasses (frozen=True, slots=True):
        DayResult, TrackRunResult, DualTrackResult
    Functions:
        price_to_decimal, ceil_tick, floor_tick, compute_initial_stop,
        compute_target_price, instantiate_strategy, size_order,
        compute_min_1lot_cost_ref, is_dynamic_ruin, simulate_position,
        run_day, run_track, run_dual_track
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, localcontext
from types import MappingProxyType
from typing import Any, Callable, Mapping

from contracts import (
    Intent,
    InvalidManageAction,
    ManageAction,
    MarketState,
    NoTrade,
    PositionState,
    SkipCode,
    SkippedSignal,
    TodayTradeSummary,
    TradeRecord,
    highest_priority_skip,
    validate_intent,
    validate_manage_action,
    validate_notrade_reason,
)
from costs import (
    ONE,
    ZERO,
    buy_cost_base,
    compute_trade_accounting,
    round_paise,
    to_decimal,
    trade_max_risk_stress,
)
from data_loader import (
    BarSeries,
    DayData,
    TimeGatedView,
    get_prior_days,
    hhmm_to_minute,
    minute_to_hhmm,
)
from indicators import IndicatorAccessor, IndicatorCache

_DECIMAL_PREC: int = 28


# ------------------------------------------------------------------------------
# Exact Decimal Price & Tick Rounding Helpers (§11.3, L05)
# ------------------------------------------------------------------------------
def price_to_decimal(val: float) -> Decimal:
    """Convert a finite bar price float/numpy scalar to exact Decimal without float repr leakage."""
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"Non-finite market price: {val!r}")
    return Decimal(f"{f:.6f}").normalize()


def ceil_tick(val: Decimal, tick_size: Decimal) -> Decimal:
    """Round Decimal UP to the nearest tick_size multiple (conservative initial stop/target)."""
    v = to_decimal(val)
    ts = to_decimal(tick_size)
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        ticks = (v / ts).to_integral_value(rounding=ROUND_CEILING)
        return round_paise(ticks * ts)


def floor_tick(val: Decimal, tick_size: Decimal) -> Decimal:
    """Round Decimal DOWN to the nearest tick_size multiple (conservative TRAIL stop)."""
    v = to_decimal(val)
    ts = to_decimal(tick_size)
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        ticks = (v / ts).to_integral_value(rounding=ROUND_FLOOR)
        return round_paise(ticks * ts)


def compute_initial_stop(
    price: Decimal,
    intent: Intent,
    tick_size: Decimal,
) -> Decimal:
    """Compute ceil_tick(raw_stop) for an Intent at price (§11.3)."""
    p = to_decimal(price)
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        if intent.stop_frac is not None:
            frac = Decimal(f"{float(intent.stop_frac):.10f}").normalize()
            raw = p * (ONE - frac)
        else:
            assert intent.stop_points is not None
            pts = Decimal(f"{float(intent.stop_points):.10f}").normalize()
            raw = p - pts
    return ceil_tick(raw, tick_size)


def compute_target_price(
    fill_price: Decimal,
    initial_stop: Decimal,
    target_r: float | None,
    tick_size: Decimal,
) -> Decimal | None:
    """Compute ceil_tick(fill_price + target_r * R) when target_r is specified (§11.3)."""
    if target_r is None:
        return None
    r_val = fill_price - initial_stop
    tr = Decimal(f"{float(target_r):.10f}").normalize()
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        raw_target = fill_price + (tr * r_val)
    return ceil_tick(raw_target, tick_size)


# ------------------------------------------------------------------------------
# Per-Day Fresh Strategy Namespace Loader (Niyam 14, T34)
# ------------------------------------------------------------------------------
class _ModuleStrategyAdapter:
    """Wraps module-level decide(state) and optional manage(state, pos) functions."""

    __slots__ = ("_decide", "_manage", "PARAMS")

    def __init__(
        self,
        decide_fn: Callable[[MarketState], Any],
        manage_fn: Callable[[MarketState, PositionState], Any] | None,
        params: Mapping[str, Any],
    ) -> None:
        self._decide = decide_fn
        self._manage = manage_fn
        self.PARAMS = params

    def decide(self, state: MarketState) -> Any:
        return self._decide(state)

    def manage(self, state: MarketState, pos: PositionState) -> Any:
        if self._manage is None:
            return ManageAction(action="HOLD")
        return self._manage(state, pos)


def instantiate_strategy(
    strategy_source: str | Callable[[], Any],
    module_name: str = "strategy_day",
) -> Any:
    """Instantiate a strategy in a fresh per-day namespace with read-only PARAMS (Niyam 14)."""
    if callable(strategy_source):
        return strategy_source()

    # Niyam 14: fresh isolated namespace per trading day
    ns: dict[str, Any] = {"__name__": module_name}
    code_obj = compile(strategy_source, f"<{module_name}>", "exec")
    exec(code_obj, ns)  # noqa: S102

    raw_params = ns.get("PARAMS", {})
    if isinstance(raw_params, Mapping):
        frozen_params: Mapping[str, Any] = MappingProxyType(dict(raw_params))
        ns["PARAMS"] = frozen_params
    else:
        frozen_params = MappingProxyType({})

    if "Strategy" in ns and isinstance(ns["Strategy"], type):
        cls = ns["Strategy"]
        try:
            inst = cls(frozen_params)
        except TypeError:
            inst = cls()
        return inst

    if "decide" in ns and callable(ns["decide"]):
        manage_fn = ns.get("manage") if callable(ns.get("manage")) else None
        return _ModuleStrategyAdapter(ns["decide"], manage_fn, frozen_params)

    raise ValueError("Strategy source must define either class Strategy or function decide(state)")


# ------------------------------------------------------------------------------
# Optional Config Value Parsers
# ------------------------------------------------------------------------------
def _opt_decimal(val: Any) -> Decimal | None:
    if val is None or val == "":
        return None
    return to_decimal(val)


def _opt_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        raise ValueError(f"Config integer cannot be bool: {val!r}")
    return int(val)


# ------------------------------------------------------------------------------
# Exact Decimal Sizing Function (§5.1, §5.2, §5.4, §8, Niyam 13)
# ------------------------------------------------------------------------------
def size_order(
    *,
    track: str,
    price: Decimal,
    stop: Decimal,
    lot_size: int,
    minute_offset: int,
    is_expiry_day: bool,
    trade_date: str,
    balance_now: Decimal,
    day_open_balance: Decimal,
    realised_loss_today: Decimal,
    intent_lots: int | None,
    config: Mapping[str, Any],
    k_upper: int | None = None,
) -> tuple[int, SkipCode | None]:
    """Compute largest integer lots k passing Shart 1/2/3 (+ 100% 1-lot exception).

    Returns (k, None) when k >= 1, or (0, SkipCode) using §8 priority order.
    """
    cap_use_pct = to_decimal(config["capital_use_pct"])
    cap_exc_pct = to_decimal(config["capital_exception_pct"])
    day_cap_pct = to_decimal(config["day_cap_pct"])
    per_trade_pct = _opt_decimal(config.get("per_trade_max_risk_pct"))
    cfg_max_lots = _opt_int(config.get("max_lots"))

    if balance_now <= ZERO:
        return 0, SkipCode.SKIP_NO_BALANCE

    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        cap_limit = cap_use_pct * balance_now
        exc_limit = cap_exc_pct * balance_now
        daily_cap_limit = day_cap_pct * day_open_balance
        per_trade_limit = (
            (per_trade_pct * day_open_balance) if per_trade_pct is not None else None
        )
        # Finite upper bound for downward search (§5.1 step 3)
        raw_upper = int(cap_limit // (price * Decimal(lot_size)))

    k_hi = max(1, raw_upper)
    if track == "fixed":
        k_hi = min(k_hi, 1)
    if cfg_max_lots is not None:
        k_hi = min(k_hi, cfg_max_lots)
    if intent_lots is not None:
        k_hi = min(k_hi, intent_lots)
    if k_upper is not None:
        # Niyam 13: k_fill <= k_signal (may decrease, NEVER increase)
        k_hi = min(k_hi, k_upper)

    if k_hi < 1:
        return 0, SkipCode.SKIP_SIZE

    def _passes(k_val: int, use_exception: bool) -> bool:
        bc = buy_cost_base(k_val, price, lot_size, minute_offset, is_expiry_day, trade_date, config)
        limit = exc_limit if (use_exception and k_val == 1) else cap_limit
        if bc > limit:
            return False
        tmr = trade_max_risk_stress(
            k_val, price, stop, lot_size, minute_offset, is_expiry_day, trade_date, config
        )
        if (realised_loss_today + tmr) > daily_cap_limit:
            return False
        if per_trade_limit is not None and tmr > per_trade_limit:
            return False
        return True

    # Downward search for largest k passing Shart 1, Shart 2, Shart 3
    for k_cand in range(k_hi, 0, -1):
        if _passes(k_cand, use_exception=False):
            return k_cand, None

    # Small-Balance 1-Lot Exception (§5.1 step 4, V5.2 Rule 2.4)
    if _passes(1, use_exception=True):
        return 1, None

    # Determine exact §8 highest-priority skip code at k = 1
    bc1 = buy_cost_base(1, price, lot_size, minute_offset, is_expiry_day, trade_date, config)
    tmr1 = trade_max_risk_stress(
        1, price, stop, lot_size, minute_offset, is_expiry_day, trade_date, config
    )
    failed: list[SkipCode] = []
    if bc1 > exc_limit:
        failed.append(SkipCode.SKIP_NO_BALANCE)
    elif bc1 > cap_limit:
        failed.append(SkipCode.SKIP_SIZE)
    if (realised_loss_today + tmr1) > daily_cap_limit:
        failed.append(SkipCode.SKIP_DAYCAP)
    if per_trade_limit is not None and tmr1 > per_trade_limit:
        failed.append(SkipCode.SKIP_RISK)
    if not failed:
        failed.append(SkipCode.SKIP_SIZE)
    return 0, highest_priority_skip(failed)


# ------------------------------------------------------------------------------
# Deterministic RUIN Helpers (§5.5, T59)
# ------------------------------------------------------------------------------
def compute_min_1lot_cost_ref(
    day: DayData,
    prior_days: tuple[DayData, ...],
    t: int,
    config: Mapping[str, Any],
) -> Decimal | None:
    """Compute zero-lookahead minimum 1-lot buy_cost reference for Dynamic RUIN (§5.5).

    At day open (t == 0): uses prior_days[-1] live closes within ±chain_band_pct.
    After intraday exit (t > 0): uses current day live closes <= t-1 within ±chain_band_pct.
    Returns None if no live bar exists in the lookahead-safe window.
    """
    band_pct = float(config["chain_band_pct"])
    start_min = hhmm_to_minute(str(config["entry_window_start"]))

    if t <= 0:
        if not prior_days:
            return None
        ref_day = prior_days[-1]
        # Last live index close on prior_days[-1]
        spot = ref_day.ref_spot
        for j in range(len(ref_day.index) - 1, -1, -1):
            if bool(ref_day.index.live[j]) and math.isfinite(float(ref_day.index.close[j])):
                spot = float(ref_day.index.close[j])
                break
        lo_k = spot * (1.0 - band_pct / 100.0)
        hi_k = spot * (1.0 + band_pct / 100.0)
        min_cost: Decimal | None = None
        for (strike, _side), ser in ref_day.options.items():
            if strike < lo_k or strike > hi_k:
                continue
            for j in range(len(ser) - 1, -1, -1):
                if bool(ser.live[j]) and math.isfinite(float(ser.close[j])):
                    p = price_to_decimal(float(ser.close[j]))
                    c = buy_cost_base(1, p, day.lot_size, start_min, day.is_expiry_day, day.date, config)
                    if min_cost is None or c < min_cost:
                        min_cost = c
                    break
        return min_cost

    # Intraday post-exit: scan current day bars strictly <= t-1 (Niyam 1 & 2)
    spot = day.ref_spot
    idx_vis = day.index.slice_visible(t)
    for j in range(len(idx_vis) - 1, -1, -1):
        if bool(idx_vis.live[j]) and math.isfinite(float(idx_vis.close[j])):
            spot = float(idx_vis.close[j])
            break
    lo_k = spot * (1.0 - band_pct / 100.0)
    hi_k = spot * (1.0 + band_pct / 100.0)
    min_cost = None
    for (strike, _side), ser in day.options.items():
        if strike < lo_k or strike > hi_k:
            continue
        for j in range(min(t - 1, len(ser) - 1), -1, -1):
            if bool(ser.live[j]) and int(ser.avail_minute[j]) <= t and math.isfinite(float(ser.close[j])):
                p = price_to_decimal(float(ser.close[j]))
                c = buy_cost_base(1, p, day.lot_size, t, day.is_expiry_day, day.date, config)
                if min_cost is None or c < min_cost:
                    min_cost = c
                break
    return min_cost


def is_dynamic_ruin(balance_now: Decimal, min_1lot_cost_ref: Decimal | None) -> bool:
    """Return True iff Dynamic track triggers §5.5 deterministic RUIN."""
    if balance_now <= ZERO:
        return True
    if min_1lot_cost_ref is not None and balance_now < min_1lot_cost_ref:
        return True
    return False


# ------------------------------------------------------------------------------
# Position Simulation — Bar Lifecycle Steps 0, 1, 2A, 2B, 2C (§11.4)
# ------------------------------------------------------------------------------
def _stop_exit_reason(stop_source: str) -> str:
    if stop_source == "breakeven":
        return "breakeven"
    if stop_source == "trail":
        return "trail"
    return "stop"


@dataclass(frozen=True, slots=True)
class _ExitResult:
    exit_bar: int
    exit_price: Decimal
    exit_reason: str
    gap_exit: bool
    is_no_bar: bool
    highest_high: Decimal
    lowest_low: Decimal
    mfe_min: int
    mae_min: int


def simulate_position(
    *,
    day: DayData,
    prior_days: tuple[DayData, ...],
    opt_series: BarSeries,
    intent: Intent,
    entry_bar: int,
    fill_price: Decimal,
    initial_stop: Decimal,
    target_price: Decimal | None,
    lots: int,
    strategy: Any,
    ind_cache: IndicatorCache,
    today_trades: tuple[TodayTradeSummary, ...],
    balance_now: Decimal,
    day_open_balance: Decimal,
    realised_loss_today: Decimal,
    entries_today: int,
    track: str,
    config: Mapping[str, Any],
) -> _ExitResult:
    """Execute §11.4 bar lifecycle (Steps 0, 1, 2A, 2B, 2C) from entry_bar until exit."""
    tick_size = to_decimal(config["tick_size"])
    step_min = int(config["decision_step_min"])
    start_min = hhmm_to_minute(str(config["entry_window_start"]))
    ff_key = "expiry_force_flat_time" if day.is_expiry_day else "force_flat_time"
    force_flat_min = hhmm_to_minute(str(config[ff_key]))
    exit_wait_bars = int(config["exit_wait_bars"])
    chain_band = float(config["chain_band_pct"])
    cache_band = float(config["cache_band_pct"])

    r_unit = fill_price - initial_stop
    be_trigger = (
        Decimal(f"{float(intent.breakeven_trigger_r):.10f}").normalize()
        if intent.breakeven_trigger_r > 0.0
        else ZERO
    )

    current_stop = initial_stop
    stop_source = "initial"
    highest_high = fill_price
    lowest_low = fill_price
    mfe_min = 0
    mae_min = 0
    last_live_low = fill_price
    last_live_close = fill_price
    exit_wait_count = 0

    pending_force_flat = False
    pending_strategy_exit = False
    pending_time = False
    pending_scratch = False

    qty = lots * day.lot_size
    last_bar = day.grid_len - 1

    for b in range(entry_bar, day.grid_len):
        hold_m = b - entry_bar

        # ------------------------------------------------------------------
        # Step 0: Clock & Manage (runs on EVERY bar b, live OR stale/NaN)
        # ------------------------------------------------------------------
        if b >= force_flat_min:
            pending_force_flat = True
        if intent.time_stop_min > 0 and hold_m >= intent.time_stop_min:
            pending_time = True
        if intent.scratch_min > 0 and hold_m == intent.scratch_min:
            if last_live_close <= fill_price:
                pending_scratch = True

        if (
            intent.exit_mode == "manage"
            and b > entry_bar
            and ((b - start_min) % step_min == 0)
            and b < force_flat_min
        ):
            view_b = TimeGatedView(
                day,
                t=b,
                chain_band_pct=chain_band,
                cache_band_pct=cache_band,
                ind=IndicatorAccessor(ind_cache, t=b),
            )
            state_b = MarketState(
                date=day.date,
                weekday=day.weekday,
                t=b,
                ts=minute_to_hhmm(b),
                expiry=day.expiry,
                dte=day.dte,
                dte_cal=day.dte_cal,
                lot_size=day.lot_size,
                is_expiry_day=day.is_expiry_day,
                view=view_b,
                prior_days=prior_days,
                today_trades=today_trades,
                balance_now=balance_now,
                day_open_balance=day_open_balance,
                realised_loss_today=realised_loss_today,
                entries_today=entries_today,
                track=track,
            )
            pos_b = PositionState(
                side=intent.side,
                strike=intent.strike,
                expiry=day.expiry,
                dte=day.dte,
                lots=lots,
                lot_size=day.lot_size,
                qty=qty,
                entry_bar=entry_bar,
                entry_ts=minute_to_hhmm(entry_bar),
                entry_price=fill_price,
                initial_stop=initial_stop,
                current_stop=current_stop,
                stop_source=stop_source,
                target_price=target_price,
                exit_mode=intent.exit_mode,
                hold_min=hold_m,
                highest_high=highest_high,
                lowest_low=lowest_low,
                reason=intent.reason,
                tags=intent.tags,
            )
            action = validate_manage_action(strategy.manage(state_b, pos_b))
            if action.action == "EXIT":
                pending_strategy_exit = True
            elif action.action == "TRAIL":
                assert action.new_stop is not None
                ns_raw = (
                    action.new_stop
                    if isinstance(action.new_stop, Decimal)
                    else price_to_decimal(float(action.new_stop))
                )
                floored = floor_tick(ns_raw, tick_size)
                if floored <= current_stop:
                    raise InvalidManageAction(
                        f"TRAIL floor_tick({ns_raw})={floored} must be > current_stop={current_stop}"
                    )
                current_stop = floored
                stop_source = "trail"

        # Pending exit priority: force_flat > strategy_exit > time > scratch
        top_pending: str | None = None
        if pending_force_flat:
            top_pending = "force_flat"
        elif pending_strategy_exit:
            top_pending = "strategy_exit"
        elif pending_time:
            top_pending = "time"
        elif pending_scratch:
            top_pending = "scratch"

        is_live_b = bool(opt_series.live[b])

        # ------------------------------------------------------------------
        # Step 1: Bar b NOT LIVE (stale/NaN)
        # ------------------------------------------------------------------
        if not is_live_b:
            if top_pending is not None:
                exit_wait_count += 1
                if exit_wait_count > exit_wait_bars or b == last_bar:
                    nb_price = min(last_live_low, last_live_close)
                    return _ExitResult(
                        exit_bar=b,
                        exit_price=nb_price,
                        exit_reason="no_bar",
                        gap_exit=False,
                        is_no_bar=True,
                        highest_high=highest_high,
                        lowest_low=min(lowest_low, nb_price),
                        mfe_min=mfe_min,
                        mae_min=hold_m if nb_price < lowest_low else mae_min,
                    )
            continue

        # ------------------------------------------------------------------
        # Step 2: Bar b LIVE
        # ------------------------------------------------------------------
        bo = price_to_decimal(float(opt_series.open[b]))
        bh = price_to_decimal(float(opt_series.high[b]))
        bl = price_to_decimal(float(opt_series.low[b]))
        bc = price_to_decimal(float(opt_series.close[b]))

        # 2A (Open): Gap-open stop or pending exit fills at open (skip 2B & 2C)
        if b > entry_bar and bo <= current_stop:
            if bo > highest_high:
                highest_high = bo
                mfe_min = hold_m
            if bo < lowest_low:
                lowest_low = bo
                mae_min = hold_m
            return _ExitResult(
                exit_bar=b,
                exit_price=bo,
                exit_reason=_stop_exit_reason(stop_source),
                gap_exit=True,
                is_no_bar=False,
                highest_high=highest_high,
                lowest_low=lowest_low,
                mfe_min=mfe_min,
                mae_min=mae_min,
            )

        if top_pending is not None:
            if bo > highest_high:
                highest_high = bo
                mfe_min = hold_m
            if bo < lowest_low:
                lowest_low = bo
                mae_min = hold_m
            return _ExitResult(
                exit_bar=b,
                exit_price=bo,
                exit_reason=top_pending,
                gap_exit=False,
                is_no_bar=False,
                highest_high=highest_high,
                lowest_low=lowest_low,
                mfe_min=mfe_min,
                mae_min=mae_min,
            )

        # 2B (Intrabar — runs on entry bar e too; Niyam 7 & Niyam 8)
        # Update MAE/MFE up to intrabar exit
        if bh > highest_high:
            highest_high = bh
            mfe_min = hold_m
        if bl < lowest_low:
            lowest_low = bl
            mae_min = hold_m

        # Niyam 7: stop-first if both stop and target touch in same bar
        if bl <= current_stop:
            return _ExitResult(
                exit_bar=b,
                exit_price=current_stop,
                exit_reason=_stop_exit_reason(stop_source),
                gap_exit=False,
                is_no_bar=False,
                highest_high=highest_high,
                lowest_low=lowest_low,
                mfe_min=mfe_min,
                mae_min=mae_min,
            )

        # Niyam 7: gap open above target fills at target_price (no gap bonus)
        if target_price is not None and bh >= target_price:
            return _ExitResult(
                exit_bar=b,
                exit_price=target_price,
                exit_reason="target",
                gap_exit=False,
                is_no_bar=False,
                highest_high=highest_high,
                lowest_low=lowest_low,
                mfe_min=mfe_min,
                mae_min=mae_min,
            )

        # 2C (Post-Bar): Update live trackers & breakeven arming for next bar
        last_live_low = bl
        last_live_close = bc

        if be_trigger > ZERO:
            with localcontext() as ctx:
                ctx.prec = _DECIMAL_PREC
                be_level = fill_price + (be_trigger * r_unit)
            if bh >= be_level and fill_price > current_stop:
                current_stop = fill_price
                stop_source = "breakeven"

    # Defensive guarantee (Niyam 9): zero overnight open positions
    nb_price = min(last_live_low, last_live_close)
    return _ExitResult(
        exit_bar=last_bar,
        exit_price=nb_price,
        exit_reason="no_bar",
        gap_exit=False,
        is_no_bar=True,
        highest_high=highest_high,
        lowest_low=min(lowest_low, nb_price),
        mfe_min=mfe_min,
        mae_min=mae_min,
    )


# ------------------------------------------------------------------------------
# Single-Day & Multi-Day Dual-Track Replay (§4, §5, §11.2)
# ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DayResult:
    """Output of replaying one trading day on one track."""

    date: str
    track: str
    day_open_balance: Decimal
    day_close_balance_base: Decimal
    day_close_balance_stress: Decimal
    trades: tuple[TradeRecord, ...]
    skipped: tuple[SkippedSignal, ...]
    notrade_counts: Mapping[str, int]
    decide_calls: int
    data_gap_skips: int
    cap_breach: bool
    ruin_triggered: bool
    ruin_balance: Decimal | None


@dataclass(frozen=True, slots=True)
class TrackRunResult:
    """Aggregated result of running a strategy across trading days on one track."""

    track: str
    start_balance: Decimal
    final_balance_base: Decimal
    final_balance_stress: Decimal
    trades: tuple[TradeRecord, ...]
    skipped: tuple[SkippedSignal, ...]
    day_results: tuple[DayResult, ...]
    notrade_counts: Mapping[str, int]
    decide_calls: int
    data_gap_skips: int
    cap_breach_days: tuple[str, ...]
    ruin_flag: bool
    ruin_date: str | None
    ruin_balance: Decimal | None


@dataclass(frozen=True, slots=True)
class DualTrackResult:
    """Combined independent results for [FIXED 1-LOT] and [DYNAMIC COMPOUNDING]."""

    fixed: TrackRunResult
    dynamic: TrackRunResult


def _last_live_opt_close(ser: BarSeries, t: int) -> Decimal | None:
    """Return last live option close strictly <= t-1 with avail_minute <= t (Niyam 1, 2)."""
    upper = min(t - 1, len(ser) - 1)
    for j in range(upper, -1, -1):
        if bool(ser.live[j]) and int(ser.avail_minute[j]) <= t and math.isfinite(float(ser.close[j])):
            return price_to_decimal(float(ser.close[j]))
    return None


def _build_indicator_cache(day: DayData) -> IndicatorCache:
    sources: dict[str, BarSeries] = {"index": day.index}
    if day.vix is not None:
        sources["vix"] = day.vix
    if day.futures is not None:
        sources["futures"] = day.futures
    return IndicatorCache(sources)


def run_day(
    *,
    day: DayData,
    prior_days: tuple[DayData, ...],
    strategy_source: str | Callable[[], Any],
    config: Mapping[str, Any],
    track: str,
    day_open_balance: Decimal,
    day_open_balance_stress: Decimal | None = None,
    trade_no_offset: int = 0,
    delay_d: int = 0,
    mode: str = "fast",
    seen_notrade_codes: set[str] | None = None,
) -> DayResult:
    """Replay one trading day on one track ('fixed' or 'dynamic').

    Pure function (no file I/O, no shared mutation) safe for fork parallelism
    on the Fixed 1-Lot track. Both mode='reference' and mode='fast' produce
    paise-identical results (T09).
    """
    if track not in ("fixed", "dynamic"):
        raise ValueError(f"track must be 'fixed' or 'dynamic', got {track!r}")
    if mode not in ("fast", "reference"):
        raise ValueError(f"mode must be 'fast' or 'reference', got {mode!r}")

    base_cap = to_decimal(config["base_capital"])
    open_bal = base_cap if track == "fixed" else to_decimal(day_open_balance)
    open_bal_stress = (
        base_cap
        if track == "fixed"
        else to_decimal(open_bal if day_open_balance_stress is None else day_open_balance_stress)
    )

    balance_now = open_bal
    balance_stress = open_bal_stress
    realised_loss_today = ZERO

    # Niyam 14: fresh strategy namespace per day
    strategy = instantiate_strategy(strategy_source, module_name=f"strat_{day.date.replace('-', '_')}")

    # Dynamic Track Day-Open RUIN check (§5.5)
    if track == "dynamic":
        min_ref = compute_min_1lot_cost_ref(day, prior_days, t=0, config=config)
        if is_dynamic_ruin(balance_now, min_ref):
            return DayResult(
                date=day.date,
                track=track,
                day_open_balance=open_bal,
                day_close_balance_base=balance_now,
                day_close_balance_stress=balance_stress,
                trades=(),
                skipped=(),
                notrade_counts=MappingProxyType({}),
                decide_calls=0,
                data_gap_skips=0,
                cap_breach=False,
                ruin_triggered=True,
                ruin_balance=balance_now,
            )

    start_min = hhmm_to_minute(str(config["entry_window_start"]))
    cutoff_key = "expiry_last_entry_time" if day.is_expiry_day else "last_entry_time"
    cutoff_min = hhmm_to_minute(str(config[cutoff_key]))
    step_min = int(config["decision_step_min"])
    fill_wait_bars = int(config["fill_wait_bars"])
    fill_buf = to_decimal(config["fill_buf"])
    tick_size = to_decimal(config["tick_size"])
    cooloff_min = int(config["cooloff_min"])
    max_entries = _opt_int(config.get("max_entries_per_day"))
    liq_warn_lots = int(config["liquidity_warn_lots"])
    day_cap_pct = to_decimal(config["day_cap_pct"])
    chain_band = float(config["chain_band_pct"])
    cache_band = float(config["cache_band_pct"])

    ind_cache = _build_indicator_cache(day)
    seen_codes = seen_notrade_codes if seen_notrade_codes is not None else set()
    notrade_counts: dict[str, int] = {}
    trades_out: list[TradeRecord] = []
    skipped_out: list[SkippedSignal] = []
    today_summaries: list[TodayTradeSummary] = []

    decide_calls = 0
    data_gap_skips = 0
    entries_today = 0
    next_allowed_min = start_min
    ruin_triggered = False
    ruin_balance: Decimal | None = None
    trade_seq = int(trade_no_offset)

    # Strict exclusive cutoff loop: start_min <= t < cutoff_min (Niyam 9, §11.2)
    t = start_min
    while t < cutoff_min and t < day.grid_len:
        if ruin_triggered:
            break
        if t < next_allowed_min:
            t += step_min
            continue
        if max_entries is not None and entries_today >= max_entries:
            break

        # Niyam 11: do not call decide() when index bar t-1 is NaN or not yet visible
        if t <= 0 or not bool(day.index.live[t - 1]) or int(day.index.avail_minute[t - 1]) > t:
            data_gap_skips += 1
            t += step_min
            continue

        # Reference mode rebuilds cache per decision to guarantee zero cross-minute cache artifacts
        step_cache = _build_indicator_cache(day) if mode == "reference" else ind_cache
        view_t = TimeGatedView(
            day,
            t=t,
            chain_band_pct=chain_band,
            cache_band_pct=cache_band,
            ind=IndicatorAccessor(step_cache, t=t),
        )
        state_t = MarketState(
            date=day.date,
            weekday=day.weekday,
            t=t,
            ts=minute_to_hhmm(t),
            expiry=day.expiry,
            dte=day.dte,
            dte_cal=day.dte_cal,
            lot_size=day.lot_size,
            is_expiry_day=day.is_expiry_day,
            view=view_t,
            prior_days=prior_days,
            today_trades=tuple(today_summaries),
            balance_now=balance_now,
            day_open_balance=open_bal,
            realised_loss_today=realised_loss_today,
            entries_today=entries_today,
            track=track,
        )

        decide_calls += 1
        decision = strategy.decide(state_t)

        if isinstance(decision, NoTrade):
            code = validate_notrade_reason(decision.reason, seen_codes)
            notrade_counts[code] = notrade_counts.get(code, 0) + 1
            t += step_min
            continue

        # Validate Intent (§5.4, §8 Priority 1 -> SKIP_INVALID_INTENT)
        ok_intent, detail = validate_intent(decision)
        if not ok_intent:
            skipped_out.append(
                SkippedSignal(
                    signal_ts=minute_to_hhmm(t),
                    side=str(getattr(decision, "side", "")),
                    strike=int(getattr(decision, "strike", 0)) if isinstance(getattr(decision, "strike", 0), int) and not isinstance(getattr(decision, "strike", 0), bool) else 0,
                    est_price=None,
                    est_stop=None,
                    skip_code=SkipCode.SKIP_INVALID_INTENT,
                    detail=detail,
                )
            )
            t += step_min
            continue

        assert isinstance(decision, Intent)
        opt_key = (decision.strike, decision.side)
        opt_ser = day.options.get(opt_key)

        # Niyam 13: Zero-lookahead signal-time pre-sizing from bar t-1 close
        last_close = _last_live_opt_close(opt_ser, t) if opt_ser is not None else None
        if last_close is None or opt_ser is None:
            skipped_out.append(
                SkippedSignal(
                    signal_ts=minute_to_hhmm(t),
                    side=decision.side,
                    strike=decision.strike,
                    est_price=None,
                    est_stop=None,
                    skip_code=SkipCode.SKIP_NO_PRICE,
                    detail="no live option close <= t-1",
                )
            )
            t += step_min
            continue

        with localcontext() as ctx:
            ctx.prec = _DECIMAL_PREC
            est_price = round_paise(last_close * (ONE + fill_buf))
        est_stop = compute_initial_stop(est_price, decision, tick_size)
        if est_stop <= ZERO or est_stop >= est_price:
            skipped_out.append(
                SkippedSignal(
                    signal_ts=minute_to_hhmm(t),
                    side=decision.side,
                    strike=decision.strike,
                    est_price=est_price,
                    est_stop=est_stop,
                    skip_code=SkipCode.SKIP_BAD_STOP,
                    detail=f"est_stop={est_stop} invalid for est_price={est_price}",
                )
            )
            t += step_min
            continue

        k_signal, sig_skip = size_order(
            track=track,
            price=est_price,
            stop=est_stop,
            lot_size=day.lot_size,
            minute_offset=t,
            is_expiry_day=day.is_expiry_day,
            trade_date=day.date,
            balance_now=balance_now,
            day_open_balance=open_bal,
            realised_loss_today=realised_loss_today,
            intent_lots=decision.lots,
            config=config,
        )
        if k_signal == 0:
            assert sig_skip is not None
            skipped_out.append(
                SkippedSignal(
                    signal_ts=minute_to_hhmm(t),
                    side=decision.side,
                    strike=decision.strike,
                    est_price=est_price,
                    est_stop=est_stop,
                    skip_code=sig_skip,
                    detail="signal-time sizing skip",
                )
            )
            t += step_min
            continue

        # Fill wait loop (Niyam 3, 4, 9): bar t + delay_d .. t + delay_d + fill_wait_bars
        entry_bar: int | None = None
        fill_skip: SkipCode | None = None
        base_fill_bar = t + int(delay_d)
        for offset in range(fill_wait_bars + 1):
            cand = base_fill_bar + offset
            # Niyam 9 / §11.2: fill bar strictly < cutoff_min
            if cand >= cutoff_min or cand >= day.grid_len:
                fill_skip = SkipCode.SKIP_CUTOFF
                break
            if bool(opt_ser.live[cand]):
                entry_bar = cand
                break
        if entry_bar is None:
            skipped_out.append(
                SkippedSignal(
                    signal_ts=minute_to_hhmm(t),
                    side=decision.side,
                    strike=decision.strike,
                    est_price=est_price,
                    est_stop=est_stop,
                    skip_code=fill_skip if fill_skip is not None else SkipCode.SKIP_STALE_CANCEL,
                    detail="fill wait exceeded or hit cutoff",
                )
            )
            # Advance t past the wait window (single pending order at a time, Niyam 10)
            t = max(t + step_min, base_fill_bar + fill_wait_bars + 1)
            continue

        # Fill-time re-verification on actual open price (Niyam 13: k_fill <= k_signal)
        fill_price = price_to_decimal(float(opt_ser.open[entry_bar]))
        initial_stop = compute_initial_stop(fill_price, decision, tick_size)
        if initial_stop <= ZERO or initial_stop >= fill_price:
            skipped_out.append(
                SkippedSignal(
                    signal_ts=minute_to_hhmm(t),
                    side=decision.side,
                    strike=decision.strike,
                    est_price=fill_price,
                    est_stop=initial_stop,
                    skip_code=SkipCode.SKIP_BAD_STOP,
                    detail=f"fill initial_stop={initial_stop} invalid for fill_price={fill_price}",
                )
            )
            t = max(t + step_min, entry_bar + 1)
            continue

        k_fill, fill_size_skip = size_order(
            track=track,
            price=fill_price,
            stop=initial_stop,
            lot_size=day.lot_size,
            minute_offset=entry_bar,
            is_expiry_day=day.is_expiry_day,
            trade_date=day.date,
            balance_now=balance_now,
            day_open_balance=open_bal,
            realised_loss_today=realised_loss_today,
            intent_lots=decision.lots,
            config=config,
            k_upper=k_signal,
        )
        if k_fill == 0:
            assert fill_size_skip is not None
            skipped_out.append(
                SkippedSignal(
                    signal_ts=minute_to_hhmm(t),
                    side=decision.side,
                    strike=decision.strike,
                    est_price=fill_price,
                    est_stop=initial_stop,
                    skip_code=fill_size_skip,
                    detail="fill-time sizing skip",
                )
            )
            t = max(t + step_min, entry_bar + 1)
            continue

        target_price = compute_target_price(fill_price, initial_stop, decision.target_r, tick_size)
        entries_today += 1

        # Simulate position lifecycle (§11.4)
        exit_res = simulate_position(
            day=day,
            prior_days=prior_days,
            opt_series=opt_ser,
            intent=decision,
            entry_bar=entry_bar,
            fill_price=fill_price,
            initial_stop=initial_stop,
            target_price=target_price,
            lots=k_fill,
            strategy=strategy,
            ind_cache=step_cache,
            today_trades=tuple(today_summaries),
            balance_now=balance_now,
            day_open_balance=open_bal,
            realised_loss_today=realised_loss_today,
            entries_today=entries_today,
            track=track,
            config=config,
        )

        # Exact Decimal accounting across BASE, STRESS, PROXY (§10.3, §10.4)
        qty = k_fill * day.lot_size
        px_e = (
            opt_ser.half_spread_proxy[entry_bar]
            if opt_ser.half_spread_proxy is not None
            else None
        )
        px_x = (
            opt_ser.half_spread_proxy[exit_res.exit_bar]
            if opt_ser.half_spread_proxy is not None
            else None
        )
        bd_base = compute_trade_accounting(
            fill_price, exit_res.exit_price, qty, entry_bar, exit_res.exit_bar,
            day.is_expiry_day, day.date, config, scenario="BASE", is_no_bar=exit_res.is_no_bar,
        )
        bd_stress = compute_trade_accounting(
            fill_price, exit_res.exit_price, qty, entry_bar, exit_res.exit_bar,
            day.is_expiry_day, day.date, config, scenario="STRESS", is_no_bar=exit_res.is_no_bar,
        )
        bd_proxy = compute_trade_accounting(
            fill_price, exit_res.exit_price, qty, entry_bar, exit_res.exit_bar,
            day.is_expiry_day, day.date, config, scenario="PROXY", is_no_bar=exit_res.is_no_bar,
            proxy_entry_hs=px_e, proxy_exit_hs=px_x,
        )
        assert bd_base is not None and bd_stress is not None

        # Update intraday balance & profit buffer (§5.3)
        balance_now = round_paise(balance_now + bd_base.net)
        balance_stress = round_paise(balance_stress + bd_stress.net)
        realised_loss_today = max(ZERO, round_paise(open_bal - balance_now))

        outcome = "WIN" if bd_base.net > ZERO else "LOSS"
        trade_seq += 1
        with localcontext() as ctx:
            ctx.prec = _DECIMAL_PREC
            stop_pct = ((fill_price - initial_stop) / fill_price).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
        mfe_val = max(ZERO, round_paise(exit_res.highest_high - fill_price))
        mae_val = max(ZERO, round_paise(fill_price - exit_res.lowest_low))
        hold_min = exit_res.exit_bar - entry_bar

        tr = TradeRecord(
            track=track,
            trade_no=trade_seq,
            date=day.date,
            weekday=day.weekday,
            signal_ts=minute_to_hhmm(t),
            entry_ts=minute_to_hhmm(entry_bar),
            side=decision.side,
            strike=decision.strike,
            expiry=day.expiry,
            dte=day.dte,
            lots_signal=k_signal,
            lots=k_fill,
            lot_size=day.lot_size,
            qty=qty,
            liquidity_warn=(k_fill >= liq_warn_lots),
            entry_price=round_paise(fill_price),
            initial_stop=round_paise(initial_stop),
            stop_pct=stop_pct,
            target_price=round_paise(target_price) if target_price is not None else None,
            exit_ts=minute_to_hhmm(exit_res.exit_bar),
            exit_price=round_paise(exit_res.exit_price),
            exit_reason=exit_res.exit_reason,
            gap_exit=exit_res.gap_exit,
            hold_min=hold_min,
            gross=bd_base.gross,
            charges_base=bd_base.charges,
            fill_cost_base=bd_base.fill_cost,
            net_base=bd_base.net,
            charges_stress=bd_stress.charges,
            fill_cost_stress=bd_stress.fill_cost,
            net_stress=bd_stress.net,
            net_proxy=bd_proxy.net if bd_proxy is not None else None,
            balance_base=balance_now,
            balance_stress=balance_stress,
            outcome=outcome,
            mae=mae_val,
            mfe=mfe_val,
            mfe_min=exit_res.mfe_min,
            mae_min=exit_res.mae_min,
            reason=decision.reason,
            tags="+".join(decision.tags),
        )
        trades_out.append(tr)
        today_summaries.append(
            TodayTradeSummary(
                trade_no=trade_seq,
                side=decision.side,
                strike=decision.strike,
                lots=k_fill,
                entry_ts=tr.entry_ts,
                exit_ts=tr.exit_ts,
                entry_price=tr.entry_price,
                exit_price=tr.exit_price,
                exit_reason=tr.exit_reason,
                net_base=tr.net_base,
                outcome=outcome,
            )
        )

        # Post-exit RUIN check on Dynamic track (§5.5)
        next_t = exit_res.exit_bar + 1
        if track == "dynamic":
            min_ref = compute_min_1lot_cost_ref(day, prior_days, t=next_t, config=config)
            if is_dynamic_ruin(balance_now, min_ref):
                ruin_triggered = True
                ruin_balance = balance_now
                break

        # Advance clock past exit bar (and cooloff_min on loss if configured)
        if cooloff_min > 0 and outcome == "LOSS":
            next_allowed_min = exit_res.exit_bar + cooloff_min
        else:
            next_allowed_min = next_t

        # Align next decision minute to decision_step_min grid starting from start_min
        if next_t <= t:
            t += step_min
        else:
            rem = (next_t - start_min) % step_min
            t = next_t if rem == 0 else next_t + (step_min - rem)

    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        cap_breach = realised_loss_today > (day_cap_pct * open_bal)

    return DayResult(
        date=day.date,
        track=track,
        day_open_balance=open_bal,
        day_close_balance_base=balance_now,
        day_close_balance_stress=balance_stress,
        trades=tuple(trades_out),
        skipped=tuple(skipped_out),
        notrade_counts=MappingProxyType(notrade_counts),
        decide_calls=decide_calls,
        data_gap_skips=data_gap_skips,
        cap_breach=cap_breach,
        ruin_triggered=ruin_triggered,
        ruin_balance=ruin_balance,
    )


def run_track(
    *,
    days: Mapping[str, DayData],
    trading_dates: list[str] | tuple[str, ...],
    strategy_source: str | Callable[[], Any],
    config: Mapping[str, Any],
    track: str,
    initial_balance: Decimal | None = None,
    delay_d: int = 0,
    mode: str = "fast",
) -> TrackRunResult:
    """Run a strategy across trading_dates on one track ('fixed' or 'dynamic').

    Dynamic track strictly refuses non-chronological date order (T53) and carries
    closing balance across days; Fixed track resets to base_capital every day.
    """
    if track not in ("fixed", "dynamic"):
        raise ValueError(f"track must be 'fixed' or 'dynamic', got {track!r}")

    dates = list(trading_dates)
    if track == "dynamic" and dates != sorted(dates):
        raise ValueError("Dynamic Compounding track requires strictly chronological trading_dates (T53)")

    base_cap = to_decimal(config["base_capital"])
    start_bal = base_cap if (track == "fixed" or initial_balance is None) else to_decimal(initial_balance)
    cur_bal = start_bal
    cur_bal_stress = start_bal
    warmup_n = int(config["warmup_days"])
    holdout_start = str(config["holdout_start"])

    all_trades: list[TradeRecord] = []
    all_skipped: list[SkippedSignal] = []
    day_results: list[DayResult] = []
    agg_notrade: dict[str, int] = {}
    cap_breach_days: list[str] = []
    seen_codes: set[str] = set()

    total_decide = 0
    total_gaps = 0
    trade_no = 0
    ruin_flag = False
    ruin_date: str | None = None
    ruin_balance: Decimal | None = None

    for d_str in dates:
        if track == "dynamic" and ruin_flag:
            # Dynamic track remains halted for the remainder of the run (§5.5)
            break
        day = days[d_str]
        prior = get_prior_days(days, d_str, warmup_days=warmup_n, holdout_start=holdout_start)
        d_res = run_day(
            day=day,
            prior_days=prior,
            strategy_source=strategy_source,
            config=config,
            track=track,
            day_open_balance=(base_cap if track == "fixed" else cur_bal),
            day_open_balance_stress=(base_cap if track == "fixed" else cur_bal_stress),
            trade_no_offset=trade_no,
            delay_d=delay_d,
            mode=mode,
            seen_notrade_codes=seen_codes,
        )
        day_results.append(d_res)
        all_trades.extend(d_res.trades)
        all_skipped.extend(d_res.skipped)
        trade_no += len(d_res.trades)
        total_decide += d_res.decide_calls
        total_gaps += d_res.data_gap_skips
        for k, cnt in d_res.notrade_counts.items():
            agg_notrade[k] = agg_notrade.get(k, 0) + cnt
        if d_res.cap_breach:
            cap_breach_days.append(d_str)

        if track == "dynamic":
            cur_bal = d_res.day_close_balance_base
            cur_bal_stress = d_res.day_close_balance_stress
            if d_res.ruin_triggered:
                ruin_flag = True
                ruin_date = d_str
                ruin_balance = d_res.ruin_balance

    if track == "fixed":
        # Fixed track final balance is informational cumulative sum over base_capital
        final_base = round_paise(start_bal + sum((tr.net_base for tr in all_trades), ZERO))
        final_stress = round_paise(start_bal + sum((tr.net_stress for tr in all_trades), ZERO))
    else:
        final_base = cur_bal
        final_stress = cur_bal_stress

    return TrackRunResult(
        track=track,
        start_balance=start_bal,
        final_balance_base=final_base,
        final_balance_stress=final_stress,
        trades=tuple(all_trades),
        skipped=tuple(all_skipped),
        day_results=tuple(day_results),
        notrade_counts=MappingProxyType(agg_notrade),
        decide_calls=total_decide,
        data_gap_skips=total_gaps,
        cap_breach_days=tuple(cap_breach_days),
        ruin_flag=ruin_flag,
        ruin_date=ruin_date,
        ruin_balance=ruin_balance,
    )


def run_dual_track(
    *,
    days: Mapping[str, DayData],
    trading_dates: list[str] | tuple[str, ...],
    strategy_source: str | Callable[[], Any],
    config: Mapping[str, Any],
    dynamic_initial_balance: Decimal | None = None,
    delay_d: int = 0,
    mode: str = "fast",
) -> DualTrackResult:
    """Execute BOTH [FIXED 1-LOT] and [DYNAMIC COMPOUNDING] independently (§4)."""
    fixed_res = run_track(
        days=days,
        trading_dates=trading_dates,
        strategy_source=strategy_source,
        config=config,
        track="fixed",
        delay_d=delay_d,
        mode=mode,
    )
    dynamic_res = run_track(
        days=days,
        trading_dates=trading_dates,
        strategy_source=strategy_source,
        config=config,
        track="dynamic",
        initial_balance=dynamic_initial_balance,
        delay_d=delay_d,
        mode=mode,
    )
    return DualTrackResult(fixed=fixed_res, dynamic=dynamic_res)
