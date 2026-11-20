"""Exact Decimal per-order Upstox charges, spread/slippage, and accounting.

Purpose:
    Implements all monetary cost calculations in exact Python Decimal using
    an explicit localcontext (prec=28, ROUND_HALF_UP to 0.01 once per order):
    dated Upstox option charges (STT date switch), MAX-rule time-window
    spread multipliers, BASE/STRESS/PROXY scenarios, no_bar exit multiplier,
    zero-double-counting trade accounting with ₹0.00 sell floor, sizing
    buy_cost (BASE) and trade_max_risk (STRESS) helpers, and Live-Bot Parity
    freeze-quantity child-order splitting.

PRD Sections:
    §2 (Niyam 5, 6), §5.1, §5.2, §9, §10.3, §10.4

Niyam Covered:
    - Niyam 5: Adverse spread & slippage tracked in fill_cost without double-counting
    - Niyam 6: Full per-order Decimal charges rounded once per order (ROUND_HALF_UP)

Guarding Tests:
    T03, T07, T18, T19, T31, T44, T54, T57

PUBLIC API:
    Constants:
        PAISE, ZERO, ONE, OPEN_WINDOW_END_MIN
    Dataclasses (frozen=True, slots=True):
        ChargeSchedule, TradeCostBreakdown
    Functions:
        to_decimal, round_paise, get_charge_schedule, order_charges,
        window_spread_mult, entry_slip_unit, exit_slip_unit,
        buy_cost_base, trade_max_risk_stress, compute_trade_accounting,
        split_freeze_qty, split_order_charges
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Any, Mapping

from data_loader import hhmm_to_minute

# ------------------------------------------------------------------------------
# Structural Constants (no tunable magic numbers; L12)
# ------------------------------------------------------------------------------
PAISE: Decimal = Decimal("0.01")
ZERO: Decimal = Decimal("0.00")
ONE: Decimal = Decimal("1")
_DECIMAL_PREC: int = 28
# Open spread window is 09:15..09:24 IST inclusive (offsets 0..9 from 09:15; §10.4)
OPEN_WINDOW_END_MIN: int = hhmm_to_minute("09:24")


# ------------------------------------------------------------------------------
# Exact Decimal Conversion & Rounding (L05)
# ------------------------------------------------------------------------------
def to_decimal(val: Any) -> Decimal:
    """Convert Decimal, exact str, or plain int to finite Decimal.

    Rejects bool and float to prevent IEEE-754 rounding contamination (L05).
    """
    if isinstance(val, bool):
        raise TypeError("bool is forbidden in Decimal money math")
    if isinstance(val, Decimal):
        if not val.is_finite():
            raise ValueError(f"Non-finite Decimal: {val}")
        return val
    if type(val) is int:
        return Decimal(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValueError("Empty string cannot be converted to Decimal")
        d = Decimal(s)
        if not d.is_finite():
            raise ValueError(f"Non-finite Decimal string: {val!r}")
        return d
    # Handle numpy integer scalars by converting to plain Python int first (L05)
    if hasattr(val, "item") and not isinstance(val, (bytes, bytearray)):
        py_val = val.item()
        if type(py_val) is int:
            return Decimal(py_val)
    raise TypeError(
        f"Float/unsupported type {type(val).__name__} forbidden in Decimal money math"
    )


def round_paise(val: Decimal) -> Decimal:
    """Quantize a Decimal to ₹0.01 using ROUND_HALF_UP inside an explicit context."""
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        ctx.rounding = ROUND_HALF_UP
        return val.quantize(PAISE, rounding=ROUND_HALF_UP)


# ------------------------------------------------------------------------------
# Dated Upstox Per-Order Charges (§10.3, Niyam 6)
# ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ChargeSchedule:
    """Validated Decimal charge rates active for a given trading date."""

    effective_from: str
    brokerage_flat: Decimal
    brokerage_in_gst_base: bool
    stt_sell_rate: Decimal
    stt_in_gst_base: bool
    exchange_rate: Decimal
    exchange_in_gst_base: bool
    ipft_rate: Decimal
    ipft_in_gst_base: bool
    sebi_rate: Decimal
    sebi_in_gst_base: bool
    stamp_buy_rate: Decimal
    stamp_in_gst_base: bool
    gst_rate: Decimal


def get_charge_schedule(config: Mapping[str, Any], trade_date: str) -> ChargeSchedule:
    """Resolve the active dated Upstox charge schedule for trade_date (YYYY-MM-DD)."""
    rows = config.get("charges")
    if not rows:
        raise ValueError("config['charges'] table is missing or empty")
    active: Mapping[str, Any] | None = None
    for row in sorted(rows, key=lambda r: str(r["effective_from"])):
        if str(row["effective_from"]) <= trade_date:
            active = row
    if active is None:
        raise ValueError(f"No charge schedule effective on or before {trade_date}")
    return ChargeSchedule(
        effective_from=str(active["effective_from"]),
        brokerage_flat=to_decimal(active["brokerage_flat"]),
        brokerage_in_gst_base=bool(active["brokerage_in_gst_base"]),
        stt_sell_rate=to_decimal(active["stt_sell_rate"]),
        stt_in_gst_base=bool(active["stt_in_gst_base"]),
        exchange_rate=to_decimal(active["exchange_rate"]),
        exchange_in_gst_base=bool(active["exchange_in_gst_base"]),
        ipft_rate=to_decimal(active["ipft_rate"]),
        ipft_in_gst_base=bool(active["ipft_in_gst_base"]),
        sebi_rate=to_decimal(active["sebi_rate"]),
        sebi_in_gst_base=bool(active["sebi_in_gst_base"]),
        stamp_buy_rate=to_decimal(active["stamp_buy_rate"]),
        stamp_in_gst_base=bool(active["stamp_in_gst_base"]),
        gst_rate=to_decimal(active["gst_rate"]),
    )


def order_charges(
    side: str,
    value: Decimal,
    config: Mapping[str, Any],
    trade_date: str,
    *,
    charges_mult: Decimal = ONE,
) -> Decimal:
    """Compute per-order Upstox charges rounded ONCE to ₹0.01 (ROUND_HALF_UP; Niyam 6)."""
    side_u = side.upper()
    if side_u not in ("BUY", "SELL"):
        raise ValueError(f"Order side must be 'BUY' or 'SELL', got {side!r}")
    val = to_decimal(value)
    if val < ZERO:
        raise ValueError(f"Order value must be >= 0, got {val}")
    cmult = to_decimal(charges_mult)
    sched = get_charge_schedule(config, trade_date)

    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        ctx.rounding = ROUND_HALF_UP

        brok = sched.brokerage_flat
        stt = (val * sched.stt_sell_rate) if side_u == "SELL" else ZERO
        exch = val * sched.exchange_rate
        ipft = val * sched.ipft_rate
        sebi = val * sched.sebi_rate
        stamp = (val * sched.stamp_buy_rate) if side_u == "BUY" else ZERO

        gst_base = ZERO
        if sched.brokerage_in_gst_base:
            gst_base += brok
        if sched.stt_in_gst_base:
            gst_base += stt
        if sched.exchange_in_gst_base:
            gst_base += exch
        if sched.ipft_in_gst_base:
            gst_base += ipft
        if sched.sebi_in_gst_base:
            gst_base += sebi
        if sched.stamp_in_gst_base:
            gst_base += stamp

        gst = gst_base * sched.gst_rate
        total = (brok + stt + exch + ipft + sebi + stamp + gst) * cmult
        return total.quantize(PAISE, rounding=ROUND_HALF_UP)


# ------------------------------------------------------------------------------
# Spread, Slippage & Window Multipliers (§10.4, Niyam 5)
# ------------------------------------------------------------------------------
def window_spread_mult(
    minute_offset: int,
    is_expiry_day: bool,
    config: Mapping[str, Any],
) -> Decimal:
    """Return MAX of applicable time-window spread multipliers (never added or multiplied)."""
    m_normal = to_decimal(config["spread_mult_normal"])
    m_expiry = to_decimal(config["spread_mult_expiry"])
    m_open_close = to_decimal(config["spread_mult_open_close"])
    close_win = int(config["close_window_minutes"])

    ff_key = "expiry_force_flat_time" if is_expiry_day else "force_flat_time"
    ff_min = hhmm_to_minute(str(config[ff_key]))
    close_win_start = ff_min - close_win

    applicable = [m_normal]
    if is_expiry_day:
        applicable.append(m_expiry)
    if minute_offset <= OPEN_WINDOW_END_MIN or minute_offset >= close_win_start:
        applicable.append(m_open_close)
    return max(applicable)


def _proxy_hs_to_decimal(proxy_half_spread: Any) -> Decimal | None:
    """Convert optional float/Decimal half_spread_PROXY to Decimal, or None if unavailable."""
    if proxy_half_spread is None:
        return None
    if isinstance(proxy_half_spread, Decimal):
        return proxy_half_spread if proxy_half_spread.is_finite() and proxy_half_spread >= ZERO else None
    if isinstance(proxy_half_spread, bool):
        return None
    if isinstance(proxy_half_spread, (int, float)):
        if not math.isfinite(float(proxy_half_spread)) or float(proxy_half_spread) < 0.0:
            return None
        return Decimal(f"{float(proxy_half_spread):.6f}").normalize()
    if hasattr(proxy_half_spread, "item"):
        return _proxy_hs_to_decimal(proxy_half_spread.item())
    return None


def entry_slip_unit(
    minute_offset: int,
    is_expiry_day: bool,
    config: Mapping[str, Any],
    *,
    scenario: str = "BASE",
    proxy_half_spread: Any = None,
    spread_scale: Decimal | None = None,
) -> Decimal | None:
    """Compute per-unit entry adverse slip (half_spread*mult + slippage) in exact Decimal."""
    scen = scenario.upper()
    hs = to_decimal(config["half_spread"])
    sl = to_decimal(config["slippage"])
    mult = window_spread_mult(minute_offset, is_expiry_day, config)
    sc = ONE if spread_scale is None else to_decimal(spread_scale)

    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        if scen == "BASE":
            return (hs * mult * sc) + sl
        if scen == "STRESS":
            ssm = to_decimal(config["stress_spread_mult"])
            slm = to_decimal(config["stress_slippage_mult"])
            return (hs * mult * ssm * sc) + (sl * slm)
        if scen == "PROXY":
            phs = _proxy_hs_to_decimal(proxy_half_spread)
            if phs is None:
                return None
            return (phs * sc) + sl
        raise ValueError(f"Unknown cost scenario: {scenario!r}")


def exit_slip_unit(
    minute_offset: int,
    is_expiry_day: bool,
    config: Mapping[str, Any],
    *,
    scenario: str = "BASE",
    is_no_bar: bool = False,
    proxy_half_spread: Any = None,
    spread_scale: Decimal | None = None,
) -> Decimal | None:
    """Compute per-unit exit adverse slip, applying no_bar_cost_mult when is_no_bar=True."""
    base_slip = entry_slip_unit(
        minute_offset,
        is_expiry_day,
        config,
        scenario=scenario,
        proxy_half_spread=proxy_half_spread,
        spread_scale=spread_scale,
    )
    if base_slip is None:
        return None
    if not is_no_bar:
        return base_slip
    nb_mult = to_decimal(config["no_bar_cost_mult"])
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        return base_slip * nb_mult


# ------------------------------------------------------------------------------
# Sizing Helpers (§5.1, §5.2)
# ------------------------------------------------------------------------------
def buy_cost_base(
    k: int,
    price: Decimal,
    lot_size: int,
    minute_offset: int,
    is_expiry_day: bool,
    trade_date: str,
    config: Mapping[str, Any],
) -> Decimal:
    """Compute §5.1 buy_cost(k, p) using BASE entry slippage and per-order BUY charges."""
    if isinstance(k, bool) or type(k) is not int or k < 1:
        raise ValueError(f"k must be int >= 1, got {k!r}")
    p = to_decimal(price)
    qty = Decimal(k) * Decimal(int(lot_size))
    slip_b = entry_slip_unit(minute_offset, is_expiry_day, config, scenario="BASE")
    assert slip_b is not None

    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        ctx.rounding = ROUND_HALF_UP
        entry_fc = slip_b * qty
        buy_val = (p + slip_b) * qty
        chg = order_charges("BUY", buy_val, config, trade_date)
        return round_paise((p * qty) + chg + entry_fc)


def trade_max_risk_stress(
    k: int,
    price: Decimal,
    stop: Decimal,
    lot_size: int,
    minute_offset: int,
    is_expiry_day: bool,
    trade_date: str,
    config: Mapping[str, Any],
) -> Decimal:
    """Compute §5.1 trade_max_risk(k, p, s) using STRESS round-trip cost and cap_basis."""
    if isinstance(k, bool) or type(k) is not int or k < 1:
        raise ValueError(f"k must be int >= 1, got {k!r}")
    p = to_decimal(price)
    s = to_decimal(stop)
    qty = Decimal(k) * Decimal(int(lot_size))
    slip_s = entry_slip_unit(minute_offset, is_expiry_day, config, scenario="STRESS")
    assert slip_s is not None
    chg_mult = to_decimal(config["stress_charges_mult"])
    cap_basis = str(config.get("cap_basis", "stop"))

    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        ctx.rounding = ROUND_HALF_UP
        s_buy_val = (p + slip_s) * qty
        buy_chg = order_charges("BUY", s_buy_val, config, trade_date, charges_mult=chg_mult)

        if cap_basis == "premium":
            return round_paise((p * qty) + buy_chg + (slip_s * qty))
        if cap_basis != "stop":
            raise ValueError(f"Invalid cap_basis: {cap_basis!r}")

        s_exit_slip = min(s, slip_s)
        s_sell_val = (s - s_exit_slip) * qty
        sell_chg = order_charges("SELL", s_sell_val, config, trade_date, charges_mult=chg_mult)
        stress_fc = (slip_s + s_exit_slip) * qty
        return round_paise(((p - s) * qty) + buy_chg + sell_chg + stress_fc)


# ------------------------------------------------------------------------------
# Zero Double-Counting Trade Accounting (§10.3, Niyam 5 & 6)
# ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TradeCostBreakdown:
    """Exact Decimal accounting breakdown for one completed trade under one scenario."""

    gross: Decimal
    charges: Decimal
    fill_cost: Decimal
    net: Decimal
    buy_value: Decimal
    sell_value: Decimal


def compute_trade_accounting(
    entry_price: Decimal,
    exit_price: Decimal,
    qty: int,
    entry_min: int,
    exit_min: int,
    is_expiry_day: bool,
    trade_date: str,
    config: Mapping[str, Any],
    *,
    scenario: str = "BASE",
    is_no_bar: bool = False,
    proxy_entry_hs: Any = None,
    proxy_exit_hs: Any = None,
    spread_scale: Decimal | None = None,
) -> TradeCostBreakdown | None:
    """Compute gross, charges, fill_cost, and net with sell floor ₹0.00 (§10.3).

    Returns None when scenario == 'PROXY' and proxy half-spread is unavailable.
    """
    ep = to_decimal(entry_price)
    xp = to_decimal(exit_price)
    q = Decimal(int(qty))

    e_slip = entry_slip_unit(
        entry_min,
        is_expiry_day,
        config,
        scenario=scenario,
        proxy_half_spread=proxy_entry_hs,
        spread_scale=spread_scale,
    )
    x_slip_raw = exit_slip_unit(
        exit_min,
        is_expiry_day,
        config,
        scenario=scenario,
        is_no_bar=is_no_bar,
        proxy_half_spread=proxy_exit_hs,
        spread_scale=spread_scale,
    )
    if e_slip is None or x_slip_raw is None:
        return None

    scen = scenario.upper()
    chg_mult = to_decimal(config["stress_charges_mult"]) if scen == "STRESS" else ONE

    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PREC
        ctx.rounding = ROUND_HALF_UP

        # Sell floor ₹0.00 (§10.3)
        x_slip = min(xp, x_slip_raw)
        gross = round_paise((xp - ep) * q)
        fill_cost = round_paise((e_slip + x_slip) * q)
        buy_value = (ep + e_slip) * q
        sell_value = (xp - x_slip) * q
        buy_chg = order_charges("BUY", buy_value, config, trade_date, charges_mult=chg_mult)
        sell_chg = order_charges("SELL", sell_value, config, trade_date, charges_mult=chg_mult)
        charges = round_paise(buy_chg + sell_chg)
        net = round_paise(gross - charges - fill_cost)

    return TradeCostBreakdown(
        gross=gross,
        charges=charges,
        fill_cost=fill_cost,
        net=net,
        buy_value=round_paise(buy_value),
        sell_value=round_paise(sell_value),
    )


# ------------------------------------------------------------------------------
# Live-Bot Parity: NSE Freeze-Qty Child-Order Split (§9, T57)
# ------------------------------------------------------------------------------
def split_freeze_qty(
    qty: int,
    lot_size: int,
    nse_freeze_qty: int | None,
) -> tuple[int, ...]:
    """Split qty into ceil(qty / nse_freeze_qty) child orders, each <= freeze_qty & multiple of lot_size."""
    if isinstance(qty, bool) or type(qty) is not int or qty <= 0:
        raise ValueError(f"qty must be positive int, got {qty!r}")
    if isinstance(lot_size, bool) or type(lot_size) is not int or lot_size <= 0:
        raise ValueError(f"lot_size must be positive int, got {lot_size!r}")
    if qty % lot_size != 0:
        raise ValueError(f"qty ({qty}) must be an exact multiple of lot_size ({lot_size})")

    if nse_freeze_qty is None:
        return (qty,)
    if isinstance(nse_freeze_qty, bool) or type(nse_freeze_qty) is not int or nse_freeze_qty <= 0:
        raise ValueError(f"nse_freeze_qty must be positive int or None, got {nse_freeze_qty!r}")
    if nse_freeze_qty % lot_size != 0:
        raise ValueError(
            f"nse_freeze_qty ({nse_freeze_qty}) must be an exact multiple of lot_size ({lot_size})"
        )
    if qty <= nse_freeze_qty:
        return (qty,)

    children: list[int] = []
    remaining = qty
    while remaining > 0:
        chunk = min(remaining, nse_freeze_qty)
        children.append(chunk)
        remaining -= chunk
    return tuple(children)


def split_order_charges(
    side: str,
    unit_price: Decimal,
    qty: int,
    lot_size: int,
    nse_freeze_qty: int | None,
    config: Mapping[str, Any],
    trade_date: str,
) -> tuple[tuple[int, ...], Decimal]:
    """Split order across freeze-qty children and sum per-child Upstox charges (§9, T57)."""
    p = to_decimal(unit_price)
    children = split_freeze_qty(qty, lot_size, nse_freeze_qty)
    total = ZERO
    for child_qty in children:
        total += order_charges(side, p * Decimal(child_qty), config, trade_date)
    return children, round_paise(total)
