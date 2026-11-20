"""Canonical contracts, immutable dataclasses, skip codes, and validators.

Purpose:
    Defines the locked data types, enums, exceptions, and validation helpers
    shared across data_loader, costs, indicators, engine, report, diagnose,
    and run. Strategies may import Intent, NoTrade, ManageAction, and
    read-only state types from this module.

PRD Sections:
    §2 (Niyam 13, Niyam 14), §3.3, §5.4, §8, §10.2, §11.1, §11.3

Niyam Covered:
    - Niyam 13: Pre-sizing Intent.lots contract (int | None, strict bool rejection)
    - Niyam 14: Immutable frozen/slotted contracts preventing state mutation

Guarding Tests:
    T12, T21, T26, T30, T33, T34, T49

PUBLIC API:
    Constants:
        ENGINE_TAG, TRADE_RECORD_COLUMNS, VALID_SIDES, VALID_EXIT_MODES,
        VALID_MANAGE_ACTIONS, VALID_EXIT_REASONS, VALID_TRACKS, VALID_OUTCOMES,
        NOTRADE_REASON_MAX_LEN, NOTRADE_MAX_UNIQUE_CODES, SKIP_CODE_PRIORITY,
        SHADOW_SKIP_CODES
    Exceptions:
        EngineError, LookaheadError, HoldoutLocked, InvalidManageAction,
        InvalidNoTradeReason, DataMissingError, ConfigError
    Enums:
        SkipCode
    Protocols:
        MarketView
    Dataclasses (frozen=True, slots=True):
        Intent, NoTrade, ManageAction, PositionState, TodayTradeSummary,
        MarketState, SkippedSignal, TradeRecord
    Functions:
        is_valid_int_strict, is_finite_real, highest_priority_skip,
        validate_intent, validate_notrade_reason, validate_manage_action,
        format_decimal
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Iterable, Protocol, runtime_checkable

# ------------------------------------------------------------------------------
# Locked Structural Constants (§3.3, §8, §11.1)
# ------------------------------------------------------------------------------
ENGINE_TAG: str = "engine-v2.0"

TRADE_RECORD_COLUMNS: tuple[str, ...] = (
    # Identity & Track (10)
    "track",
    "trade_no",
    "date",
    "weekday",
    "signal_ts",
    "entry_ts",
    "side",
    "strike",
    "expiry",
    "dte",
    # Sizing & Liquidity (5)
    "lots_signal",
    "lots",
    "lot_size",
    "qty",
    "liquidity_warn",
    # Levels & Exit (9)
    "entry_price",
    "initial_stop",
    "stop_pct",
    "target_price",
    "exit_ts",
    "exit_price",
    "exit_reason",
    "gap_exit",
    "hold_min",
    # Accounting (8)
    "gross",
    "charges_base",
    "fill_cost_base",
    "net_base",
    "charges_stress",
    "fill_cost_stress",
    "net_stress",
    "net_proxy",
    # Account & Diagnostics (9)
    "balance_base",
    "balance_stress",
    "outcome",
    "mae",
    "mfe",
    "mfe_min",
    "mae_min",
    "reason",
    "tags",
)

VALID_SIDES: frozenset[str] = frozenset({"CE", "PE"})
VALID_EXIT_MODES: frozenset[str] = frozenset({"preset", "manage"})
VALID_MANAGE_ACTIONS: frozenset[str] = frozenset({"HOLD", "EXIT", "TRAIL"})
VALID_EXIT_REASONS: frozenset[str] = frozenset(
    {
        "stop",
        "target",
        "breakeven",
        "trail",
        "scratch",
        "time",
        "force_flat",
        "strategy_exit",
        "no_bar",
    }
)
VALID_TRACKS: frozenset[str] = frozenset({"fixed", "dynamic"})
VALID_OUTCOMES: frozenset[str] = frozenset({"WIN", "LOSS"})

NOTRADE_REASON_MAX_LEN: int = 24
NOTRADE_MAX_UNIQUE_CODES: int = 30
_NOTRADE_REASON_RE: re.Pattern[str] = re.compile(r"^[a-z0-9_]{1,24}$")


# ------------------------------------------------------------------------------
# Project Exceptions (§10.1, §10.2, §11.3)
# ------------------------------------------------------------------------------
class EngineError(Exception):
    """Base exception for all banknifty-backtest engine errors."""


class LookaheadError(EngineError):
    """Raised on any attempt to access bar index i >= t (Niyam 1)."""


class HoldoutLocked(EngineError):
    """Raised on unauthorized access to data >= holdout_start (Niyam 12)."""


class InvalidManageAction(EngineError):
    """Raised when manage() returns an invalid or downward/NaN TRAIL action."""


class InvalidNoTradeReason(EngineError):
    """Raised when NoTrade.reason violates charset/length or unique-code cap."""


class DataMissingError(EngineError):
    """Raised when a strategy explicitly requests an absent optional table."""


class ConfigError(EngineError):
    """Raised when config.toml fails schema, type, or legacy-key validation."""


# ------------------------------------------------------------------------------
# Canonical Skip Code Registry (§8, N_SKIP_CODES = 9)
# ------------------------------------------------------------------------------
class SkipCode(Enum):
    """Exactly 9 canonical skip/cancel codes in §8 funnel priority order."""

    SKIP_INVALID_INTENT = "SKIP_INVALID_INTENT"
    SKIP_CUTOFF = "SKIP_CUTOFF"
    SKIP_NO_PRICE = "SKIP_NO_PRICE"
    SKIP_BAD_STOP = "SKIP_BAD_STOP"
    SKIP_NO_BALANCE = "SKIP_NO_BALANCE"
    SKIP_SIZE = "SKIP_SIZE"
    SKIP_DAYCAP = "SKIP_DAYCAP"
    SKIP_RISK = "SKIP_RISK"
    SKIP_STALE_CANCEL = "SKIP_STALE_CANCEL"

    @property
    def priority(self) -> int:
        """Return 1-based §8 funnel priority (lowest number wins)."""
        return SKIP_CODE_PRIORITY[self]


SKIP_CODE_PRIORITY: dict[SkipCode, int] = {
    SkipCode.SKIP_INVALID_INTENT: 1,
    SkipCode.SKIP_CUTOFF: 2,
    SkipCode.SKIP_NO_PRICE: 3,
    SkipCode.SKIP_BAD_STOP: 4,
    SkipCode.SKIP_NO_BALANCE: 5,
    SkipCode.SKIP_SIZE: 6,
    SkipCode.SKIP_DAYCAP: 7,
    SkipCode.SKIP_RISK: 8,
    SkipCode.SKIP_STALE_CANCEL: 9,
}

SHADOW_SKIP_CODES: frozenset[SkipCode] = frozenset(
    {
        SkipCode.SKIP_NO_BALANCE,
        SkipCode.SKIP_SIZE,
        SkipCode.SKIP_DAYCAP,
        SkipCode.SKIP_RISK,
    }
)


def highest_priority_skip(codes: Iterable[SkipCode]) -> SkipCode:
    """Return the highest-priority (lowest §8 priority number) SkipCode."""
    best: SkipCode | None = None
    for code in codes:
        if not isinstance(code, SkipCode):
            raise TypeError(f"Expected SkipCode, got {type(code).__name__}")
        if best is None or code.priority < best.priority:
            best = code
    if best is None:
        raise ValueError("highest_priority_skip requires at least one SkipCode")
    return best


# ------------------------------------------------------------------------------
# Read-Only Market View Protocol (§11.1)
# ------------------------------------------------------------------------------
@runtime_checkable
class MarketView(Protocol):
    """Read-only time-gated market accessor exposed to strategies on MarketState."""

    ind: Any

    def index(self) -> Any:
        """Return read-only index bars visible strictly before t."""

    def vix(self) -> Any:
        """Return read-only VIX bars visible strictly before t."""

    def futures(self) -> Any:
        """Return read-only futures bars visible strictly before t."""

    def option(self, strike: int, side: str) -> Any:
        """Return read-only option bars for (strike, side) visible before t."""

    def get_chain(
        self,
        i: int,
        band_pct: float | None = None,
        asof: bool = False,
    ) -> Any:
        """Return read-only option chain snapshot at bar i (< t)."""

    def stocks(self, symbol: str) -> Any:
        """Return read-only constituent stock bars visible strictly before t."""


# ------------------------------------------------------------------------------
# Immutable Dataclasses (§3.3, §11.1, Niyam 13, Niyam 14)
# ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Intent:
    """Strategy entry intent returned by decide(state)."""

    side: str
    strike: int
    lots: int | None = None
    stop_frac: float | None = None
    stop_points: float | None = None
    exit_mode: str = "preset"
    target_r: float | None = None
    scratch_min: int = 0
    time_stop_min: int = 0
    breakeven_trigger_r: float = 0.0
    reason: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NoTrade:
    """Strategy no-trade decision returned by decide(state)."""

    reason: str = ""


@dataclass(frozen=True, slots=True)
class ManageAction:
    """Position management instruction returned by manage(state, pos)."""

    action: str
    new_stop: float | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PositionState:
    """Read-only snapshot of the currently open position."""

    side: str
    strike: int
    expiry: str
    dte: int
    lots: int
    lot_size: int
    qty: int
    entry_bar: int
    entry_ts: str
    entry_price: Decimal
    initial_stop: Decimal
    current_stop: Decimal
    stop_source: str
    target_price: Decimal | None
    exit_mode: str
    hold_min: int
    highest_high: Decimal
    lowest_low: Decimal
    reason: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TodayTradeSummary:
    """Read-only summary of a trade already closed earlier today (<= t-1)."""

    trade_no: int
    side: str
    strike: int
    lots: int
    entry_ts: str
    exit_ts: str
    entry_price: Decimal
    exit_price: Decimal
    exit_reason: str
    net_base: Decimal
    outcome: str


@dataclass(frozen=True, slots=True)
class MarketState:
    """Read-only decision-time market and account state passed to strategy."""

    date: str
    weekday: str
    t: int
    ts: str
    expiry: str
    dte: int
    dte_cal: int
    lot_size: int
    is_expiry_day: bool
    view: MarketView
    prior_days: tuple[Any, ...]
    today_trades: tuple[TodayTradeSummary, ...]
    balance_now: Decimal
    day_open_balance: Decimal
    realised_loss_today: Decimal
    entries_today: int
    track: str


@dataclass(frozen=True, slots=True)
class SkippedSignal:
    """Record of an Intent that did not fill, with its §8 SkipCode."""

    signal_ts: str
    side: str
    strike: int
    est_price: Decimal | None
    est_stop: Decimal | None
    skip_code: SkipCode
    detail: str = ""


def format_decimal(val: Decimal | None, places: int = 2) -> str:
    """Render a Decimal as a fixed-decimal string ('' for None)."""
    if val is None:
        return ""
    if not isinstance(val, Decimal):
        raise TypeError(f"Expected Decimal or None, got {type(val).__name__}")
    if not val.is_finite():
        raise ValueError(f"Non-finite Decimal cannot be formatted: {val}")
    quant = Decimal("1") if places == 0 else Decimal("1").scaleb(-places)
    return str(val.quantize(quant, rounding=ROUND_HALF_UP))


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Canonical 41-column closed trade record (§3.3)."""

    # Identity & Track (10)
    track: str
    trade_no: int
    date: str
    weekday: str
    signal_ts: str
    entry_ts: str
    side: str
    strike: int
    expiry: str
    dte: int
    # Sizing & Liquidity (5)
    lots_signal: int
    lots: int
    lot_size: int
    qty: int
    liquidity_warn: bool
    # Levels & Exit (9)
    entry_price: Decimal
    initial_stop: Decimal
    stop_pct: Decimal
    target_price: Decimal | None
    exit_ts: str
    exit_price: Decimal
    exit_reason: str
    gap_exit: bool
    hold_min: int
    # Accounting (8)
    gross: Decimal
    charges_base: Decimal
    fill_cost_base: Decimal
    net_base: Decimal
    charges_stress: Decimal
    fill_cost_stress: Decimal
    net_stress: Decimal
    net_proxy: Decimal | None
    # Account & Diagnostics (9)
    balance_base: Decimal
    balance_stress: Decimal
    outcome: str
    mae: Decimal
    mfe: Decimal
    mfe_min: int
    mae_min: int
    reason: str
    tags: str

    def to_csv_row(self) -> dict[str, str]:
        """Serialize TradeRecord into the locked 41-column CSV string mapping."""
        return {
            "track": self.track,
            "trade_no": str(self.trade_no),
            "date": self.date,
            "weekday": self.weekday,
            "signal_ts": self.signal_ts,
            "entry_ts": self.entry_ts,
            "side": self.side,
            "strike": str(self.strike),
            "expiry": self.expiry,
            "dte": str(self.dte),
            "lots_signal": str(self.lots_signal),
            "lots": str(self.lots),
            "lot_size": str(self.lot_size),
            "qty": str(self.qty),
            "liquidity_warn": "true" if self.liquidity_warn else "false",
            "entry_price": format_decimal(self.entry_price, 2),
            "initial_stop": format_decimal(self.initial_stop, 2),
            "stop_pct": format_decimal(self.stop_pct, 4),
            "target_price": format_decimal(self.target_price, 2),
            "exit_ts": self.exit_ts,
            "exit_price": format_decimal(self.exit_price, 2),
            "exit_reason": self.exit_reason,
            "gap_exit": "true" if self.gap_exit else "false",
            "hold_min": str(self.hold_min),
            "gross": format_decimal(self.gross, 2),
            "charges_base": format_decimal(self.charges_base, 2),
            "fill_cost_base": format_decimal(self.fill_cost_base, 2),
            "net_base": format_decimal(self.net_base, 2),
            "charges_stress": format_decimal(self.charges_stress, 2),
            "fill_cost_stress": format_decimal(self.fill_cost_stress, 2),
            "net_stress": format_decimal(self.net_stress, 2),
            "net_proxy": format_decimal(self.net_proxy, 2),
            "balance_base": format_decimal(self.balance_base, 2),
            "balance_stress": format_decimal(self.balance_stress, 2),
            "outcome": self.outcome,
            "mae": format_decimal(self.mae, 2),
            "mfe": format_decimal(self.mfe, 2),
            "mfe_min": str(self.mfe_min),
            "mae_min": str(self.mae_min),
            "reason": self.reason,
            "tags": self.tags,
        }


# ------------------------------------------------------------------------------
# Validation Helpers (§5.4, §8, §11.1, §11.3, Niyam 13)
# ------------------------------------------------------------------------------
def is_valid_int_strict(val: Any) -> bool:
    """Return True iff val is a plain Python int and NOT a bool (AUD-018)."""
    # Niyam 13 / AUD-018: isinstance(True, int) is True; reject bool first.
    if isinstance(val, bool):
        return False
    return type(val) is int


def is_finite_real(val: Any) -> bool:
    """Return True iff val is a plain Python int or float (not bool) and finite."""
    if isinstance(val, bool):
        return False
    if type(val) is int:
        return True
    if type(val) is float:
        return math.isfinite(val)
    return False


def validate_intent(intent: Any) -> tuple[bool, str]:
    """Validate an Intent against §5.4 and §8 Priority 1 rules.

    Returns (True, "") if valid, else (False, reason_detail) so engine.py
    emits SkipCode.SKIP_INVALID_INTENT.
    """
    if not isinstance(intent, Intent):
        return False, f"expected Intent, got {type(intent).__name__}"

    if intent.side not in VALID_SIDES:
        return False, f"invalid side: {intent.side!r}"

    if not is_valid_int_strict(intent.strike) or intent.strike <= 0:
        return False, f"invalid strike: {intent.strike!r}"

    # Niyam 13 / §5.4: lots must be None or positive plain int (bool rejected first)
    if intent.lots is not None:
        if not is_valid_int_strict(intent.lots) or intent.lots < 1:
            return False, f"invalid lots: {intent.lots!r}"

    # Exactly one of stop_frac or stop_points must be provided
    has_frac = intent.stop_frac is not None
    has_pts = intent.stop_points is not None
    if has_frac == has_pts:
        return False, "must specify exactly one of stop_frac or stop_points"

    if has_frac:
        if not is_finite_real(intent.stop_frac) or not (0.0 < float(intent.stop_frac) < 1.0):
            return False, f"invalid stop_frac: {intent.stop_frac!r}"
    else:
        if not is_finite_real(intent.stop_points) or float(intent.stop_points) <= 0.0:
            return False, f"invalid stop_points: {intent.stop_points!r}"

    if intent.exit_mode not in VALID_EXIT_MODES:
        return False, f"invalid exit_mode: {intent.exit_mode!r}"

    if intent.target_r is not None:
        if not is_finite_real(intent.target_r) or float(intent.target_r) <= 0.0:
            return False, f"invalid target_r: {intent.target_r!r}"

    if not is_valid_int_strict(intent.scratch_min) or intent.scratch_min < 0:
        return False, f"invalid scratch_min: {intent.scratch_min!r}"

    if not is_valid_int_strict(intent.time_stop_min) or intent.time_stop_min < 0:
        return False, f"invalid time_stop_min: {intent.time_stop_min!r}"

    if (
        not is_finite_real(intent.breakeven_trigger_r)
        or float(intent.breakeven_trigger_r) < 0.0
    ):
        return False, f"invalid breakeven_trigger_r: {intent.breakeven_trigger_r!r}"

    if not isinstance(intent.reason, str):
        return False, f"invalid reason type: {type(intent.reason).__name__}"

    if not isinstance(intent.tags, tuple) or not all(
        isinstance(tag, str) for tag in intent.tags
    ):
        return False, "tags must be a tuple of str"

    return True, ""


def validate_notrade_reason(
    reason: Any,
    seen_codes: set[str] | None = None,
) -> str:
    """Validate NoTrade.reason against §11.1 charset, length, and unique-code cap.

    Raises InvalidNoTradeReason on violation. Updates seen_codes in-place when
    provided and reason is non-empty.
    """
    if not isinstance(reason, str):
        raise InvalidNoTradeReason(
            f"NoTrade.reason must be str, got {type(reason).__name__}"
        )
    if reason == "":
        return ""
    if len(reason) > NOTRADE_REASON_MAX_LEN or not _NOTRADE_REASON_RE.match(reason):
        raise InvalidNoTradeReason(
            f"Invalid NoTrade.reason {reason!r}: must match [a-z0-9_], max {NOTRADE_REASON_MAX_LEN} chars"
        )
    if seen_codes is not None and reason not in seen_codes:
        if len(seen_codes) >= NOTRADE_MAX_UNIQUE_CODES:
            raise InvalidNoTradeReason(
                f"NoTrade.reason exceeds max {NOTRADE_MAX_UNIQUE_CODES} unique codes per strategy"
            )
        seen_codes.add(reason)
    return reason


def validate_manage_action(
    action: Any,
    current_stop: Decimal | None = None,
) -> ManageAction:
    """Validate a ManageAction returned by manage(state, pos) (§11.1, §11.3).

    Raises InvalidManageAction if action is malformed or TRAIL stop is
    non-finite, non-positive, or not strictly above current_stop.
    """
    if not isinstance(action, ManageAction):
        raise InvalidManageAction(
            f"Expected ManageAction, got {type(action).__name__}"
        )
    if action.action not in VALID_MANAGE_ACTIONS:
        raise InvalidManageAction(f"Invalid manage action: {action.action!r}")
    if not isinstance(action.reason, str):
        raise InvalidManageAction(
            f"ManageAction.reason must be str, got {type(action.reason).__name__}"
        )

    if action.action == "TRAIL":
        ns = action.new_stop
        if isinstance(ns, Decimal):
            if not ns.is_finite() or ns <= Decimal("0"):
                raise InvalidManageAction(f"Invalid TRAIL new_stop: {ns!r}")
            ns_dec = ns
        elif is_finite_real(ns) and float(ns) > 0.0:
            ns_dec = Decimal(str(ns))
        else:
            raise InvalidManageAction(f"Invalid TRAIL new_stop: {ns!r}")
        if current_stop is not None and ns_dec <= current_stop:
            raise InvalidManageAction(
                f"TRAIL new_stop ({ns_dec}) must be strictly > current_stop ({current_stop})"
            )
    else:
        if action.new_stop is not None:
            raise InvalidManageAction(
                f"ManageAction({action.action!r}) must have new_stop=None, got {action.new_stop!r}"
            )
    return action
