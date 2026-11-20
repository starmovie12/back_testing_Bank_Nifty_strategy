# BANKNIFTY Backtest Engine (`engine-v2.0`)

Deterministic, offline, zero-lookahead 1-minute historical replay and paper-account backtest engine for BANKNIFTY monthly index options (single long `BUY CE` / `BUY PE`), built strictly to **Backtest Engine PRD V4.0 (`engine-v2.0`)**.

## What This Engine Is (and Is Not)

- **Is:** An air-gapped, deterministic Python 3.11+ backtest engine (`stdlib + numpy`, optional `pyarrow`) using exact `Decimal` money math, verified Upstox per-order charges, Dual-Track execution, progressive weekly/design/holdout gates, and read-only postmortem diagnosis.
- **Is Not:** A live trading bot. It contains zero broker API clients, zero network/HTTP/socket calls, zero order placement, and zero credential handling. Live-bot parameters (`balance_max_age_sec`, `nse_freeze_qty`, SL-L gap-cancel contract) are validated as pure parity contracts only.

## Repository & Storage Layout

```text
banknifty-backtest/                 # Locked engine repository
  config.toml                       # Locked configuration (all numeric constants & registries)
  contracts.py                      # Locked (1/8): frozen dataclasses, 9 SkipCode enum, 41-col TradeRecord
  data_loader.py                    # Locked (2/8): updl CSV/Parquet loader, stale->NaN, holdout guard
  costs.py                          # Locked (3/8): Decimal Upstox charges, BASE/STRESS/PROXY, freeze split
  indicators.py                     # Locked (4/8): causal EMA, ATR, RSI, rolling high/low
  engine.py                         # Locked (5/8): 14 Niyam replay, Dual-Track sizing, cutoffs, RUIN
  report.py                         # Locked (6/8): Dual-Track REPORT.html/md, CSVs, robustness, verdicts
  diagnose.py                       # Locked (7/8): read-only postmortem, 12 autopsy classes, AI_HANDOFF.md
  run.py                            # Locked (8/8): CLI, LOCK.sha256, AST check, O_EXCL lock, fork+Pipe
  LOCK.sha256                       # SHA-256 manifest (9 locked files + sorted tests/ tree = 10 entries)
  strategies/                       # Unlocked: place strat_<name>.py files here
  tests/                            # Hash-locked 62-test acceptance suite (T01..T62)

Download/BankNiftyBT/               # base_dir (separate from repo)
  data/                             # updl week_XX_<from>_to_<to>/ folders
  strategies/                       # Optional external strategy folder (duplicate name with repo = error)
  output/<YYYY-MM-DD>_<HHMM>_<strategy>/
  weekly_state.json                 # Append-only sha256 hash-chained progressive state
  .engine.lock                      # Android FUSE-safe O_EXCL concurrency lockfile
  trials.jsonl                      # Append-only trial & variant counter (m, K)
  holdout_access.log                # Append-only fsync'd holdout audit log
```

Cache resides in internal storage (`~/.banknifty-bt/cache`), never inside Android FUSE `Download/`.

## Dual-Track Execution

Every run executes both tracks independently and labels every report table and header explicitly:

1. **`[FIXED 1-LOT]`:** Resets day-opening balance to `base_capital` every trading day (`k in {0, 1}`). Pure per-day execution parallelizable via `fork + Pipe`. Measures pure strategy edge without compounding distortion.
2. **`[DYNAMIC COMPOUNDING]`:** Strictly serial, chronological day-by-day and week-by-week balance carryover. Sizes the largest integer `k` passing Shart 1 (`85%` capital, with `100%` 1-lot exception), Shart 2 (`15%` daily loss cap anchored to `day_open_balance`), and optional Shart 3 (`per_trade_max_risk_pct`). Triggers deterministic `[DYNAMIC RUIN]` if `balance_now <= 0` or below zero-lookahead cheapest 1-lot cost (halting Dynamic while Fixed continues).

## Progressive Gate Order

Strategies advance strictly in order (no rewinding completed weeks, no summing weekly runs for the design gate):

1. **Weekly Progressive (`--week`):** Data-driven ISO weeks (`YYYY-Www`). `FAIL` (Dynamic DD `> 15%` or `RUIN`, checked first) -> streak resets to `0`. `LOW_SAMPLE` (`< 3` trades on either track) -> streak pauses (`> 2` consecutive resets to `0`). `PASS` (`>= 3` trades/track, `BASE > 0` & `STRESS > 0` on both tracks, Dynamic DD `<= 15%`) -> streak `+1`. **`WEEKLY_PASS`** unlocks at 4 consecutive PASS weeks with `>= 20` combined trades/track, combined `STRESS > 0`/track, and constant strategy hash.
2. **Design Gate (`--design-gate`):** Requires `WEEKLY_PASS`. Replays the entire DESIGN period (`< holdout_start`) fresh and evaluates all 6 gates (`T >= 100`, `STRESS > 0`, Bonferroni `p <= 0.05 / m`, bootstrap 95% CI `> 0`, positive folds `>= 60%`, delay-1 `> 0`, no Dynamic RUIN) -> **`DESIGN_PASS`**.
3. **Sealed Holdout (`--holdout`):** Requires `WEEKLY_PASS` and `DESIGN_PASS`. Unlocks once (`holdout_max_opens = 1`), evaluates Holdout gates (`T >= 20`, `STRESS > 0`, delay-1 `> 0`, `p <= 0.10 / K`), and suppresses tuning diagnosis -> **`HOLDOUT_PASS`**.
4. **`PAPER_READY`:** Awarded only when `WEEKLY_PASS`, `DESIGN_PASS`, and `HOLDOUT_PASS` are all achieved (followed by `paper_min_trades = 60` live paper trades).

## CLI Commands & Exit Codes

```bash
python run.py strat_orb --week next
python run.py strat_orb --week 2026-W47 --exp E-D01-1
python run.py strat_orb --design-gate --exp E-D04-1
python run.py strat_orb --holdout
python run.py strat_orb --reset-ruin --confirm-capital 50000
python run.py strat_orb --from 2026-10-01 --to 2026-11-20 --workers 2 --diag-level lite
python run.py export-csv
```

| Exit Code | Meaning |
|---|---|
| `0` | Success (including `FAIL` / `NOT_YET` strategy verdicts) |
| `1` | Unexpected runtime error |
| `2` | Invalid CLI arguments or config validation error (e.g. empty `base_capital`, legacy V3.1 keys) |
| `3` | `LOCK.sha256` / `git diff` failure or strategy AST static-check violation |
| `4` | `weekly_state.json` corrupt/tampered or active `.engine.lock` concurrency conflict |
| `5` | Week rewind refused (`WEEK_ALREADY_EXECUTED_IN_LINEAGE`) |
| `6` | Holdout guard refusal (logged to `holdout_access.log`) |
| `7` | Prior Dynamic `RUIN` requires `--reset-ruin --confirm-capital <INR>` |
| `8` | Gate order violation (`WEEKLY_PASS` required before `--design-gate`) |

## Configuration Summary

All 83 engine constants, dated Upstox charge rates, spread/slippage multipliers, 19 `[diag_thresholds]`, and 17 `[diag_grid]` entries live exclusively in locked `config.toml` (zero magic numbers in code). Only Machine Knobs (`base_dir`, `cache_dir`, `workers`, `peak_memory_mb`, `batch_days`, `diag_level`) may be overridden via CLI without changing trade outputs.

## Run Outputs (11 Mandatory + Optional `DIFF.md`)

1. `REPORT.html` (self-contained HTML5 + `<details>` + 5 inline SVGs, zero JS/CDN, 500-row cap with reconciling `Others` row)
2. `REPORT.md` (plain Markdown mirror with `[FIXED 1-LOT]` and `[DYNAMIC COMPOUNDING]` headers)
3. `trades_fixed.csv` (41-column streaming UTF-8 CSV)
4. `trades_dynamic.csv` (41-column streaming UTF-8 CSV)
5. `summary.json` (separate `fixed` and `dynamic` summaries, robustness pack, verdict)
6. `manifest.json` (hashes, `engine-v2.0`, mode, `seed`, `m`, `K`, `git_check`, `parallel`, `diagnosis`)
7. `verify_sample.md` (seeded sample of up to 10 Fixed + 5 Dynamic trades with 1m candles)
8. `strategy_snapshot.py` (exact byte copy of executed strategy)
9. `DIAGNOSIS.md` (Doctor ki Parchi on Fixed track + Dynamic risk panel)
10. `AI_HANDOFF.md` (plain Markdown strictly `<= 200` lines and `<= 24576` bytes)
11. `diagnosis.json` (machine-readable postmortem, 12 autopsy classes, 6-point `milan`)
12. `DIFF.md` *(optional — emitted on non-holdout runs when a prior same-family `engine-v2.0` run exists)*

## Adding a Strategy

Place `strat_<name>.py` inside `strategies/`. Define `PARAMS = {...}` and either `class Strategy` or top-level `decide(state)` (plus optional `manage(state, pos)`). Return `Intent(...)` or `NoTrade("short_code")`. Only these imports are permitted by the AST static checker: `math`, `numpy`, `statistics`, `itertools`, `dataclasses`, `collections`, `typing`, `contracts`, `indicators`. No file I/O, randomness, `global`/`nonlocal`, or class-level mutable state.

## Locking & Tagging Procedure (`engine-v2.0`)

After setting `base_capital` in `config.toml`, verifying `62/62` tests green (`pytest`), and regenerating `LOCK.sha256`:

```bash
chmod a-w config.toml contracts.py data_loader.py costs.py indicators.py engine.py report.py diagnose.py run.py LOCK.sha256
git add .
git commit -m "Lock BANKNIFTY Backtest Engine engine-v2.0"
git tag engine-v2.0
```

## Open Owner Decisions (`OWNER_DECISION_NEEDED`)

1. **`base_capital`**: Blank in `config.toml` (startup exits `2` until set). Recommended: `"50000"`.
2. **`per_trade_max_risk_pct`**: Default `""` (`None`). Recommended: `"0.10"` if `base_capital >= 20300`.
3. **`holdout_start` & `holdout_prior_opens`**: Defaults `"2026-07-06"` and `1` (S2 open on 2026-11-18).
4. **`nse_freeze_qty`**: Default `""` (`None`). Confirm from latest NSE BankNifty circular before live deployment.
5. **`lot_size` fallback & `nse_holidays`**: Defaults `[]` (`contracts.csv` is primary).

See `SETUP_TERMUX.md` for Android Termux setup instructions.
