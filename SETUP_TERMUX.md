# Android Termux Setup Guide (`engine-v2.0`)

Step-by-step setup for running **BANKNIFTY Backtest Engine (`engine-v2.0`)** on Android Termux (Python 3.11+, zero `sem_open` / `Pool` / `Queue` / `fcntl.flock` dependencies).

> **Note:** Steps marked **(verify on device)** depend on your specific Termux build and Android storage permissions and could not be executed inside a desktop Linux sandbox.

## 1. Storage Permission & Base Packages

Grant Termux access to Android `Download/` storage **(verify on device)**:

```bash
termux-setup-storage
```

Update packages and install Python 3.11+, Git, and NumPy **(verify on device)**:

```bash
pkg update && pkg upgrade -y
pkg install -y python git python-numpy
```

Check Python and NumPy versions:

```bash
python --version
python -c "import numpy; print(numpy.__version__)"
```

Install `pytest` to run the 62-test acceptance suite **(verify on device)**:

```bash
pip install pytest
```

*(Note: `pyarrow` is optional and often unavailable on Android ARM Termux. CSV mode works natively without `pyarrow`; if your `updl` data is Parquet-only, run `python run.py export-csv` on a PC or Python environment with `pyarrow` to generate `.csv.gz` tables.)*

## 2. Folder Layout (§3.1)

Create the exact runtime folders (`base_dir` in shared Downloads, `cache_dir` strictly in internal Termux storage — NEVER on Android FUSE `Download/`):

```bash
mkdir -p ~/storage/downloads/BankNiftyBT/data
mkdir -p ~/storage/downloads/BankNiftyBT/strategies
mkdir -p ~/storage/downloads/BankNiftyBT/output
mkdir -p ~/.banknifty-bt/cache
```

Extract the repository into `~/banknifty-backtest`:

```bash
cd ~
unzip ~/storage/downloads/banknifty-backtest_files-01-to-27.zip
cd ~/banknifty-backtest
```

Verify the repo layout matches PRD §3.1:
- `config.toml`, `contracts.py`, `data_loader.py`, `costs.py`, `indicators.py`, `engine.py`, `report.py`, `diagnose.py`, `run.py`, `LOCK.sha256`
- `strategies/` (unlocked; place `strat_<name>.py` here)
- `tests/` (62 acceptance tests `T01`..`T62`)

## 3. Set Owner Config Decisions (Before Locking)

Edit `config.toml` to fill `base_capital` (currently `""`, which exits with code `2` until set) and confirm any other `OWNER_DECISION_NEEDED` keys (`per_trade_max_risk_pct`, `holdout_start`, `holdout_prior_opens`, `nse_freeze_qty`, `lot_size`, `nse_holidays`).

## 4. Run Acceptance Tests & Termux Smoke Test (T43 + T16)

Run the Termux `fork + Pipe` smoke test (`T43`) and folder bootstrap test (`T16`) first **(verify on device)**:

```bash
cd ~/banknifty-backtest
pytest tests/test_termux_smoke.py -v
pytest tests/test_diagnose_core.py -k test_T16 -v
```

Run the full 62-test acceptance suite (must be `62 passed` before locking):

```bash
pytest -v
```

## 5. Regenerate `LOCK.sha256`, Write-Protect Locked Files & Tag `engine-v2.0`

If you edited `base_capital` (or any other owner decision) in `config.toml`, regenerate `LOCK.sha256` using `run.py`'s canonical manifest builder:

```bash
python -c "from pathlib import Path; import run; Path('LOCK.sha256').write_text(run.build_lock_manifest(Path('.'), require_all=True), encoding='utf-8', newline='\n')"
```

Write-protect the 9 locked files plus `LOCK.sha256` (`chmod a-w` is an owner step, never applied inside the zip):

```bash
chmod a-w config.toml contracts.py data_loader.py costs.py indicators.py engine.py report.py diagnose.py run.py LOCK.sha256
```

Initialize Git (if not already a git repo) and create the locked `engine-v2.0` tag:

```bash
git init
git add .
git commit -m "Lock BANKNIFTY Backtest Engine engine-v2.0"
git tag engine-v2.0
```

## 6. First Backtest Run on Phone

Copy your `updl` `week_*` folders into `~/storage/downloads/BankNiftyBT/data/`, place your strategy in `~/banknifty-backtest/strategies/strat_<name>.py`, and run:

```bash
python run.py strat_<name> --week next
```

Outputs are written atomically to `~/storage/downloads/BankNiftyBT/output/<YYYY-MM-DD>_<HHMM>_strat_<name>/`. Open `REPORT.html` directly in any mobile browser (zero JS, zero CDN, 5 inline SVGs).
