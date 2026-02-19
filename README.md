# tenis

Tennis betting helper pipeline:
- warm up rating/state from historical ATP/WTA datasets,
- fetch current odds,
- run tour-specific model inference,
- produce risk-controlled stake recommendations.

## Requirements

Recommended Python: `3.10+`

Install dependencies in your venv:

```bash
pip install -U pip
pip install pandas openpyxl xgboost scikit-learn joblib requests python-dotenv pytest streamlit
```

## Quick start

### 1) Warm up ATP and WTA states separately

```bash
MODE=ATP python wrump.py
MODE=WTA python wrump.py
```

By default this creates:
- `state/engine_state_atp.pkl`
- `state/engine_state_wta.pkl`

### 2) Run app

```bash
STRICT_MODE_MATCH=1 python app.py
```

## Runtime configuration (env)

Core:
- `ODDS_REGIONS` (default `eu`)
- `MAX_EVENTS` (default `30`)
- `ONLY_ACTIVE_TENNIS` (default `1`) — set `0` to fetch all tennis sport keys, not only active ones
- `EXTRA_TENNIS_KEYS` (optional) — comma-separated manual keys to force include
- `STRICT_MODE_MATCH` (default `1`)

State paths:
- `STATE_PATH_ATP` (default `state/engine_state_atp.pkl`)
- `STATE_PATH_WTA` (default `state/engine_state_wta.pkl`)

Risk profile:
- `RISK_PROFILE`: `conservative` / `balanced` / `aggressive` (default `balanced`)
- overrides: `MAX_STAKE_PCT`, `KELLY_FRACTION`, `MIN_EDGE`, `PROB_FLOOR`, `PROB_CEIL`, `MAX_OVERROUND`, `SOFT_CAP_EDGE`, `SOFT_CAP_FACTOR`

Reports:
- `SAVE_REPORT` (default `1`)
- `REPORT_DIR` (default `reports`)
- `PRINT_TOP_N` (default `25`)

Money / ledger:
- `CURRENCY` (default `UAH`)
- `BETS_LEDGER_PATH` (default `bets/ledger.csv`)

Gemini:
- `GEMINI_PICK_OPINION` (default `0`) — ask Gemini for secondary opinion on each BET pick
- `GEMINI_MODEL` (optional) — force model name, e.g. `gemini-1.5-flash-latest`


## Streamlit UI (MVP)

Launch visual UI:

```bash
streamlit run ui_app.py
```

Do not start it with `python ui_app.py` — Streamlit apps require `streamlit run`.

UI includes:
The Streamlit UI now always discovers from all tennis keys and lets you manually include specific tournament keys.

- settings panel (bankroll, risk profile, max events, discovered tournament key picker, manual tournament keys, calibration toggle),
- one-click run button,
- recommendations table with color highlights (green = BET, red = NO_BET/SKIP),
- Gemini secondary opinion columns,
- ledger tab reading `bets/ledger.csv`,
- basic charts for decisions, stakes, and edges.

## Typical troubleshooting

### WTA/ATP predictor unavailable

If you see:
- `Predictor WTA unavailable ... Missing: state/engine_state_wta.pkl`

Run warmup for that tour:

```bash
MODE=WTA python wrump.py
```

### Mode mismatch (strict mode)

If `engine_mode != model_mode`, app stops in strict mode.
Use matching state/model files or set (not recommended):

```bash
STRICT_MODE_MATCH=0
```

### Gemini shows "Gemini unavailable"

Set API key in project `.env` and restart Streamlit:

```bash
GEMINI_API_KEY=your_key_here
GEMINI_PICK_OPINION=1
# optional override
GEMINI_MODEL=gemini-1.5-flash-latest
```

Then run:

```bash
streamlit run ui_app.py
```

### Too many `SKIP_MARKET`

Check:
- odds range filters,
- overround (`MAX_OVERROUND`),
- whether odds feed quality is low.

## Bankroll calculator

Interactive utility for one-off stake sizing:

```bash
python bankroll.py
```

Inputs:
- bankroll
- decimal odds
- your estimated probability (`0..1`)
- Kelly fraction
- max stake cap
- minimum edge threshold

Outputs:
- implied probability
- edge
- Kelly fractions
- recommended stake
- `BET` / `NO_BET`

## Bet ledger and post-analysis

`app.py` now writes every bet decision (`BET_A` / `BET_B`) to CSV ledger.

Default path:

```bash
bets/ledger.csv
```

Ledger columns include:
- timestamp
- event id
- match players
- pick/decision
- odds / probability / edge
- stake in selected `CURRENCY` (UAH by default)
- result and PnL (for settled bets)

Current app runtime writes new bet picks to ledger only (clean mode, no auto-settlement in `app.py`).

## Tests

```bash
pytest -q
```


## Retrain + rebuild (ATP/WTA, includes 2026 data)

The datasets under `data/ATP/*.xls*` and `data/WTA/*.xls*` include 2026 files.
Use the helper script to retrain both models and rebuild both states:

```bash
python retrain_rebuild_validate.py
```

This runs:
- `train/train.py` with `MODE=ATP`
- `train/train.py` with `MODE=WTA`
- `wrump.py` with `MODE=ATP`
- `wrump.py` with `MODE=WTA`
- `app.py` smoke run in strict mode
