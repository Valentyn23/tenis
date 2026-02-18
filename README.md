# tenis

## Runtime notes

For mixed ATP/WTA inference, use **separate warmed states**:

```bash
MODE=ATP python wrump.py
MODE=WTA python wrump.py
```

By default warmup saves to:
- `state/engine_state_atp.pkl`
- `state/engine_state_wta.pkl`

`app.py` loads them automatically for each tour, but you can override:
- `STATE_PATH_ATP`
- `STATE_PATH_WTA`

Enable strict mode consistency check (recommended, default on):

```bash
STRICT_MODE_MATCH=1 python app.py
```

## Bankroll calculator

Interactive script `bankroll.py` calculates recommended stake for a single outcome.

Run:

```bash
python bankroll.py
```

Inputs:
- bankroll
- decimal odds
- your estimated probability for the selected outcome (`0..1`)
- Kelly fraction (default `0.5`)
- max stake cap as percent of bankroll (default `0.03`)
- minimum edge threshold (default `0.015`)

Output:
- market implied probability
- edge
- Kelly fractions
- recommended stake amount
- `BET` / `NO_BET` decision
