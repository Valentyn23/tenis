# tenis

## Bankroll calculator

Added interactive script `bankroll.py` to calculate recommended stake for a single outcome.

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
