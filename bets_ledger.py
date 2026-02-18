from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


LEDGER_FIELDS = [
    "timestamp_utc",
    "event_id",
    "tour_mode",
    "commence_time",
    "playerA",
    "playerB",
    "pick",
    "decision",
    "reason",
    "oddsA",
    "oddsB",
    "pick_odds",
    "prob_A_win",
    "pick_edge",
    "stake",
    "currency",
    "result",
    "pnl",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_bets(ledger_path: str, picks: Iterable[dict], currency: str) -> int:
    """Append BET_A/B rows to ledger and return number of appended rows."""
    path = Path(ledger_path)
    _ensure_parent(path)

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for p in picks:
        decision = str(p.get("decision") or "")
        if decision not in {"BET_A", "BET_B"}:
            continue

        rows.append(
            {
                "timestamp_utc": now,
                "event_id": p.get("event_id"),
                "tour_mode": p.get("tour_mode"),
                "commence_time": p.get("commence_time"),
                "playerA": p.get("playerA"),
                "playerB": p.get("playerB"),
                "pick": p.get("pick"),
                "decision": decision,
                "reason": p.get("reason"),
                "oddsA": p.get("oddsA"),
                "oddsB": p.get("oddsB"),
                "pick_odds": p.get("pick_odds"),
                "prob_A_win": p.get("prob_A_win"),
                "pick_edge": p.get("pick_edge"),
                "stake": p.get("stake"),
                "currency": currency,
                "result": "",
                "pnl": "",
            }
        )

    if not rows:
        return 0

    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def settle_from_results_csv(ledger_path: str, results_csv: str) -> int:
    """Settle open bets using CSV with columns: event_id,winner.

    winner must match one of player names from ledger row.
    Returns number of settled rows.
    """
    ledger = Path(ledger_path)
    results = Path(results_csv)
    if not ledger.exists() or not results.exists():
        return 0

    with results.open("r", encoding="utf-8", newline="") as f:
        rr = csv.DictReader(f)
        winners = {str(r.get("event_id", "")).strip(): str(r.get("winner", "")).strip() for r in rr}

    with ledger.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    settled = 0
    for row in rows:
        if row.get("result"):
            continue
        event_id = str(row.get("event_id", "")).strip()
        winner = winners.get(event_id)
        if not winner:
            continue

        pick = str(row.get("pick", "")).strip()
        try:
            stake = float(row.get("stake") or 0.0)
            pick_odds = float(row.get("pick_odds") or 0.0)
        except Exception:
            continue

        if winner == pick:
            row["result"] = "win"
            row["pnl"] = f"{stake * (pick_odds - 1.0):.2f}"
        else:
            row["result"] = "loss"
            row["pnl"] = f"{-stake:.2f}"
        settled += 1

    with ledger.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return settled


def analyze_ledger(ledger_path: str, bankroll_start: Optional[float] = None) -> dict:
    path = Path(ledger_path)
    if not path.exists():
        return {
            "total_bets": 0,
            "closed_bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "total_staked": 0.0,
            "total_pnl": 0.0,
            "roi": None,
            "current_bankroll": bankroll_start,
        }

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total_bets = len(rows)
    closed = [r for r in rows if (r.get("result") or "") in {"win", "loss"}]
    wins = sum(1 for r in closed if r.get("result") == "win")
    losses = sum(1 for r in closed if r.get("result") == "loss")

    total_staked = 0.0
    total_pnl = 0.0
    for r in closed:
        try:
            total_staked += float(r.get("stake") or 0.0)
            total_pnl += float(r.get("pnl") or 0.0)
        except Exception:
            pass

    roi = (total_pnl / total_staked) if total_staked > 0 else None
    current_bankroll = (bankroll_start + total_pnl) if bankroll_start is not None else None

    return {
        "total_bets": total_bets,
        "closed_bets": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(closed)) if closed else None,
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 4) if roi is not None else None,
        "current_bankroll": round(current_bankroll, 2) if current_bankroll is not None else None,
    }
