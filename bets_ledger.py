from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


LEDGER_FIELDS = [
    "timestamp_utc",
    "strategy",
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
                "strategy": p.get("strategy", "balanced"),
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


def append_selected_bets(ledger_path: str, selected_picks: list[dict], currency: str) -> int:
    """Append only selected picks to ledger."""
    path = Path(ledger_path)
    _ensure_parent(path)

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for p in selected_picks:
        rows.append(
            {
                "timestamp_utc": now,
                "strategy": p.get("strategy", "balanced"),
                "event_id": p.get("event_id"),
                "tour_mode": p.get("tour_mode"),
                "commence_time": p.get("commence_time"),
                "playerA": p.get("playerA"),
                "playerB": p.get("playerB"),
                "pick": p.get("pick"),
                "decision": p.get("decision"),
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


def update_ledger_result(ledger_path: str, row_index: int, result: str, pnl: float) -> bool:
    """Update result and pnl for a specific row in ledger."""
    path = Path(ledger_path)
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if row_index < 0 or row_index >= len(rows):
        return False

    rows[row_index]["result"] = result
    rows[row_index]["pnl"] = str(pnl)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return True




def _parse_iso_dt(value: str):
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _norm_player_name(name: str) -> str:
    s = str(name or "").lower().strip()
    for ch in [".", "-", "_"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def settle_from_finished_events(
    ledger_path: str,
    finished_events: list[dict],
    max_hours_diff: float = 18.0,
) -> dict[str, int]:
    """Settle open bets by exact event_id first, then by (players + time) fallback."""
    ledger = Path(ledger_path)
    if not ledger.exists() or not finished_events:
        return {"settled": 0, "by_event_id": 0, "by_match_fallback": 0}

    with ledger.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    by_id: dict[str, str] = {}
    fallback_candidates: list[dict] = []
    for ev in finished_events:
        if not isinstance(ev, dict):
            continue
        winner = str(ev.get("winner") or "").strip()
        if not winner:
            continue
        eid = str(ev.get("event_id") or "").strip()
        if eid:
            by_id[eid] = winner
        a = _norm_player_name(ev.get("playerA"))
        b = _norm_player_name(ev.get("playerB"))
        if a and b:
            fallback_candidates.append({
                "p1": a,
                "p2": b,
                "winner": winner,
                "commence_time": _parse_iso_dt(ev.get("commence_time") or ""),
            })

    settled = 0
    by_event_id_count = 0
    by_fallback_count = 0

    for row in rows:
        if row.get("result"):
            continue

        winner = ""
        event_id = str(row.get("event_id") or "").strip()
        if event_id and event_id in by_id:
            winner = by_id[event_id]
            by_event_id_count += 1
        else:
            ra = _norm_player_name(row.get("playerA"))
            rb = _norm_player_name(row.get("playerB"))
            if not (ra and rb):
                continue
            row_dt = _parse_iso_dt(row.get("commence_time") or "")

            best = None
            best_hours = float("inf")
            for c in fallback_candidates:
                if {ra, rb} != {c["p1"], c["p2"]}:
                    continue
                cdt = c.get("commence_time")
                if row_dt is not None and cdt is not None:
                    diff_h = abs((row_dt - cdt).total_seconds()) / 3600.0
                    if diff_h > max_hours_diff:
                        continue
                else:
                    diff_h = 0.0
                if diff_h < best_hours:
                    best_hours = diff_h
                    best = c

            if best is not None:
                winner = str(best["winner"])
                by_fallback_count += 1

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

    return {"settled": settled, "by_event_id": by_event_id_count, "by_match_fallback": by_fallback_count}

def settle_from_winners_map(ledger_path: str, winners: dict[str, str]) -> int:
    """Settle open bets using in-memory winners map {event_id: winner_name}."""
    ledger = Path(ledger_path)
    if not ledger.exists() or not winners:
        return 0

    with ledger.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    settled = 0
    for row in rows:
        if row.get("result"):
            continue
        event_id = str(row.get("event_id", "")).strip()
        winner = str(winners.get(event_id, "")).strip()
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


def settle_from_results_csv(ledger_path: str, results_csv: str) -> int:
    """Settle open bets using CSV with columns: event_id,winner.

    winner must match one of player names from ledger row.
    Returns number of settled rows.
    """
    results = Path(results_csv)
    if not results.exists():
        return 0

    with results.open("r", encoding="utf-8", newline="") as f:
        rr = csv.DictReader(f)
        winners = {str(r.get("event_id", "")).strip(): str(r.get("winner", "")).strip() for r in rr}

    return settle_from_winners_map(ledger_path, winners)


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
    by_strategy: dict[str, dict[str, float | int | None]] = {}
    for r in closed:
        try:
            stake = float(r.get("stake") or 0.0)
            pnl = float(r.get("pnl") or 0.0)
            total_staked += stake
            total_pnl += pnl

            strategy = str(r.get("strategy") or "balanced")
            agg = by_strategy.setdefault(strategy, {"closed_bets": 0, "total_staked": 0.0, "total_pnl": 0.0, "roi": None})
            agg["closed_bets"] = int(agg["closed_bets"]) + 1
            agg["total_staked"] = float(agg["total_staked"]) + stake
            agg["total_pnl"] = float(agg["total_pnl"]) + pnl
        except Exception:
            pass

    for strategy, agg in by_strategy.items():
        staked = float(agg["total_staked"])
        pnl = float(agg["total_pnl"])
        agg["total_staked"] = round(staked, 2)
        agg["total_pnl"] = round(pnl, 2)
        agg["roi"] = round((pnl / staked), 4) if staked > 0 else None

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
        "by_strategy": by_strategy,
    }
