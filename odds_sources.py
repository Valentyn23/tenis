from __future__ import annotations

from typing import Callable, Dict, List

from odds_theoddsapi import best_decimal_odds_from_event, fetch_h2h_odds_for_sport
from odds_rapidapi_flashscore import RapidAPIOddsError, fetch_flashscore_odds


def _dedupe(events: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for ev in events:
        a = str(ev.get("playerA", "")).strip().lower()
        b = str(ev.get("playerB", "")).strip().lower()
        dt = str(ev.get("commence_time", ""))[:10]
        key = tuple(sorted((a, b))) + (dt,)
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def _normalize_theodds_event(event: Dict) -> Dict:
    out = dict(event)
    tournament_key = str(out.get("sport_key") or "unknown_tournament")
    out["source"] = "theodds"
    out["tournament_key"] = tournament_key
    out["tournament_name"] = tournament_key
    return out


def collect_events_with_odds(
    provider: str,
    tennis_keys: List[str],
    regions: str,
    max_events: int,
    logger: Callable[[str], None] | None = None,
) -> tuple[List[Dict], Dict[str, object]]:
    provider = (provider or "theodds").strip().lower()
    stats: Dict[str, object] = {
        "theodds": 0,
        "rapidapi_flashscore": 0,
        "tournaments_by_source": {"theodds": [], "rapidapi_flashscore": []},
    }
    events: List[Dict] = []

    use_theodds = provider in {"theodds", "both"}
    use_rapid = provider in {"rapidapi_flashscore", "both"}

    found_theodds_tournaments = set()
    found_rapid_tournaments = set()

    if use_theodds:
        for k in tennis_keys:
            try:
                data = fetch_h2h_odds_for_sport(k, regions=regions, odds_format="decimal")
                any_for_key = False
                for e in data:
                    best = best_decimal_odds_from_event(e)
                    if best:
                        any_for_key = True
                        norm = _normalize_theodds_event(best)
                        events.append(norm)
                        stats["theodds"] = int(stats["theodds"]) + 1
                        if len(events) >= max_events and provider != "both":
                            break
                if any_for_key:
                    found_theodds_tournaments.add(k)
                if len(events) >= max_events and provider != "both":
                    break
            except Exception as ex:
                if logger:
                    logger(f"TheOdds error for {k}: {ex}")

    if use_rapid:
        try:
            rapid = fetch_flashscore_odds(max_events=max_events * 2)
            for ev in rapid:
                tk = str(ev.get("tournament_key") or "")
                # if TheOdds already covers this tournament key, skip rapid duplicate tournaments
                if provider == "both" and tk and tk in found_theodds_tournaments:
                    continue
                found_rapid_tournaments.add(tk or "unknown_tournament")
                events.append(ev)
        except RapidAPIOddsError as ex:
            if logger:
                logger(f"RapidAPI config error: {ex}")
        except Exception as ex:
            if logger:
                logger(f"RapidAPI fetch error: {ex}")

    deduped = _dedupe(events)[:max_events]

    # recompute final counts by source after dedupe
    theodds_count = 0
    rapid_count = 0
    final_theodds_tournaments = set()
    final_rapid_tournaments = set()
    for ev in deduped:
        src = str(ev.get("source") or "")
        tk = str(ev.get("tournament_key") or "unknown_tournament")
        if src == "rapidapi_flashscore":
            rapid_count += 1
            final_rapid_tournaments.add(tk)
        else:
            theodds_count += 1
            final_theodds_tournaments.add(tk)

    stats["theodds"] = theodds_count
    stats["rapidapi_flashscore"] = rapid_count
    stats["tournaments_by_source"] = {
        "theodds": sorted(final_theodds_tournaments),
        "rapidapi_flashscore": sorted(final_rapid_tournaments),
    }

    return deduped, stats
