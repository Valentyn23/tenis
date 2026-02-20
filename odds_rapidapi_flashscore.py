from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

TIMEOUT = 20


TOUR_KEY_ALIASES = {
    "aus open": {"atp": "tennis_atp_aus_open_singles", "wta": "tennis_wta_aus_open_singles"},
    "australian open": {"atp": "tennis_atp_aus_open_singles", "wta": "tennis_wta_aus_open_singles"},
    "canadian open": {"atp": "tennis_atp_canadian_open", "wta": "tennis_wta_canadian_open"},
    "china open": {"atp": "tennis_atp_china_open", "wta": "tennis_wta_china_open"},
    "cincinnati": {"atp": "tennis_atp_cincinnati_open", "wta": "tennis_wta_cincinnati_open"},
    "dubai": {"atp": "tennis_atp_dubai", "wta": "tennis_wta_dubai"},
    "french open": {"atp": "tennis_atp_french_open", "wta": "tennis_wta_french_open"},
    "indian wells": {"atp": "tennis_atp_indian_wells", "wta": "tennis_wta_indian_wells"},
    "italian open": {"atp": "tennis_atp_italian_open", "wta": "tennis_wta_italian_open"},
    "madrid": {"atp": "tennis_atp_madrid_open", "wta": "tennis_wta_madrid_open"},
    "miami": {"atp": "tennis_atp_miami_open", "wta": "tennis_wta_miami_open"},
    "qatar": {"atp": "tennis_atp_qatar_open", "wta": "tennis_wta_qatar_open"},
    "us open": {"atp": "tennis_atp_us_open", "wta": "tennis_wta_us_open"},
    "wimbledon": {"atp": "tennis_atp_wimbledon", "wta": "tennis_wta_wimbledon"},
    "wuhan": {"wta": "tennis_wta_wuhan_open"},
    "monte-carlo": {"atp": "tennis_atp_monte_carlo_masters"},
    "paris masters": {"atp": "tennis_atp_paris_masters"},
    "shanghai": {"atp": "tennis_atp_shanghai_masters"},
}



class RapidAPIOddsError(RuntimeError):
    pass


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _pick(*vals: Any) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _dig(d: Any, *path: str) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if v > 1.0 else None
    except Exception:
        return None


def _extract_odds(ev: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    # common structures
    oa = _to_float(_dig(ev, "odds", "home")) or _to_float(_dig(ev, "odds", "1"))
    ob = _to_float(_dig(ev, "odds", "away")) or _to_float(_dig(ev, "odds", "2"))
    if oa and ob:
        return oa, ob

    markets = ev.get("markets") or ev.get("betting") or []
    for m in markets if isinstance(markets, list) else []:
        outcomes = m.get("outcomes") if isinstance(m, dict) else None
        if not isinstance(outcomes, list):
            continue
        vals: Dict[str, float] = {}
        for out in outcomes:
            if not isinstance(out, dict):
                continue
            name = str(out.get("name") or out.get("label") or "").lower()
            price = _to_float(out.get("price") or out.get("odds") or out.get("value"))
            if not price:
                continue
            if name in {"1", "home", "player 1", "a"}:
                vals["a"] = price
            elif name in {"2", "away", "player 2", "b"}:
                vals["b"] = price
        if vals.get("a") and vals.get("b"):
            return vals["a"], vals["b"]

    return None, None


def _extract_events(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("events", "data", "results", "response", "matches"):
            obj = payload.get(key)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                nested = obj.get("events") or obj.get("data")
                if isinstance(nested, list):
                    return [x for x in nested if isinstance(x, dict)]
    return []


def _infer_tour_and_tournament(ev: Dict[str, Any]) -> tuple[str, str, str]:
    tournament_name = _pick(
        _dig(ev, "tournament", "name"),
        ev.get("tournament"),
        _dig(ev, "league", "name"),
        ev.get("league"),
        ev.get("category"),
    ) or "unknown_tournament"

    text = " ".join(
        str(x)
        for x in [
            ev.get("sport"),
            tournament_name,
            ev.get("category"),
            ev.get("series"),
        ]
        if x
    ).lower()
    tour = "wta" if "wta" in text else "atp"

    normalized_key = f"tennis_{tour}_flashscore"
    for alias, mapping in TOUR_KEY_ALIASES.items():
        if alias in text:
            normalized_key = mapping.get(tour) or mapping.get("atp") or mapping.get("wta") or normalized_key
            break

    sport_key = f"tennis_{tour}_flashscore"
    return sport_key, normalized_key, tournament_name


def normalize_rapidapi_event(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    home = _pick(
        _dig(ev, "homeTeam", "name"),
        _dig(ev, "home", "name"),
        ev.get("home_name"),
        ev.get("homeTeamName"),
        ev.get("home"),
        _dig(ev, "competitors", "home", "name"),
    )
    away = _pick(
        _dig(ev, "awayTeam", "name"),
        _dig(ev, "away", "name"),
        ev.get("away_name"),
        ev.get("awayTeamName"),
        ev.get("away"),
        _dig(ev, "competitors", "away", "name"),
    )
    if not home or not away:
        return None

    odds_a, odds_b = _extract_odds(ev)
    if not odds_a or not odds_b:
        return None

    sport_key, tournament_key, tournament_name = _infer_tour_and_tournament(ev)
    commence = _pick(ev.get("startTime"), ev.get("start_time"), ev.get("commence_time"), ev.get("date"))
    return {
        "playerA": home,
        "playerB": away,
        "oddsA": float(odds_a),
        "oddsB": float(odds_b),
        "commence_time": commence,
        "bookmakers_used": ["rapidapi_flashscore", "rapidapi_flashscore"],
        "sport_key": sport_key,
        "tournament_key": tournament_key,
        "tournament_name": tournament_name,
        "id": _pick(ev.get("id"), ev.get("event_id"), f"rapid_{home}_{away}_{commence}"),
        "source": "rapidapi_flashscore",
    }


def fetch_flashscore_odds(max_events: int = 200) -> List[Dict[str, Any]]:
    api_key = _env("RAPIDAPI_KEY")
    host = _env("RAPIDAPI_HOST")
    url = _env("RAPIDAPI_FLASHSCORE_URL", "https://flashscore-api.p.rapidapi.com/v1/events/list")
    sport = _env("RAPIDAPI_FLASHSCORE_SPORT", "tennis")

    if not api_key or not host:
        raise RapidAPIOddsError("Missing RAPIDAPI_KEY or RAPIDAPI_HOST")

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": host,
    }

    # endpoint contract differs between providers; keep params minimal and overridable
    params = {
        "sport": sport,
    }

    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    raw_events = _extract_events(r.json())

    out: List[Dict[str, Any]] = []
    for ev in raw_events:
        norm = normalize_rapidapi_event(ev)
        if norm:
            out.append(norm)
            if len(out) >= max_events:
                break
    return out
