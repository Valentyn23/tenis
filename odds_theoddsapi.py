# odds_theoddsapi.py
import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com"

TIMEOUT = 25


class OddsAPIError(RuntimeError):
    pass


def _get(url: str, params: dict) -> Any:
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def list_active_tennis_sports() -> List[Dict[str, Any]]:
    """
    Возвращает список активных sport keys, где key содержит tennis.
    GET /v4/sports — бесплатный, не тратит квоту. :contentReference[oaicite:4]{index=4}
    """
    if not THE_ODDS_API_KEY:
        raise OddsAPIError("Missing THE_ODDS_API_KEY in env/.env")

    url = f"{BASE_URL}/v4/sports/"
    sports = _get(url, {"apiKey": THE_ODDS_API_KEY})

    tennis = []
    for s in sports:
        key = (s.get("key") or "").lower()
        title = (s.get("title") or "").lower()
        group = (s.get("group") or "").lower()
        if "tennis" in key or "tennis" in title or "tennis" in group:
            if s.get("active") is True:
                tennis.append(s)
    return tennis


def fetch_h2h_odds_for_sport(
    sport_key: str,
    regions: str = "eu",
    odds_format: str = "decimal",
    date_format: str = "iso",
) -> List[Dict[str, Any]]:
    """
    GET /v4/sports/{sport_key}/odds?regions=...&markets=h2h&oddsFormat=decimal :contentReference[oaicite:5]{index=5}
    Возвращает список events с букмекерами.
    """
    if not THE_ODDS_API_KEY:
        raise OddsAPIError("Missing THE_ODDS_API_KEY in env/.env")

    url = f"{BASE_URL}/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    return _get(url, params)


def best_decimal_odds_from_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Вытаскивает best odds (максимальные) по игрокам из всех букмекеров.
    Возвращает:
      {
        "playerA": "...",
        "playerB": "...",
        "oddsA": 1.87,
        "oddsB": 2.05,
        "commence_time": "...",
        "bookmakers_used": ["Pinnacle", ...]
      }
    """
    home = event.get("home_team")
    away = event.get("away_team")
    if not home or not away:
        return None

    best = {home: None, away: None}
    books_used = {home: None, away: None}

    for bm in event.get("bookmakers", []) or []:
        title = bm.get("title") or bm.get("key") or "book"
        for m in bm.get("markets", []) or []:
            if m.get("key") != "h2h":
                continue
            for out in m.get("outcomes", []) or []:
                name = out.get("name")
                price = out.get("price")
                if name in best and isinstance(price, (int, float)):
                    if best[name] is None or price > best[name]:
                        best[name] = float(price)
                        books_used[name] = title

    if best[home] is None or best[away] is None:
        return None

    return {
        "playerA": home,
        "playerB": away,
        "oddsA": best[home],
        "oddsB": best[away],
        "commence_time": event.get("commence_time"),
        "bookmakers_used": [books_used[home], books_used[away]],
        "sport_key": event.get("sport_key"),
        "id": event.get("id"),
    }
