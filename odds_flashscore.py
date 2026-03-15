# odds_flashscore.py
"""
FlashScore API integration via RapidAPI.
Fetches tennis matches and odds from FlashScore.
"""
import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

FLASHSCORE_RAPIDAPI_KEY = os.getenv("FLASHSCORE_RAPIDAPI_KEY", "cadb82da62msh680c446a0f36885p10956bjsnff4d6ccad769")
FLASHSCORE_HOST = "flashscore4.p.rapidapi.com"
BASE_URL = f"https://{FLASHSCORE_HOST}"

TIMEOUT = 30


class FlashScoreAPIError(RuntimeError):
    pass


def _get(endpoint: str, params: dict = None) -> Any:
    """Make GET request to FlashScore API."""
    headers = {
        "x-rapidapi-key": FLASHSCORE_RAPIDAPI_KEY,
        "x-rapidapi-host": FLASHSCORE_HOST,
    }
    url = f"{BASE_URL}{endpoint}"

    print(f"[FlashScore] Requesting: {url} params={params}")
    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_matches_list(day: int = 0, sport_id: int = 2) -> List[Dict[str, Any]]:
    """
    Fetch matches list from FlashScore.
    day: 0 = today, -1 = yesterday, 1 = tomorrow
    sport_id: 2 = Tennis
    Returns list of tournaments with matches.
    """
    try:
        data = _get("/api/flashscore/v2/matches/list", {
            "day": day,
            "sport_id": sport_id,
            "locale": "en_INT"
        })
        return data if isinstance(data, list) else []
    except Exception as ex:
        print(f"[FlashScore] Error fetching matches list: {ex}")
        return []


def determine_sport_key(tournament_name: str) -> str:
    """
    Determine sport_key (ATP/WTA) based on tournament name.
    """
    name_lower = tournament_name.lower()

    # WTA indicators
    if any(x in name_lower for x in ["wta", "women", "ladies", "w-", "itf women"]):
        return "tennis_wta_generic"

    # ATP indicators - default for men's tennis
    if any(x in name_lower for x in ["atp", "challenger", "itf men"]):
        return "tennis_atp_generic"

    # Default to ATP
    return "tennis_atp_generic"
def is_doubles_match(player_a: str, player_b: str) -> bool:
    """Detect doubles format (e.g. "Player A / Player B")."""
    return "/" in player_a or "/" in player_b


def parse_match(match: Dict[str, Any], tournament_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse FlashScore match into unified format.
    """
    try:
        if not isinstance(match, dict):
            return None
        home_team = match.get("home_team", {})
        away_team = match.get("away_team", {})
        if not isinstance(home_team, dict) or not isinstance(away_team, dict):
            return None
        player_a = home_team.get("name", "")
        player_b = away_team.get("name", "")

        if not player_a or not player_b:
            return None

            # Skip doubles matches (team names usually contain "/")
        if is_doubles_match(player_a, player_b):
            return None
        # Get odds
        odds_data = match.get("odds", {})
        if not isinstance(odds_data, dict):
            return None
        odds_a = odds_data.get("1")
        odds_b = odds_data.get("2")

        # Must have both odds
        if odds_a is None or odds_b is None:
            return None

        try:
            odds_a = float(odds_a)
            odds_b = float(odds_b)
        except (ValueError, TypeError):
            return None

        # Skip if odds are invalid
        if odds_a <= 1.0 or odds_b <= 1.0:
            return None

        # Get match status
        match_status = match.get("match_status", {})
        if not isinstance(match_status, dict):
            match_status = {}
        is_live = match_status.get("is_in_progress", False) or match_status.get("is_started", False)
        is_finished = match_status.get("is_finished", False)

        # Get timestamp and convert to ISO
        timestamp = match.get("timestamp")
        commence_time = ""
        if timestamp:
            from datetime import datetime, timezone
            commence_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

        match_id = match.get("match_id", "")

        return {
            "playerA": player_a,
            "playerB": player_b,
            "oddsA": odds_a,
            "oddsB": odds_b,
            "commence_time": commence_time,
            "bookmakers_used": ["FlashScore"],
            "sport_key": determine_sport_key(tournament_name),
            "id": match_id,
            "source": "FlashScore",
            "tournament": tournament_name,
            "is_live": is_live,
            "is_finished": is_finished,
        }
    except Exception as ex:
        print(f"[FlashScore] Error parsing match: {ex}")
        return None




def _team_flag_is_winner(team: dict) -> bool:
    if not isinstance(team, dict):
        return False
    for key in ("winner", "is_winner", "isWinner", "won"):
        v = team.get(key)
        if isinstance(v, bool) and v:
            return True
        if isinstance(v, (int, float)) and int(v) == 1:
            return True
        if isinstance(v, str) and v.strip().lower() in {"1", "true", "yes", "win", "winner"}:
            return True
    return False


def extract_match_winner_name(match: Dict[str, Any]) -> Optional[str]:
    """Best-effort winner extraction from FlashScore match payload."""
    if not isinstance(match, dict):
        return None

    home_team = match.get("home_team", {}) if isinstance(match.get("home_team", {}), dict) else {}
    away_team = match.get("away_team", {}) if isinstance(match.get("away_team", {}), dict) else {}
    home_name = str(home_team.get("name") or "").strip()
    away_name = str(away_team.get("name") or "").strip()

    winner_raw = match.get("winner")
    if isinstance(winner_raw, str):
        w = winner_raw.strip()
        wl = w.lower()
        if wl in {"home", "1", "a", "playera"}:
            return home_name or None
        if wl in {"away", "2", "b", "playerb"}:
            return away_name or None
        if w and w in {home_name, away_name}:
            return w
    elif isinstance(winner_raw, (int, float)):
        if int(winner_raw) == 1:
            return home_name or None
        if int(winner_raw) == 2:
            return away_name or None

    if _team_flag_is_winner(home_team):
        return home_name or None
    if _team_flag_is_winner(away_team):
        return away_name or None

    score = match.get("score") if isinstance(match.get("score"), dict) else {}
    home_score = score.get("home")
    away_score = score.get("away")
    try:
        if home_score is not None and away_score is not None:
            hs = float(home_score)
            as_ = float(away_score)
            if hs > as_:
                return home_name or None
            if as_ > hs:
                return away_name or None
    except Exception:
        pass

    return None


def fetch_flashscore_finished_results(days: tuple[int, ...] = (0, -1), sport_id: int = 2) -> Dict[str, str]:
    """Fetch finished tennis match results as {event_id: winner_name}."""
    winners: Dict[str, str] = {}
    for day in days:
        tournaments = fetch_matches_list(day=day, sport_id=sport_id)
        if not isinstance(tournaments, list):
            continue

        for tournament in tournaments:
            if not isinstance(tournament, dict):
                continue
            matches = tournament.get("matches", [])
            if not isinstance(matches, list):
                continue

            for match in matches:
                if not isinstance(match, dict):
                    continue
                match_id = str(match.get("match_id") or "").strip()
                if not match_id:
                    continue

                match_status = match.get("match_status", {}) if isinstance(match.get("match_status", {}), dict) else {}
                is_finished = bool(match_status.get("is_finished", False))
                if not is_finished:
                    continue

                winner_name = extract_match_winner_name(match)
                if winner_name:
                    winners[match_id] = winner_name

    print(f"[FlashScore] Finished winners collected: {len(winners)}")
    return winners



def fetch_flashscore_finished_events(days: tuple[int, ...] = (0, -1), sport_id: int = 2) -> List[Dict[str, str]]:
    """Fetch finished tennis events with winner and participant names."""
    events: List[Dict[str, str]] = []

    for day in days:
        tournaments = fetch_matches_list(day=day, sport_id=sport_id)
        if not isinstance(tournaments, list):
            continue

        for tournament in tournaments:
            if not isinstance(tournament, dict):
                continue
            matches = tournament.get("matches", [])
            if not isinstance(matches, list):
                continue

            for match in matches:
                if not isinstance(match, dict):
                    continue

                match_status = match.get("match_status", {}) if isinstance(match.get("match_status", {}), dict) else {}
                if not bool(match_status.get("is_finished", False)):
                    continue

                home_team = match.get("home_team", {}) if isinstance(match.get("home_team", {}), dict) else {}
                away_team = match.get("away_team", {}) if isinstance(match.get("away_team", {}), dict) else {}
                player_a = str(home_team.get("name") or "").strip()
                player_b = str(away_team.get("name") or "").strip()
                if not player_a or not player_b:
                    continue
                if is_doubles_match(player_a, player_b):
                    continue

                winner_name = extract_match_winner_name(match)
                if not winner_name:
                    continue

                match_id = str(match.get("match_id") or "").strip()

                timestamp = match.get("timestamp")
                commence_time = ""
                if timestamp:
                    from datetime import datetime, timezone
                    commence_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

                events.append({
                    "event_id": match_id,
                    "playerA": player_a,
                    "playerB": player_b,
                    "winner": winner_name,
                    "commence_time": commence_time,
                })

    print(f"[FlashScore] Finished events collected: {len(events)}")
    return events

def fetch_flashscore_tennis_events(max_events: int = 30, only_prematch: bool = False) -> List[Dict[str, Any]]:
    """
    Main function to fetch tennis events from FlashScore with odds.

    Args:
        max_events: Maximum number of events to return
        only_prematch: If True, only return matches that haven't started yet

    Returns list of events in unified format.
    """
    events = []
    skipped_doubles = 0

    print(f"[FlashScore] Fetching tennis matches (max {max_events}, prematch_only={only_prematch})...")

    # Fetch today's matches (day=0)
    tournaments = fetch_matches_list(day=0, sport_id=2)
    if not isinstance(tournaments, list):
        tournaments = []
    print(f"[FlashScore] Found {len(tournaments)} tennis tournaments")

    for tournament in tournaments:
        if len(events) >= max_events:
            break
        if not isinstance(tournament, dict):
            continue

        tournament_name = tournament.get("name", "")
        matches = tournament.get("matches", [])
        if not isinstance(matches, list):
            continue
        for match in matches:
            if len(events) >= max_events:
                break

            home_team = match.get("home_team", {})
            away_team = match.get("away_team", {})
            player_a = home_team.get("name", "")
            player_b = away_team.get("name", "")
            if is_doubles_match(player_a, player_b):
                skipped_doubles += 1
                continue

            parsed = parse_match(match, tournament_name)
            if not parsed:
                continue

            # Skip finished matches
            if parsed.get("is_finished", False):
                continue

            # Skip live matches if only_prematch is enabled
            if only_prematch and parsed.get("is_live", False):
                print(f"[FlashScore] Skipping live: {parsed['playerA']} vs {parsed['playerB']}")
                continue

            events.append(parsed)
            status = "LIVE" if parsed.get("is_live") else "PREMATCH"
            print(
                f"[FlashScore] Added [{status}]: {parsed['playerA']} vs {parsed['playerB']} @ {parsed['oddsA']}/{parsed['oddsB']}")

    print(f"[FlashScore] Total events with odds: {len(events)} | skipped_doubles: {skipped_doubles}")
    return events