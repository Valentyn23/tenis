"""
Модуль для подтягивания последних результатов матчей из Tennis API RapidAPI
и нормализации данных для дообучения моделей.

Формат данных должен соответствовать Excel файлам:
ATP: ATP, Location, Tournament, Date, Series, Court, Surface, Round, Best of,
     Winner, Loser, WRank, LRank, WPts, LPts, W1, L1, W2, L2, W3, L3, W4, L4, W5, L5,
     Wsets, Lsets, Comment, B365W, B365L, PSW, PSL, MaxW, MaxL, AvgW, AvgL, BFEW, BFEL

WTA: WTA, Location, Tournament, Date, Tier, Court, Surface, Round, Best of,
     Winner, Loser, WRank, LRank, WPts, LPts, W1, L1, W2, L2, W3, L3,
     Wsets, Lsets, Comment, B365W, B365L, PSW, PSL, MaxW, MaxL, AvgW, AvgL, BFEW, BFEL
"""

import os
import re
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", os.getenv("FLASHSCORE_RAPIDAPI_KEY", ""))
RAPIDAPI_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
BASE_URL = f"https://{RAPIDAPI_HOST}/tennis/v2"

# Последние даты в базе данных (обучающих файлах)
# ATP: 15.02.2026 -> нужны с 16.02.2026
# WTA: 14.02.2026 -> нужны с 15.02.2026
ATP_LAST_DATE = os.getenv("ATP_LAST_DATE", "2026-02-15")
WTA_LAST_DATE = os.getenv("WTA_LAST_DATE", "2026-02-14")


# =========================================================
# API HELPERS
# =========================================================
def get_headers() -> dict:
    return {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }


def api_get(endpoint: str, params: dict = None) -> Optional[dict]:
    """Выполнить GET запрос к API."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, headers=get_headers(), params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"[API ERROR] {endpoint}: {e}")
        return None


# =========================================================
# TOURNAMENT CALENDAR & RESULTS
# =========================================================
def get_tournament_calendar(mode: str, year: int = 2026) -> List[dict]:
    """Получить календарь турниров на год."""
    endpoint = f"{mode.lower()}/tournament/calendar/{year}"
    data = api_get(endpoint)
    if data and "data" in data:
        return data["data"]
    return []


def get_tournament_info(mode: str, tournament_id: int) -> Optional[dict]:
    """Получить информацию о турнире."""
    endpoint = f"{mode.lower()}/tournament/info/{tournament_id}"
    data = api_get(endpoint)
    if data and "data" in data:
        return data["data"]
    return None


def get_tournament_results(mode: str, tournament_id: int) -> List[dict]:
    """Получить результаты турнира (одиночные матчи)."""
    endpoint = f"{mode.lower()}/tournament/results/{tournament_id}"
    data = api_get(endpoint)
    if data and "data" in data and "singles" in data["data"]:
        return data["data"]["singles"]
    return []


def get_fixtures_by_date(mode: str, date: str) -> List[dict]:
    """Получить все матчи на определенную дату."""
    endpoint = f"{mode.lower()}/fixtures"
    data = api_get(endpoint, params={"date": date})
    if data and "data" in data:
        # Фильтруем только одиночные (не doubles)
        return [m for m in data["data"] if "/" not in m.get("player1", {}).get("name", "")]
    return []


# =========================================================
# NAME NORMALIZATION (API -> Dataset format)
# =========================================================
def normalize_player_name(api_name: str) -> str:
    """Улучшенная нормализация с учетом приставок."""
    if not api_name:
        return api_name

    name = str(api_name).strip()
    name = re.sub(r'\s*\([^)]+\)\s*', '', name)  # Убираем (Q), (WC)

    parts = [p for p in name.split() if p]
    if len(parts) == 1:
        return parts[0]

    # Приставки которые являются частью фамилии
    prefixes = {'de', 'da', 'van', 'von', 'del', 'della', 'di', 'la', 'le'}

    first_name = parts[0]

    # Ищем где начинается фамилия
    surname_start = 1
    surname_parts = []

    for i in range(1, len(parts)):
        if parts[i].lower() in prefixes:
            surname_start = i
            break

    # Фамилия = всё начиная с surname_start
    surname = ' '.join(parts[surname_start:])
    initial = first_name[0].upper()

    return f"{surname} {initial}."


# =========================================================
# ROUND MAPPING
# =========================================================
def round_id_to_text(round_id: int) -> str:
    """Конвертировать roundId из API в текстовое значение."""
    if round_id <= 4:
        return "1st Round"
    elif round_id == 5:
        return "2nd Round"
    elif round_id == 6:
        return "3rd Round"
    elif round_id == 7:
        return "4th Round"
    elif round_id in (8, 9):
        return "Quarterfinals"
    elif round_id in (10, 11):
        return "Semifinals"
    elif round_id >= 12:
        return "The Final"
    return "1st Round"


# =========================================================
# SERIES/TIER MAPPING
# =========================================================
def get_series_from_rank_id(rank_id: int, mode: str) -> str:
    """
    Определить Series (ATP) или Tier (WTA) по rankId турнира.
    """
    if mode.upper() == "ATP":
        if rank_id == 4:
            return "Grand Slam"
        elif rank_id == 3:
            return "Masters 1000"
        elif rank_id == 7:
            return "Masters"
        elif rank_id == 2:
            return "ATP250"
        return "ATP250"
    else:  # WTA
        if rank_id == 4:
            return "Grand Slam"
        elif rank_id == 3:
            return "WTA1000"
        elif rank_id == 7:
            return "WTA Finals"
        elif rank_id == 2:
            return "WTA250"
        return "WTA250"


# =========================================================
# COURT/SURFACE MAPPING
# =========================================================
def get_court_and_surface(court_data: dict) -> tuple:
    """Определить Court (Indoor/Outdoor) и Surface (Hard/Clay/Grass)."""
    if not court_data:
        return "Outdoor", "Hard"

    court_name = court_data.get("name", "Hard").lower()

    if "i.hard" in court_name or "indoor" in court_name:
        return "Indoor", "Hard"
    elif "hard" in court_name:
        return "Outdoor", "Hard"
    elif "clay" in court_name:
        return "Outdoor", "Clay"
    elif "grass" in court_name:
        return "Outdoor", "Grass"
    elif "carpet" in court_name:
        return "Indoor", "Carpet"

    return "Outdoor", "Hard"


# =========================================================
# SCORE PARSING
# =========================================================
def parse_score(result: str) -> dict:
    """
    Парсинг счета матча.
    Пример: "6-4 6-3" или "7-6(5) 3-6 7-5" или "6-4 6-3 ret."
    """
    out = {
        "W1": None, "L1": None,
        "W2": None, "L2": None,
        "W3": None, "L3": None,
        "W4": None, "L4": None,
        "W5": None, "L5": None,
        "Wsets": 0, "Lsets": 0,
        "Comment": "Completed"
    }

    if not result:
        return out

    result = str(result).strip()

    # Проверяем на retired/walkover
    if "ret" in result.lower() or "retired" in result.lower():
        out["Comment"] = "Retired"
    elif "w/o" in result.lower() or "walkover" in result.lower():
        out["Comment"] = "Walkover"
    elif "def" in result.lower() or "default" in result.lower():
        out["Comment"] = "Default"

    # Парсим сеты
    set_pattern = r'(\d+)-(\d+)(?:\((\d+)\))?'
    sets = re.findall(set_pattern, result)

    wsets = 0
    lsets = 0

    for i, (w_games, l_games, _tb) in enumerate(sets[:5], 1):
        w = int(w_games)
        l = int(l_games)
        out[f"W{i}"] = float(w)
        out[f"L{i}"] = float(l)

        if w > l:
            wsets += 1
        else:
            lsets += 1

    out["Wsets"] = float(wsets)
    out["Lsets"] = float(lsets)

    return out


# =========================================================
# MATCH NORMALIZATION
# =========================================================
def normalize_match_data(
        match: dict,
        tournament_info: dict,
        mode: str,
        rankings: Dict[int, dict] = None
) -> Optional[dict]:
    """Преобразовать данные матча из API в формат датасета."""
    if not match or not tournament_info:
        return None

    # Пропускаем doubles
    p1_name = match.get("player1", {}).get("name", "")
    p2_name = match.get("player2", {}).get("name", "")
    if "/" in p1_name or "/" in p2_name:
        return None

    # Определяем победителя
    winner_id = match.get("match_winner")
    p1_id = match.get("player1Id")
    p2_id = match.get("player2Id")

    if winner_id == p1_id:
        winner_name = p1_name
        loser_name = p2_name
        winner_api_id = p1_id
        loser_api_id = p2_id
    elif winner_id == p2_id:
        winner_name = p2_name
        loser_name = p1_name
        winner_api_id = p2_id
        loser_api_id = p1_id
    else:
        return None  # Матч не завершен

    # Нормализуем имена игроков
    winner_ds = normalize_player_name(winner_name)
    loser_ds = normalize_player_name(loser_name)

    # Дата
    date_str = match.get("date", "")
    if date_str:
        date_str = date_str[:10]

    # Турнирная информация
    tournament_name = tournament_info.get("name", "")
    location = tournament_name.split(" - ")[-1] if " - " in tournament_name else tournament_name

    court_data = tournament_info.get("court", {})
    court, surface = get_court_and_surface(court_data)

    rank_id = tournament_info.get("rankId", 2)
    series_tier = get_series_from_rank_id(rank_id, mode)

    # Раунд
    round_id = match.get("roundId", 1)
    round_text = round_id_to_text(round_id)

    # Счет
    result = match.get("result", "")
    score_data = parse_score(result)

    # Best of
    best_of = 3
    if rank_id == 4 and mode.upper() == "ATP":
        best_of = 5

    # Рейтинги игроков (если есть)
    w_rank, w_pts = None, None
    l_rank, l_pts = None, None
    if rankings:
        w_info = rankings.get(winner_api_id, {})
        l_info = rankings.get(loser_api_id, {})
        w_rank = w_info.get("rank")
        w_pts = w_info.get("points")
        l_rank = l_info.get("rank")
        l_pts = l_info.get("points")

    # Формируем запись
    if mode.upper() == "ATP":
        record = {
            "ATP": 1,
            "Location": location,
            "Tournament": tournament_name,
            "Date": date_str,
            "Series": series_tier,
            "Court": court,
            "Surface": surface,
            "Round": round_text,
            "Best of": best_of,
            "Winner": winner_ds,
            "Loser": loser_ds,
            "WRank": w_rank,
            "LRank": l_rank,
            "WPts": w_pts,
            "LPts": l_pts,
            **{k: score_data[k] for k in
               ["W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5", "Wsets", "Lsets", "Comment"]},
            "B365W": None, "B365L": None,
            "PSW": None, "PSL": None,
            "MaxW": None, "MaxL": None,
            "AvgW": None, "AvgL": None,
            "BFEW": None, "BFEL": None,
        }
    else:  # WTA
        record = {
            "WTA": 1,
            "Location": location,
            "Tournament": tournament_name,
            "Date": date_str,
            "Tier": series_tier,
            "Court": court,
            "Surface": surface,
            "Round": round_text,
            "Best of": best_of,
            "Winner": winner_ds,
            "Loser": loser_ds,
            "WRank": w_rank,
            "LRank": l_rank,
            "WPts": w_pts,
            "LPts": l_pts,
            **{k: score_data[k] for k in ["W1", "L1", "W2", "L2", "W3", "L3", "Wsets", "Lsets", "Comment"]},
            "B365W": None, "B365L": None,
            "PSW": None, "PSL": None,
            "MaxW": None, "MaxL": None,
            "AvgW": None, "AvgL": None,
            "BFEW": None, "BFEL": None,
        }

    return record


# =========================================================
# KNOWN TOURNAMENT IDS
# =========================================================
ATP_TOURNAMENT_IDS_2026 = [
    21296, 21297, 21298, 21299, 21300, 21301, 21302, 21303,
    21304, 21305, 21306, 21307, 21308, 21309, 21310, 21311,
    21312, 21313, 21314, 21315,
]

WTA_TOURNAMENT_IDS_2026 = [
    16700, 16701, 16702, 16703, 16704, 16705, 16706, 16707,
    16708, 16709, 16710, 16711, 16712, 16713,
]


# =========================================================
# MAIN FETCH FUNCTION
# =========================================================
def fetch_new_matches(mode: str, from_date: str) -> List[dict]:
    """
    Получить все новые матчи начиная с указанной даты.

    Args:
        mode: "ATP" или "WTA"
        from_date: Дата в формате "YYYY-MM-DD" (исключительно)

    Returns:
        Список нормализованных записей матчей
    """
    if not RAPIDAPI_KEY:
        print("[ERROR] RAPIDAPI_KEY не установлен")
        return []

    print(f"[{mode}] Fetching new matches from {from_date}...")

    # Собираем уникальные ID турниров через fixtures за последние дни
    tournament_ids = set()
    from_dt = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.now()

    # Идем по датам, проверяя fixtures
    check_date = from_dt + timedelta(days=1)
    days_checked = 0
    max_days = 14

    while check_date <= end_date and days_checked < max_days:
        date_str = check_date.strftime("%Y-%m-%d")
        fixtures = get_fixtures_by_date(mode, date_str)
        for f in fixtures:
            tid = f.get("tournamentId")
            if tid:
                tournament_ids.add(tid)
        check_date += timedelta(days=1)
        days_checked += 1

    # Добавляем известные ID турниров
    known_ids = ATP_TOURNAMENT_IDS_2026 if mode.upper() == "ATP" else WTA_TOURNAMENT_IDS_2026
    for tid in known_ids:
        tournament_ids.add(tid)

    print(f"[{mode}] Found {len(tournament_ids)} tournaments to check")

    all_matches = []
    processed_match_ids = set()

    for tid in sorted(tournament_ids):
        tournament_info = get_tournament_info(mode, tid)
        if not tournament_info:
            continue

        tournament_name = tournament_info.get('name', str(tid))

        results = get_tournament_results(mode, tid)
        if not results:
            continue

        new_matches_count = 0
        for match in results:
            match_date = match.get("date", "")[:10] if match.get("date") else ""
            if match_date and match_date > from_date:
                match_id = match.get("id")
                if match_id not in processed_match_ids:
                    processed_match_ids.add(match_id)

                    normalized = normalize_match_data(match, tournament_info, mode)
                    if normalized:
                        all_matches.append(normalized)
                        new_matches_count += 1

        if new_matches_count > 0:
            print(f"[{mode}] {tournament_name}: {new_matches_count} new matches")

    all_matches.sort(key=lambda x: x.get("Date", ""))

    print(f"[{mode}] Total new matches found: {len(all_matches)}")
    return all_matches


def fetch_all_new_matches() -> tuple:
    """Получить все новые матчи для ATP и WTA."""
    atp_matches = fetch_new_matches("ATP", ATP_LAST_DATE)
    wta_matches = fetch_new_matches("WTA", WTA_LAST_DATE)
    return atp_matches, wta_matches


# =========================================================
# DATAFRAME CONVERSION
# =========================================================
def matches_to_dataframe(matches: List[dict], mode: str) -> pd.DataFrame:
    """Преобразовать список матчей в DataFrame."""
    if not matches:
        return pd.DataFrame()

    if mode.upper() == "ATP":
        columns = [
            "ATP", "Location", "Tournament", "Date", "Series", "Court", "Surface",
            "Round", "Best of", "Winner", "Loser", "WRank", "LRank", "WPts", "LPts",
            "W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5",
            "Wsets", "Lsets", "Comment",
            "B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL", "BFEW", "BFEL"
        ]
    else:
        columns = [
            "WTA", "Location", "Tournament", "Date", "Tier", "Court", "Surface",
            "Round", "Best of", "Winner", "Loser", "WRank", "LRank", "WPts", "LPts",
            "W1", "L1", "W2", "L2", "W3", "L3",
            "Wsets", "Lsets", "Comment",
            "B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL", "BFEW", "BFEL"
        ]

    df = pd.DataFrame(matches)
    existing_cols = [c for c in columns if c in df.columns]
    df = df[existing_cols]

    return df


if __name__ == "__main__":
    print("Testing API Data Fetcher...")
    atp_matches, wta_matches = fetch_all_new_matches()
    print(f"\nTotal ATP: {len(atp_matches)}")
    print(f"Total WTA: {len(wta_matches)}")