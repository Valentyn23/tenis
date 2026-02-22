"""
Модуль для автоматического обновления моделей и состояний при запуске приложения.

При каждом запуске:
1. Подтягивает новые результаты матчей из Tennis API
2. Нормализует данные в формат обучающих файлов
3. Обновляет StateEngine новыми матчами
4. Сохраняет обновленные стейты
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state_engine import StateEngine
from state_persistence import save_engine, load_engine
from api_data_fetcher import fetch_new_matches, matches_to_dataframe
from config_shared import tournament_level_from_text

# =========================================================
# CONFIG
# =========================================================
STATE_PATH_ATP = os.getenv("STATE_PATH_ATP", "state/engine_state_atp.pkl")
STATE_PATH_WTA = os.getenv("STATE_PATH_WTA", "state/engine_state_wta.pkl")

ATP_LAST_DATE_FILE = "state/.atp_last_date"
WTA_LAST_DATE_FILE = "state/.wta_last_date"

DEFAULT_ATP_LAST_DATE = "2026-02-15"
DEFAULT_WTA_LAST_DATE = "2026-02-14"

AUTO_UPDATE_ENABLED = os.getenv("AUTO_UPDATE_ENABLED", "1") == "1"
MIN_NEW_MATCHES = int(os.getenv("MIN_NEW_MATCHES", "1"))


# =========================================================
# HELPERS
# =========================================================
def get_last_date(mode: str) -> str:
    """Получить последнюю дату из файла или вернуть дефолт."""
    file_path = ATP_LAST_DATE_FILE if mode.upper() == "ATP" else WTA_LAST_DATE_FILE
    default = DEFAULT_ATP_LAST_DATE if mode.upper() == "ATP" else DEFAULT_WTA_LAST_DATE

    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                date = f.read().strip()
                if date:
                    return date
    except Exception as e:
        print(f"[WARN] Could not read last date file: {e}")

    return default


def save_last_date(mode: str, date: str) -> None:
    """Сохранить последнюю дату в файл."""
    file_path = ATP_LAST_DATE_FILE if mode.upper() == "ATP" else WTA_LAST_DATE_FILE

    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(date)
        print(f"[{mode}] Saved last date: {date}")
    except Exception as e:
        print(f"[WARN] Could not save last date: {e}")


def parse_round(round_text: str) -> float:
    """Конвертировать текст раунда в число."""
    r = str(round_text).strip().lower()

    if "128" in r or "1st" in r:
        return 1.0
    if "64" in r or "2nd" in r:
        return 1.5
    if "32" in r or "3rd" in r:
        return 2.0
    if "16" in r or "4th" in r:
        return 3.0
    if "quarter" in r or "qf" in r:
        return 4.0
    if "semi" in r or "sf" in r:
        return 5.0
    if "final" in r:
        return 6.0

    return 1.0


def parse_surface(surface: str) -> str:
    """Нормализовать поверхность."""
    s = str(surface).strip().lower()

    if "hard" in s:
        return "Hard"
    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"
    if "carpet" in s:
        return "Carpet"

    return "Hard"


# =========================================================
# STATE UPDATE
# =========================================================
def update_state_with_matches(engine: StateEngine, matches: List[dict], mode: str) -> int:
    """Обновить StateEngine новыми матчами."""
    if not matches:
        return 0

    matches_sorted = sorted(matches, key=lambda x: x.get("Date", ""))

    updated = 0
    for match in matches_sorted:
        try:
            winner = match.get("Winner")
            loser = match.get("Loser")
            surface = parse_surface(match.get("Surface", "Hard"))
            date_str = match.get("Date", "")

            if not winner or not loser or not date_str:
                continue

            series_key = "Series" if mode.upper() == "ATP" else "Tier"
            level = tournament_level_from_text(match.get(series_key, ""), default=1.0)

            rnd = parse_round(match.get("Round", "1st Round"))

            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                try:
                    date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                except:
                    continue

            engine.update_after_match(
                winner=winner,
                loser=loser,
                surface=surface,
                level=level,
                rnd=rnd,
                date=date,
            )

            updated += 1

        except Exception as e:
            print(f"[WARN] Error processing match: {e}")
            continue

    return updated


def update_mode(mode: str) -> Tuple[int, Optional[str]]:
    """Обновить состояние для указанного режима (ATP или WTA)."""
    state_path = STATE_PATH_ATP if mode.upper() == "ATP" else STATE_PATH_WTA

    engine = load_engine(state_path)
    if engine is None:
        print(f"[{mode}] State not found at {state_path}, creating new engine...")
        engine = StateEngine(mode=mode)

    last_date = get_last_date(mode)
    print(f"[{mode}] Last training date: {last_date}")

    try:
        new_matches = fetch_new_matches(mode, last_date)
    except Exception as e:
        print(f"[{mode}] Error fetching new matches: {e}")
        return 0, None

    if not new_matches:
        print(f"[{mode}] No new matches found")
        return 0, None

    print(f"[{mode}] Found {len(new_matches)} new matches")

    updated_count = update_state_with_matches(engine, new_matches, mode)
    print(f"[{mode}] Updated state with {updated_count} matches")

    if updated_count >= MIN_NEW_MATCHES:
        save_engine(engine, state_path)
        print(f"[{mode}] State saved to {state_path}")

        latest_date = max(m.get("Date", "") for m in new_matches if m.get("Date"))
        if latest_date:
            save_last_date(mode, latest_date)
            return updated_count, latest_date

    return updated_count, None


# =========================================================
# MAIN FUNCTION
# =========================================================
def auto_update_on_startup() -> dict:
    """Выполнить автоматическое обновление при запуске."""
    result = {
        "enabled": AUTO_UPDATE_ENABLED,
        "atp": {"matches_updated": 0, "latest_date": None},
        "wta": {"matches_updated": 0, "latest_date": None},
    }

    if not AUTO_UPDATE_ENABLED:
        print("[AUTO_UPDATE] Disabled via AUTO_UPDATE_ENABLED=0")
        return result

    print("\n" + "=" * 60)
    print("AUTO UPDATE: Checking for new match results...")
    print("=" * 60)

    try:
        atp_count, atp_date = update_mode("ATP")
        result["atp"]["matches_updated"] = atp_count
        result["atp"]["latest_date"] = atp_date
    except Exception as e:
        print(f"[ATP] Update error: {e}")

    try:
        wta_count, wta_date = update_mode("WTA")
        result["wta"]["matches_updated"] = wta_count
        result["wta"]["latest_date"] = wta_date
    except Exception as e:
        print(f"[WTA] Update error: {e}")

    print("\n" + "=" * 60)
    print(f"AUTO UPDATE COMPLETE:")
    print(f"  ATP: {result['atp']['matches_updated']} matches updated")
    print(f"  WTA: {result['wta']['matches_updated']} matches updated")
    print("=" * 60 + "\n")

    return result


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def get_state_info(mode: str) -> dict:
    """Получить информацию о текущем состоянии."""
    state_path = STATE_PATH_ATP if mode.upper() == "ATP" else STATE_PATH_WTA

    engine = load_engine(state_path)
    if engine is None:
        return {"exists": False, "path": state_path}

    total_matches = sum(st.matches for st in engine.players.values()) / 2.0

    return {
        "exists": True,
        "path": state_path,
        "players": len(engine.players),
        "estimated_matches": int(total_matches),
        "mode": getattr(engine, "mode", mode),
        "last_date": get_last_date(mode),
    }


def force_update(mode: str = None) -> dict:
    """Принудительно обновить состояние."""
    global AUTO_UPDATE_ENABLED
    original = AUTO_UPDATE_ENABLED
    AUTO_UPDATE_ENABLED = True

    try:
        if mode:
            count, date = update_mode(mode)
            return {mode.lower(): {"matches_updated": count, "latest_date": date}}
        else:
            return auto_update_on_startup()
    finally:
        AUTO_UPDATE_ENABLED = original


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-update tennis prediction models")
    parser.add_argument("--mode", choices=["ATP", "WTA", "both"], default="both")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.info:
        print("\n=== STATE INFO ===")
        for m in ["ATP", "WTA"]:
            info = get_state_info(m)
            print(f"\n{m}:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        sys.exit(0)

    if args.force or AUTO_UPDATE_ENABLED:
        if args.mode == "both":
            result = force_update() if args.force else auto_update_on_startup()
        else:
            count, date = update_mode(args.mode)
            result = {args.mode.lower(): {"matches_updated": count, "latest_date": date}}

        print("\nResult:", result)
    else:
        print("Auto-update is disabled. Use --force to override.")