# app.py
import os
from dotenv import load_dotenv
load_dotenv()

from odds_theoddsapi import list_active_tennis_sports, fetch_h2h_odds_for_sport, best_decimal_odds_from_event
from predictor import Predictor

MODE = os.getenv("MODE", "ATP")  # ATP/WTA (для модели)
MODEL_PATH = f"model/{MODE.lower()}_model.pkl"

# Можешь менять:
REGIONS = os.getenv("ODDS_REGIONS", "eu")  # us/uk/eu/au
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "30"))

# Дефолты метаданных, если odds API их не даёт (улучшим позже вторым API)
DEFAULT_SURFACE = os.getenv("DEFAULT_SURFACE", "Hard")
DEFAULT_LEVEL = float(os.getenv("DEFAULT_LEVEL", "1"))
DEFAULT_ROUND = float(os.getenv("DEFAULT_ROUND", "1"))
DEFAULT_BEST_OF = float(os.getenv("DEFAULT_BEST_OF", "3"))
DEFAULT_INDOOR = float(os.getenv("DEFAULT_INDOOR", "0"))

BANKROLL = float(os.getenv("BANKROLL", "1000"))
MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "0.03"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.5"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.015"))


def main():
    pred = Predictor(
        model_path=MODEL_PATH,
        bankroll=BANKROLL,
        max_stake_pct=MAX_STAKE_PCT,
        kelly_fraction_used=KELLY_FRACTION,
        min_edge=MIN_EDGE,
    )

    # 1) Находим теннисные sport keys
    sports = list_active_tennis_sports()
    print(f"Found active tennis sports: {len(sports)}")

    # Можешь отфильтровать под ATP/WTA по title/key:
    # Пока берём все tennis, но ограничим количеством запросов.
    tennis_keys = [s["key"] for s in sports if "tennis" in (s.get("key","").lower())]

    print("Tennis keys:", tennis_keys[:10])

    # 2) Тянем odds по каждому sport_key и собираем события
    events = []
    for k in tennis_keys[:4]:  # ограничение, чтобы не тратить кредиты
        try:
            data = fetch_h2h_odds_for_sport(k, regions=REGIONS, odds_format="decimal")
            for e in data:
                best = best_decimal_odds_from_event(e)
                if best:
                    events.append(best)
        except Exception as ex:
            print("Odds error for", k, ex)

    # ограничим
    events = events[:MAX_EVENTS]
    print(f"Loaded events with odds: {len(events)}")

    # 3) Прогнозируем и печатаем рекомендации
    picks = []
    known_players = 0
    total_players = 0
    for ev in events:
        A = ev["playerA"]
        B = ev["playerB"]
        oddsA = ev["oddsA"]
        oddsB = ev["oddsB"]
        dt = (ev.get("commence_time") or "")[:10] or None

        total_players += 2
        _, a_known = pred.resolve_dataset_name(A)
        _, b_known = pred.resolve_dataset_name(B)
        known_players += int(a_known)
        known_players += int(b_known)

        out = pred.predict_event(
            playerA=A,
            playerB=B,
            oddsA=oddsA,
            oddsB=oddsB,
            surface=DEFAULT_SURFACE,
            level=DEFAULT_LEVEL,
            rnd=DEFAULT_ROUND,
            best_of=DEFAULT_BEST_OF,
            indoor=DEFAULT_INDOOR,
            date_iso=dt
        )
        picks.append(out)

    picks = [x for x in picks if x.get("decision") != "SKIP_MARKET"]
    picks.sort(key=lambda x: (x.get("pick_edge") or 0.0), reverse=True)

    if total_players:
        print(f"Known players in warmed state: {known_players}/{total_players} ({100.0 * known_players / total_players:.1f}%)")

    print("\n=== TOP RECOMMENDATIONS ===")
    for p in picks[:25]:
        A = p["playerA"]
        B = p["playerB"]
        dec = p["decision"]
        prob = p["prob_A_win"]

        line = (
            f"{A} vs {B} | p(A)={prob:.3f} | "
            f"oddsA={p['oddsA']:.2f} oddsB={p['oddsB']:.2f} | "
            f"edgeA={p['edgeA']:+.3f} edgeB={p['edgeB']:+.3f} | "
            f"{dec} pick={p['pick']} stake={p['stake']} edge={p.get('pick_edge',0):+.3f}"
        )
        print(line)


if __name__ == "__main__":
    main()
