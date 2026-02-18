# app.py
import os
from dotenv import load_dotenv

load_dotenv()

from odds_theoddsapi import list_active_tennis_sports, fetch_h2h_odds_for_sport, best_decimal_odds_from_event
from predictor import Predictor
from config_shared import infer_level_from_sport_key, infer_mode_from_sport_key

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
MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "0.02"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.35"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.03"))


def make_predictor(mode: str) -> Predictor:
    state_path = os.getenv(f"STATE_PATH_{mode}", f"state/engine_state_{mode.lower()}.pkl")
    return Predictor(
        model_path=f"model/{mode.lower()}_model.pkl",
        state_path=state_path,
        bankroll=BANKROLL,
        max_stake_pct=MAX_STAKE_PCT,
        kelly_fraction_used=KELLY_FRACTION,
        min_edge=MIN_EDGE,
        prob_floor=float(os.getenv("PROB_FLOOR", "0.08")),
        prob_ceil=float(os.getenv("PROB_CEIL", "0.92")),
        max_overround=float(os.getenv("MAX_OVERROUND", "1.06")),
        strict_mode_match=(os.getenv("STRICT_MODE_MATCH", "1") == "1"),
    )


def main():
    # Two predictors for mixed ATP/WTA feed
    predictors = {
        "ATP": make_predictor("ATP"),
        "WTA": make_predictor("WTA"),
    }

    # 1) Находим теннисные sport keys
    sports = list_active_tennis_sports()
    print(f"Found active tennis sports: {len(sports)}")

    tennis_keys = [s["key"] for s in sports if "tennis" in (s.get("key", "").lower())]
    print("Tennis keys:", tennis_keys[:10])

    # 2) Тянем odds по каждому sport_key и собираем события
    events = []
    for k in tennis_keys[:8]:
        try:
            data = fetch_h2h_odds_for_sport(k, regions=REGIONS, odds_format="decimal")
            for e in data:
                best = best_decimal_odds_from_event(e)
                if best:
                    events.append(best)
        except Exception as ex:
            print("Odds error for", k, ex)

    events = events[:MAX_EVENTS]
    print(f"Loaded events with odds: {len(events)}")

    # 3) Прогнозируем и печатаем рекомендации
    picks = []
    known_players = 0
    total_players = 0
    fallback_level_count = 0
    skipped_unknown_tour = 0
    used_by_mode = {"ATP": 0, "WTA": 0}
    decision_counts = {"BET_A": 0, "BET_B": 0, "NO_BET": 0, "SKIP_MARKET": 0, "SKIP_UNKNOWN_PLAYERS": 0}
    market_skip_reasons = {}
    cap_stake_hits = 0

    for ev in events:
        A = ev["playerA"]
        B = ev["playerB"]
        oddsA = ev["oddsA"]
        oddsB = ev["oddsB"]
        dt = (ev.get("commence_time") or "")[:10] or None

        event_mode = infer_mode_from_sport_key(ev.get("sport_key"))
        if event_mode not in predictors:
            skipped_unknown_tour += 1
            continue

        pred = predictors[event_mode]
        used_by_mode[event_mode] += 1

        total_players += 2
        _, a_known = pred.resolve_dataset_name(A)
        _, b_known = pred.resolve_dataset_name(B)
        known_players += int(a_known)
        known_players += int(b_known)

        level_for_event, used_level_fallback = infer_level_from_sport_key(ev.get("sport_key"), default=DEFAULT_LEVEL)
        fallback_level_count += int(used_level_fallback)

        out = pred.predict_event(
            playerA=A,
            playerB=B,
            oddsA=oddsA,
            oddsB=oddsB,
            surface=DEFAULT_SURFACE,
            level=level_for_event,
            rnd=DEFAULT_ROUND,
            best_of=DEFAULT_BEST_OF,
            indoor=DEFAULT_INDOOR,
            date_iso=dt,
        )
        out["tour_mode"] = event_mode

        dec = out.get("decision", "NO_BET")
        if dec in decision_counts:
            decision_counts[dec] += 1

        if dec == "SKIP_MARKET":
            reason = out.get("reason", "unknown")
            market_skip_reasons[reason] = market_skip_reasons.get(reason, 0) + 1

        if dec in ("BET_A", "BET_B") and float(out.get("stake", 0.0)) >= (BANKROLL * MAX_STAKE_PCT - 1e-9):
            cap_stake_hits += 1

        picks.append(out)

    picks = [x for x in picks if x.get("decision") != "SKIP_MARKET"]
    picks.sort(key=lambda x: (x.get("pick_edge") or 0.0), reverse=True)

    processed_events = max(1, sum(used_by_mode.values()))
    print(f"Predictors used by tour: ATP={used_by_mode['ATP']} WTA={used_by_mode['WTA']}")
    if skipped_unknown_tour:
        print(f"Skipped events due to unknown tour mode: {skipped_unknown_tour}")

    if total_players:
        print(f"Known players in warmed state: {known_players}/{total_players} ({100.0 * known_players / total_players:.1f}%)")

    if events:
        print(f"Fallback tournament level used: {fallback_level_count}/{processed_events}")

    print(
        "Decision stats:",
        f"BET_A={decision_counts['BET_A']}",
        f"BET_B={decision_counts['BET_B']}",
        f"NO_BET={decision_counts['NO_BET']}",
        f"SKIP_MARKET={decision_counts['SKIP_MARKET']}",
        f"SKIP_UNKNOWN_PLAYERS={decision_counts['SKIP_UNKNOWN_PLAYERS']}",
    )
    if market_skip_reasons:
        print("Market skip reasons:", market_skip_reasons)

    bet_count = decision_counts["BET_A"] + decision_counts["BET_B"]
    if bet_count:
        print(f"Cap stake hits: {cap_stake_hits}/{bet_count}")

    print("\n=== TOP RECOMMENDATIONS ===")
    for p in picks[:25]:
        A = p["playerA"]
        B = p["playerB"]
        dec = p["decision"]
        prob = p["prob_A_win"]

        line = (
            f"[{p.get('tour_mode', '?')}] {A} vs {B} | p(A)={prob:.3f} | "
            f"oddsA={p['oddsA']:.2f} oddsB={p['oddsB']:.2f} | "
            f"edgeA={p['edgeA']:+.3f} edgeB={p['edgeB']:+.3f} | "
            f"{dec} pick={p['pick']} stake={p['stake']} edge={p.get('pick_edge', 0):+.3f}"
        )
        print(line)


if __name__ == "__main__":
    main()
