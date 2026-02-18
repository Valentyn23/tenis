# app.py
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - runtime guard
    def load_dotenv():
        return None

load_dotenv()

from odds_theoddsapi import list_active_tennis_sports, fetch_h2h_odds_for_sport, best_decimal_odds_from_event
from predictor import Predictor
from config_shared import infer_level_from_sport_key, infer_mode_from_sport_key
from settings import load_runtime_settings

SETTINGS = load_runtime_settings()


def make_predictor(mode: str) -> Optional[Predictor]:
    state_path = SETTINGS.state_path_atp if mode == "ATP" else SETTINGS.state_path_wta
    try:
        return Predictor(
            model_path=f"model/{mode.lower()}_model.pkl",
            state_path=state_path,
            bankroll=SETTINGS.bankroll,
            max_stake_pct=SETTINGS.max_stake_pct,
            kelly_fraction_used=SETTINGS.kelly_fraction,
            min_edge=SETTINGS.min_edge,
            prob_floor=SETTINGS.prob_floor,
            prob_ceil=SETTINGS.prob_ceil,
            max_overround=SETTINGS.max_overround,
            strict_mode_match=SETTINGS.strict_mode_match,
            soft_cap_edge=SETTINGS.soft_cap_edge,
            soft_cap_factor=SETTINGS.soft_cap_factor,
        )
    except RuntimeError as exc:
        print(f"[WARN] Predictor {mode} unavailable: {exc}")
        print(
            f"[WARN] Create state for {mode}: MODE={mode} python wrump.py "
            f"(or set STATE_PATH_{mode})"
        )
        return None


def validate_artifacts(required_modes: set[str]) -> dict[str, str]:
    status = {}
    for mode in sorted(required_modes):
        model_path = Path(f"model/{mode.lower()}_model.pkl")
        state_path = Path(SETTINGS.state_path_atp if mode == "ATP" else SETTINGS.state_path_wta)
        if not model_path.exists():
            status[mode] = f"MISSING_MODEL:{model_path}"
            continue
        if not state_path.exists():
            status[mode] = f"MISSING_STATE:{state_path}"
            continue
        status[mode] = "OK"
    return status


def save_session_report(report: dict) -> Optional[Path]:
    if not SETTINGS.save_report:
        return None
    report_dir = Path(SETTINGS.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"session_{ts}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return path


def main():
    print(f"Runtime settings: {SETTINGS.summary()}")

    sports = list_active_tennis_sports()
    print(f"Found active tennis sports: {len(sports)}")

    tennis_keys = [s["key"] for s in sports if "tennis" in (s.get("key", "").lower())]
    print("Tennis keys:", tennis_keys[:10])

    required_modes = {m for m in (infer_mode_from_sport_key(k) for k in tennis_keys) if m in ("ATP", "WTA")}
    preflight = validate_artifacts(required_modes)
    print("Preflight:", preflight)

    predictors = {}
    for mode in sorted(required_modes):
        pred = make_predictor(mode)
        if pred is not None:
            predictors[mode] = pred

    events = []
    for k in tennis_keys[:8]:
        try:
            data = fetch_h2h_odds_for_sport(k, regions=SETTINGS.regions, odds_format="decimal")
            for e in data:
                best = best_decimal_odds_from_event(e)
                if best:
                    events.append(best)
        except Exception as ex:
            print("Odds error for", k, ex)

    events = events[: SETTINGS.max_events]
    print(f"Loaded events with odds: {len(events)}")

    if not predictors:
        print("No predictors available. Warm up states and retry.")
        return

    picks = []
    known_players = 0
    total_players = 0
    fallback_level_count = 0
    skipped_unknown_tour = 0
    used_by_mode = {"ATP": 0, "WTA": 0}
    decision_counts = {"BET_A": 0, "BET_B": 0, "NO_BET": 0, "SKIP_MARKET": 0, "SKIP_UNKNOWN_PLAYERS": 0}
    market_skip_reasons: dict[str, int] = {}
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

        level_for_event, used_level_fallback = infer_level_from_sport_key(ev.get("sport_key"), default=SETTINGS.default_level)
        fallback_level_count += int(used_level_fallback)

        out = pred.predict_event(
            playerA=A,
            playerB=B,
            oddsA=oddsA,
            oddsB=oddsB,
            surface=SETTINGS.default_surface,
            level=level_for_event,
            rnd=SETTINGS.default_round,
            best_of=SETTINGS.default_best_of,
            indoor=SETTINGS.default_indoor,
            date_iso=dt,
        )
        out["tour_mode"] = event_mode

        dec = out.get("decision", "NO_BET")
        if dec in decision_counts:
            decision_counts[dec] += 1

        if dec == "SKIP_MARKET":
            reason = out.get("reason", "unknown")
            market_skip_reasons[reason] = market_skip_reasons.get(reason, 0) + 1

        if dec in ("BET_A", "BET_B") and float(out.get("stake", 0.0)) >= (SETTINGS.bankroll * SETTINGS.max_stake_pct - 1e-9):
            cap_stake_hits += 1

        picks.append(out)

    picks_no_market_skip = [x for x in picks if x.get("decision") != "SKIP_MARKET"]
    picks_no_market_skip.sort(key=lambda x: (x.get("pick_edge") or 0.0), reverse=True)

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

    report = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "settings": SETTINGS.summary(),
        "preflight": preflight,
        "events_loaded": len(events),
        "used_by_mode": used_by_mode,
        "skipped_unknown_tour": skipped_unknown_tour,
        "known_players_ratio": (known_players / total_players) if total_players else None,
        "fallback_level_ratio": (fallback_level_count / processed_events) if processed_events else None,
        "decision_counts": decision_counts,
        "market_skip_reasons": market_skip_reasons,
        "cap_stake_hits": cap_stake_hits,
        "bet_count": bet_count,
    }
    report_path = save_session_report(report)
    if report_path:
        print(f"Session report saved to: {report_path}")

    print("\n=== TOP RECOMMENDATIONS ===")
    for p in picks_no_market_skip[: SETTINGS.print_top_n]:
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
