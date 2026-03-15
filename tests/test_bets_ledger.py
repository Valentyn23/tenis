from pathlib import Path

from bets_ledger import (
    append_bets,
    analyze_ledger,
    settle_from_results_csv,
    settle_from_winners_map,
    settle_from_finished_events,
)


def test_ledger_append_settle_and_analyze(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    results = tmp_path / "results.csv"

    picks = [
        {
            "decision": "BET_A",
            "strategy": "aggressive",
            "event_id": "evt1",
            "tour_mode": "ATP",
            "commence_time": "2026-02-18T10:00:00Z",
            "playerA": "A",
            "playerB": "B",
            "pick": "A",
            "reason": None,
            "oddsA": 2.0,
            "oddsB": 1.9,
            "pick_odds": 2.0,
            "prob_A_win": 0.6,
            "pick_edge": 0.1,
            "stake": 100.0,
        },
        {
            "decision": "NO_BET",
            "event_id": "evt2",
            "playerA": "C",
            "playerB": "D",
            "stake": 0.0,
        },
    ]

    appended = append_bets(str(ledger), picks, currency="UAH")
    assert appended == 1

    results.write_text("event_id,winner\nevt1,A\n", encoding="utf-8")
    settled = settle_from_results_csv(str(ledger), str(results))
    assert settled == 1

    stats = analyze_ledger(str(ledger), bankroll_start=1000.0)
    assert stats["total_bets"] == 1
    assert stats["closed_bets"] == 1
    assert stats["wins"] == 1
    assert stats["total_pnl"] == 100.0
    assert stats["current_bankroll"] == 1100.0
    assert stats["by_strategy"]["aggressive"]["closed_bets"] == 1
    assert stats["by_strategy"]["aggressive"]["roi"] == 1.0


def test_settle_from_winners_map(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"

    picks = [
        {
            "decision": "BET_B",
            "strategy": "balanced",
            "event_id": "evt42",
            "tour_mode": "ATP",
            "commence_time": "2026-02-18T10:00:00Z",
            "playerA": "A",
            "playerB": "B",
            "pick": "B",
            "reason": None,
            "oddsA": 1.8,
            "oddsB": 2.2,
            "pick_odds": 2.2,
            "prob_A_win": 0.45,
            "pick_edge": 0.08,
            "stake": 50.0,
        },
    ]

    assert append_bets(str(ledger), picks, currency="UAH") == 1
    settled = settle_from_winners_map(str(ledger), {"evt42": "B"})
    assert settled == 1

    stats = analyze_ledger(str(ledger), bankroll_start=1000.0)
    assert stats["wins"] == 1
    assert stats["total_pnl"] == 60.0


def test_settle_from_finished_events_fallback_players(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"

    picks = [
        {
            "decision": "BET_A",
            "strategy": "conservative",
            "event_id": "odds_api_evt_1",
            "tour_mode": "ATP",
            "commence_time": "2026-02-18T10:00:00+00:00",
            "playerA": "Carlos Alcaraz",
            "playerB": "Novak Djokovic",
            "pick": "Carlos Alcaraz",
            "reason": None,
            "oddsA": 1.9,
            "oddsB": 2.0,
            "pick_odds": 1.9,
            "prob_A_win": 0.55,
            "pick_edge": 0.03,
            "stake": 40.0,
        },
    ]

    assert append_bets(str(ledger), picks, currency="UAH") == 1

    finished_events = [
        {
            "event_id": "flashscore_match_777",
            "playerA": "Carlos Alcaraz",
            "playerB": "Novak Djokovic",
            "winner": "Carlos Alcaraz",
            "commence_time": "2026-02-18T12:00:00+00:00",
        }
    ]

    stats = settle_from_finished_events(str(ledger), finished_events, max_hours_diff=6)
    assert stats["settled"] == 1
    assert stats["by_event_id"] == 0
    assert stats["by_match_fallback"] == 1

    agg = analyze_ledger(str(ledger), bankroll_start=1000.0)
    assert agg["wins"] == 1


def test_append_migrates_legacy_header(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    legacy_header = "timestamp_utc,event_id,tour_mode,commence_time,playerA,playerB,pick,decision,reason,oddsA,oddsB,pick_odds,prob_A_win,pick_edge,stake,currency,result,pnl\n"
    legacy_row = "2026-01-01T00:00:00Z,evt_old,ATP,2026-01-01T12:00:00Z,A,B,A,BET_A,,2.0,1.9,2.0,0.6,0.1,100,UAH,,\n"
    ledger.write_text(legacy_header + legacy_row, encoding="utf-8")

    picks = [{
        "decision": "BET_B",
        "strategy": "aggressive",
        "event_id": "evt_new",
        "tour_mode": "ATP",
        "commence_time": "2026-01-02T12:00:00Z",
        "playerA": "C",
        "playerB": "D",
        "pick": "D",
        "pick_odds": 2.1,
        "stake": 50.0,
    }]

    assert append_bets(str(ledger), picks, currency="UAH") == 1

    content = ledger.read_text(encoding="utf-8").splitlines()
    assert content[0].split(",")[1] == "strategy"
    assert len(content) == 3
