from pathlib import Path

from bets_ledger import append_bets, analyze_ledger, settle_from_results_csv


def test_ledger_append_settle_and_analyze(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    results = tmp_path / "results.csv"

    picks = [
        {
            "decision": "BET_A",
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
