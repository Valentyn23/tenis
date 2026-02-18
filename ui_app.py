from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from odds_theoddsapi import (
    best_decimal_odds_from_event,
    fetch_h2h_odds_for_sport,
    list_active_tennis_sports,
)
from predictor import Predictor
from config_shared import infer_level_from_sport_key, infer_mode_from_sport_key
from settings import PROFILE_DEFAULTS, load_runtime_settings
from bets_ledger import append_bets
from gemini_features import get_pick_opinion

st.set_page_config(page_title="Tennis Betting MVP", layout="wide")

SETTINGS = load_runtime_settings()
CURRENCY_SYMBOLS = {"UAH": "₴", "USD": "$", "EUR": "€"}


def format_money(amount: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    return f"{symbol}{amount:.2f}"


def make_predictor(mode: str, cfg: dict[str, Any]) -> Optional[Predictor]:
    state_path = cfg["state_path_atp"] if mode == "ATP" else cfg["state_path_wta"]
    try:
        return Predictor(
            model_path=f"model/{mode.lower()}_model.pkl",
            state_path=state_path,
            bankroll=cfg["bankroll"],
            max_stake_pct=cfg["max_stake_pct"],
            kelly_fraction_used=cfg["kelly_fraction"],
            min_edge=cfg["min_edge"],
            prob_floor=cfg["prob_floor"],
            prob_ceil=cfg["prob_ceil"],
            max_overround=cfg["max_overround"],
            strict_mode_match=cfg["strict_mode_match"],
            soft_cap_edge=cfg["soft_cap_edge"],
            soft_cap_factor=cfg["soft_cap_factor"],
            use_calibration=cfg["use_calibration"],
        )
    except Exception as exc:
        st.warning(f"{mode} predictor unavailable: {exc}")
        return None


def run_predictions(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], int]:
    sports = list_active_tennis_sports()
    tennis_keys = [s["key"] for s in sports if "tennis" in s.get("key", "").lower()]

    required_modes = {m for m in (infer_mode_from_sport_key(k) for k in tennis_keys) if m in {"ATP", "WTA"}}
    predictors: dict[str, Predictor] = {}
    for mode in sorted(required_modes):
        p = make_predictor(mode, cfg)
        if p is not None:
            predictors[mode] = p

    if not predictors:
        return pd.DataFrame(), {"error": "No predictors available", "tennis_keys": tennis_keys}, 0

    events = []
    for k in tennis_keys:
        try:
            data = fetch_h2h_odds_for_sport(k, regions=cfg["regions"], odds_format="decimal")
            for e in data:
                best = best_decimal_odds_from_event(e)
                if best:
                    events.append(best)
                    if len(events) >= cfg["max_events"]:
                        break
            if len(events) >= cfg["max_events"]:
                break
        except Exception as ex:
            st.warning(f"Odds error for {k}: {ex}")

    picks = []
    decision_counts = {"BET_A": 0, "BET_B": 0, "NO_BET": 0, "SKIP_MARKET": 0, "SKIP_UNKNOWN_PLAYERS": 0}
    market_skip_reasons: dict[str, int] = {}

    for ev in events:
        event_mode = infer_mode_from_sport_key(ev.get("sport_key"))
        if event_mode not in predictors:
            continue

        out = predictors[event_mode].predict_event(
            playerA=ev["playerA"],
            playerB=ev["playerB"],
            oddsA=ev["oddsA"],
            oddsB=ev["oddsB"],
            surface=cfg["default_surface"],
            level=infer_level_from_sport_key(ev.get("sport_key"), cfg["default_level"])[0],
            rnd=cfg["default_round"],
            best_of=cfg["default_best_of"],
            indoor=cfg["default_indoor"],
            date_iso=(ev.get("commence_time") or "")[:10] or None,
        )
        out["tour_mode"] = event_mode
        out["event_id"] = ev.get("id")
        out["commence_time"] = ev.get("commence_time")

        if cfg["gemini_pick_opinion"] and out.get("decision") in {"BET_A", "BET_B"}:
            out["gemini_opinion"] = get_pick_opinion(
                {
                    "playerA": out.get("playerA"),
                    "playerB": out.get("playerB"),
                    "decision": out.get("decision"),
                    "pick": out.get("pick"),
                    "prob_A_win": out.get("prob_A_win"),
                    "oddsA": out.get("oddsA"),
                    "oddsB": out.get("oddsB"),
                    "pick_edge": out.get("pick_edge"),
                }
            )

        dec = out.get("decision", "NO_BET")
        if dec in decision_counts:
            decision_counts[dec] += 1
        if dec == "SKIP_MARKET":
            reason = out.get("reason", "unknown")
            market_skip_reasons[reason] = market_skip_reasons.get(reason, 0) + 1

        picks.append(out)

    append_count = append_bets(cfg["bets_ledger_path"], picks, currency=cfg["currency"])

    records = []
    for p in picks:
        g = p.get("gemini_opinion") or {}
        records.append(
            {
                "tour": p.get("tour_mode"),
                "match": f"{p.get('playerA')} vs {p.get('playerB')}",
                "decision": p.get("decision"),
                "pick": p.get("pick") or "—",
                "stake": float(p.get("stake", 0.0)),
                "stake_display": format_money(float(p.get("stake", 0.0)), cfg["currency"]),
                "pA": float(p.get("prob_A_win", 0.5)),
                "edge": float(p.get("pick_edge", 0.0)),
                "oddsA": float(p.get("oddsA", 0.0)),
                "oddsB": float(p.get("oddsB", 0.0)),
                "reason": p.get("reason") or "",
                "gemini_stance": g.get("stance", ""),
                "gemini_conf": g.get("confidence", ""),
                "gemini_reason": g.get("short_reason", ""),
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=["edge"], ascending=False)

    meta = {
        "tennis_keys": tennis_keys,
        "events_loaded": len(events),
        "decision_counts": decision_counts,
        "market_skip_reasons": market_skip_reasons,
    }
    return df, meta, append_count


st.title("🎾 Tennis Betting Streamlit MVP")

with st.sidebar:
    st.header("Настройки")
    bankroll = st.number_input("Банкролл", min_value=1.0, value=float(SETTINGS.bankroll), step=100.0)
    risk_profile = st.selectbox("Risk profile", options=list(PROFILE_DEFAULTS.keys()), index=list(PROFILE_DEFAULTS.keys()).index(SETTINGS.risk_profile if SETTINGS.risk_profile in PROFILE_DEFAULTS else "balanced"))
    max_events = st.number_input("Max events", min_value=1, max_value=300, value=int(SETTINGS.max_events), step=1)
    use_calibration = st.toggle("Use calibration", value=bool(SETTINGS.use_calibration))
    gemini_pick_opinion = st.toggle("Gemini opinion", value=bool(SETTINGS.gemini_pick_opinion))
    currency = st.selectbox("Currency", options=["UAH", "USD", "EUR"], index=["UAH", "USD", "EUR"].index(SETTINGS.currency if SETTINGS.currency in {"UAH", "USD", "EUR"} else "UAH"))
    run = st.button("Запустить прогноз", type="primary")

profile_defaults = PROFILE_DEFAULTS[risk_profile]
cfg = {
    "regions": SETTINGS.regions,
    "max_events": int(max_events),
    "default_surface": SETTINGS.default_surface,
    "default_level": SETTINGS.default_level,
    "default_round": SETTINGS.default_round,
    "default_best_of": SETTINGS.default_best_of,
    "default_indoor": SETTINGS.default_indoor,
    "bankroll": float(bankroll),
    "currency": currency,
    "max_stake_pct": profile_defaults["max_stake_pct"],
    "kelly_fraction": profile_defaults["kelly_fraction"],
    "min_edge": profile_defaults["min_edge"],
    "prob_floor": profile_defaults["prob_floor"],
    "prob_ceil": profile_defaults["prob_ceil"],
    "max_overround": profile_defaults["max_overround"],
    "strict_mode_match": SETTINGS.strict_mode_match,
    "soft_cap_edge": profile_defaults["soft_cap_edge"],
    "soft_cap_factor": profile_defaults["soft_cap_factor"],
    "state_path_atp": SETTINGS.state_path_atp,
    "state_path_wta": SETTINGS.state_path_wta,
    "use_calibration": use_calibration,
    "gemini_pick_opinion": gemini_pick_opinion,
    "bets_ledger_path": SETTINGS.bets_ledger_path,
}

if run:
    df, meta, append_count = run_predictions(cfg)
    st.session_state["pred_df"] = df
    st.session_state["pred_meta"] = meta
    st.session_state["saved_bets"] = append_count

pred_df: pd.DataFrame = st.session_state.get("pred_df", pd.DataFrame())
pred_meta: dict[str, Any] = st.session_state.get("pred_meta", {})
saved_bets: int = st.session_state.get("saved_bets", 0)

tab_predictions, tab_ledger = st.tabs(["Прогнозы", "Ledger"])

with tab_predictions:
    if pred_meta:
        st.caption(f"Tennis keys: {pred_meta.get('tennis_keys', [])}")
        st.write(f"Events loaded: **{pred_meta.get('events_loaded', 0)}**")
        st.write("Decision stats:", pred_meta.get("decision_counts", {}))
        if pred_meta.get("market_skip_reasons"):
            st.write("Market skip reasons:", pred_meta.get("market_skip_reasons"))
        if saved_bets:
            st.success(f"Saved bets to ledger: {saved_bets}")

    if pred_df.empty:
        st.info("Нажми 'Запустить прогноз', чтобы получить рекомендации.")
    else:
        view_cols = [
            "tour",
            "match",
            "decision",
            "pick",
            "stake_display",
            "pA",
            "edge",
            "oddsA",
            "oddsB",
            "reason",
            "gemini_stance",
            "gemini_conf",
            "gemini_reason",
        ]
        styled = pred_df[view_cols].style.apply(
            lambda row: [
                "background-color: #d1fae5" if str(row["decision"]).startswith("BET_") else "background-color: #fee2e2"
                for _ in row
            ],
            axis=1,
        )
        st.dataframe(styled, use_container_width=True, height=500)

        c1, c2, c3 = st.columns(3)
        with c1:
            decision_chart = pred_df["decision"].value_counts()
            st.subheader("BET/NO_BET")
            st.bar_chart(decision_chart)
        with c2:
            st.subheader("Stakes")
            st.bar_chart(pred_df[pred_df["stake"] > 0].set_index("match")["stake"])
        with c3:
            st.subheader("Edges")
            st.bar_chart(pred_df.set_index("match")["edge"])

with tab_ledger:
    st.subheader("История ставок")
    ledger_path = Path(SETTINGS.bets_ledger_path)
    if not ledger_path.exists():
        st.info(f"Ledger file not found: {ledger_path}")
    else:
        ledger_df = pd.read_csv(ledger_path)
        st.dataframe(ledger_df, use_container_width=True, height=420)
        total_stake = float(pd.to_numeric(ledger_df.get("stake", 0), errors="coerce").fillna(0).sum())
        st.metric("Всего записанных ставок", value=len(ledger_df))
        st.metric("Сумма stake", value=format_money(total_stake, cfg["currency"]))
