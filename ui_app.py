# ui_app.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone
import app
import pandas as pd
import streamlit as st

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    get_script_run_ctx = None

if get_script_run_ctx is not None and get_script_run_ctx() is None:
    print("ui_app.py must be launched via 'streamlit run ui_app.py' (not 'python ui_app.py').")
    raise SystemExit(0)

from odds_theoddsapi import (
    best_decimal_odds_from_event,
    fetch_h2h_odds_for_sport,
    list_tennis_sports,
    list_supported_tennis_keys,
)
from odds_flashscore import fetch_flashscore_tennis_events
from predictor import Predictor
from config_shared import infer_level_from_sport_key, infer_mode_from_sport_key
from settings import PROFILE_DEFAULTS, load_runtime_settings
from bets_ledger import append_bets
from gemini_features import get_pick_opinion, gemini_is_configured

# Page config
st.set_page_config(
    page_title="Tennis Betting Pro",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .main > div {
        padding-top: 1rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 10px 40px rgba(5, 150, 105, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* ========== SIDEBAR STYLING ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1rem 1rem;
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #334155 !important;
    }

    /* Sidebar section dividers */
    section[data-testid="stSidebar"] hr {
        border-color: #e2e8f0;
        margin: 1rem 0;
    }

    /* Sidebar inputs */
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }

    section[data-testid="stSidebar"] .stNumberInput input:focus,
    section[data-testid="stSidebar"] .stSelectbox > div > div:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }

    /* Sidebar toggle */
    section[data-testid="stSidebar"] .stCheckbox label span {
        color: #334155 !important;
        font-weight: 500 !important;
    }

    /* Sidebar expander */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background: #f1f5f9 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }

    /* ========== METRICS CARDS ========== */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    }

    div[data-testid="metric-container"] label {
        color: #64748b !important;
        font-weight: 600 !important;
    }

    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    /* ========== SOURCE BADGES ========== */
    .source-odds {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }

    .source-flash {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.3);
    }

    /* ========== BUTTON STYLING ========== */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.85rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.5);
        transform: translateY(-2px);
    }

    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0);
    }

    /* ========== TABLE STYLING ========== */
    .dataframe {
        font-size: 0.9rem;
    }

    /* ========== SUCCESS/WARNING MESSAGES ========== */
    .stSuccess {
        background: linear-gradient(90deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 4px solid #22c55e;
        border-radius: 8px;
    }

    .stWarning {
        background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
    }

    .stInfo {
        background: linear-gradient(90deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
    }

    /* ========== TAB STYLING ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        padding: 4px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        color: #64748b;
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #10b981 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* ========== SLIDER ========== */
    section[data-testid="stSidebar"] .stSlider > div > div {
        color: #334155 !important;
    }

    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }

    /* ========== EXPANDER IN MAIN ========== */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* ========== DOWNLOAD BUTTON ========== */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    /* ========== SELECTBOX DROPDOWN ========== */
    [data-baseweb="select"] > div {
        background: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }

    [data-baseweb="select"] > div:focus-within {
        border-color: #10b981 !important;
    }
</style>
""", unsafe_allow_html=True)

SETTINGS = load_runtime_settings()
CURRENCY_SYMBOLS = {"UAH": "₴", "USD": "$", "EUR": "€"}

SOURCE_BADGES = {
    "TheOddsAPI": "🔵 Odds API",
    "FlashScore": "🟢 FlashScore",
}


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
    tennis_keys = list_supported_tennis_keys()

    try:
        sports = list_tennis_sports(only_active=False)
        discovered = [s["key"] for s in sports if "tennis" in s.get("key", "").lower()]
        for k in discovered:
            if k not in tennis_keys:
                tennis_keys.append(k)
    except Exception:
        pass

    required_modes = {m for m in (infer_mode_from_sport_key(k) for k in tennis_keys) if m in {"ATP", "WTA"}}
    predictors: dict[str, Predictor] = {}
    for mode in sorted(required_modes):
        p = make_predictor(mode, cfg)
        if p is not None:
            predictors[mode] = p

    if not predictors:
        return pd.DataFrame(), {"error": "No predictors available", "tennis_keys": tennis_keys}, 0

    events = []
    events_by_source = {"TheOddsAPI": 0, "FlashScore": 0}
    skipped_live = 0

    now = datetime.now(timezone.utc)

    # === FETCH FROM THE ODDS API ===
    for k in tennis_keys:
        try:
            data = fetch_h2h_odds_for_sport(k, regions=cfg["regions"], odds_format="decimal")
            for e in data:
                best = best_decimal_odds_from_event(e)
                if best:
                    is_live = False
                    if cfg.get("only_prematch", False):
                        commence_str = best.get("commence_time", "")
                        if commence_str:
                            try:
                                commence_time = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
                                if commence_time <= now:
                                    is_live = True
                                    skipped_live += 1
                                    continue
                            except:
                                pass

                    best["is_live"] = is_live
                    events.append(best)
                    events_by_source["TheOddsAPI"] += 1
                    if len(events) >= cfg["max_events"]:
                        break
            if len(events) >= cfg["max_events"]:
                break
        except Exception:
            continue

    # === FETCH FROM FLASHSCORE ===
    remaining_slots = cfg["max_events"] - len(events)
    if remaining_slots > 0:
        try:
            flashscore_events = fetch_flashscore_tennis_events(
                max_events=remaining_slots,
                only_prematch=cfg.get("only_prematch", False)
            )
            for fs_event in flashscore_events:
                is_duplicate = False
                for existing in events:
                    if (existing["playerA"] == fs_event["playerA"] and
                            existing["playerB"] == fs_event["playerB"]):
                        is_duplicate = True
                        break
                    if (existing["playerA"] == fs_event["playerB"] and
                            existing["playerB"] == fs_event["playerA"]):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    events.append(fs_event)
                    events_by_source["FlashScore"] += 1
                    if len(events) >= cfg["max_events"]:
                        break
        except Exception as ex:
            st.warning(f"FlashScore fetch error: {ex}")

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
        out["source"] = ev.get("source", "TheOddsAPI")
        out["is_live"] = ev.get("is_live", False)

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

    # Risk multiplier for different sources
    SOURCE_RISK_MULTIPLIER = {
        "TheOddsAPI": 1.0,  # 100% от расчетной ставки
        "FlashScore": 0.5,  # 50% от расчетной ставки (более рисковые)
    }

    records = []
    for p in picks:
        g = p.get("gemini_opinion") or {}
        source = p.get("source", "TheOddsAPI")
        source_badge = SOURCE_BADGES.get(source, source)

        # Apply risk multiplier based on source
        original_stake = float(p.get("stake", 0.0))
        risk_multiplier = SOURCE_RISK_MULTIPLIER.get(source, 1.0)
        adjusted_stake = original_stake * risk_multiplier

        records.append(
            {
                "source": source_badge,
                "tour": p.get("tour_mode"),
                "match": f"{p.get('playerA')} vs {p.get('playerB')}",
                "decision": p.get("decision"),
                "pick": p.get("pick") or "—",
                "stake": adjusted_stake,
                "stake_display": format_money(adjusted_stake, cfg["currency"]),
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
        "events_by_source": events_by_source,
        "skipped_live": skipped_live,
        "decision_counts": decision_counts,
        "market_skip_reasons": market_skip_reasons,
    }
    return df, meta, append_count


# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>🎾 Tennis Betting Pro</h1>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    st.markdown("---")

    st.markdown("### 💰 Банкролл")
    bankroll = st.number_input(
        "Сумма",
        min_value=1.0,
        value=float(SETTINGS.bankroll),
        step=100.0,
        format="%.2f",
        label_visibility="collapsed"
    )

    currency = st.selectbox(
        "Валюта",
        options=["UAH", "USD", "EUR"],
        index=["UAH", "USD", "EUR"].index(SETTINGS.currency if SETTINGS.currency in {"UAH", "USD", "EUR"} else "UAH")
    )

    st.markdown("---")

    st.markdown("### 📊 Риск-профиль")
    risk_profile = st.selectbox(
        "Профиль",
        options=list(PROFILE_DEFAULTS.keys()),
        index=list(PROFILE_DEFAULTS.keys()).index(
            SETTINGS.risk_profile if SETTINGS.risk_profile in PROFILE_DEFAULTS else "balanced"),
        label_visibility="collapsed"
    )

    profile_info = PROFILE_DEFAULTS[risk_profile]
    with st.expander("Детали профиля"):
        st.write(f"Kelly: {profile_info['kelly_fraction']}")
        st.write(f"Max stake: {profile_info['max_stake_pct'] * 100}%")
        st.write(f"Min edge: {profile_info['min_edge']}")

    st.markdown("---")

    st.markdown("### 📡 Источники данных")
    max_events = st.slider(
        "Макс. матчей",
        min_value=5,
        max_value=100,
        value=int(SETTINGS.max_events),
        step=5
    )

    st.markdown("""
    <div style="display: flex; gap: 8px; margin: 8px 0;">
        <span class="source-odds">🔵 Odds API</span>
        <span class="source-flash">🟢 FlashScore</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🎯 Фильтры")
    only_prematch = st.toggle(
        "Только прематч",
        value=True,
        help="Показывать только матчи которые ещё не начались"
    )

    use_calibration = st.toggle(
        "Калибровка вероятностей",
        value=bool(SETTINGS.use_calibration)
    )

    gemini_pick_opinion = st.toggle(
        "Gemini мнение",
        value=bool(SETTINGS.gemini_pick_opinion)
    )

    if gemini_pick_opinion:
        ok, status_reason = gemini_is_configured()
        if not ok:
            st.warning(f"⚠️ {status_reason}")

    st.markdown("---")

    run = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)

# Build config
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
    "only_prematch": only_prematch,
}

# Run predictions
if run:
    with st.spinner("🔄 Загрузка данных и анализ..."):
        df, meta, append_count = run_predictions(cfg)
        st.session_state["pred_df"] = df
        st.session_state["pred_meta"] = meta
        st.session_state["saved_bets"] = append_count

pred_df: pd.DataFrame = st.session_state.get("pred_df", pd.DataFrame())
pred_meta: dict[str, Any] = st.session_state.get("pred_meta", {})
saved_bets: int = st.session_state.get("saved_bets", 0)

# ==================== MAIN CONTENT ====================
tab_predictions, tab_ledger, tab_stats = st.tabs(["📋 Рекомендации", "📒 Ledger", "📊 Статистика"])

with tab_predictions:
    if pred_meta:
        col1, col2, col3, col4, col5 = st.columns(5)

        events_by_source = pred_meta.get("events_by_source", {})
        decision_counts = pred_meta.get("decision_counts", {})

        with col1:
            st.metric("📊 Всего матчей", pred_meta.get("events_loaded", 0))
        with col2:
            st.metric("🔵 Odds API", events_by_source.get("TheOddsAPI", 0))
        with col3:
            st.metric("🟢 FlashScore", events_by_source.get("FlashScore", 0))
        with col4:
            bet_count = decision_counts.get("BET_A", 0) + decision_counts.get("BET_B", 0)
            st.metric("✅ Ставок", bet_count)
        with col5:
            st.metric("⏭️ Пропущено лайв", pred_meta.get("skipped_live", 0))

        st.markdown("---")

        with st.expander("📈 Decision Stats", expanded=False):
            cols = st.columns(5)
            for i, (key, value) in enumerate(decision_counts.items()):
                with cols[i % 5]:
                    st.metric(key, value)

            if pred_meta.get("market_skip_reasons"):
                st.write("**Market skip reasons:**", pred_meta.get("market_skip_reasons"))

        if saved_bets:
            st.success(f"✅ Сохранено ставок в ledger: {saved_bets}")

    if pred_df.empty:
        st.info("👆 Нажми **'Запустить анализ'** чтобы получить рекомендации.")
    else:
        col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 6])
        with col_filter1:
            filter_decision = st.selectbox(
                "Фильтр решений",
                options=["Все", "Только BET", "BET_A", "BET_B", "NO_BET", "SKIP"],
                index=0
            )
        with col_filter2:
            filter_source = st.selectbox(
                "Фильтр источника",
                options=["Все", "🔵 Odds API", "🟢 FlashScore"],
                index=0
            )

        filtered_df = pred_df.copy()

        if filter_decision == "Только BET":
            filtered_df = filtered_df[filtered_df["decision"].str.startswith("BET_")]
        elif filter_decision in ["BET_A", "BET_B", "NO_BET"]:
            filtered_df = filtered_df[filtered_df["decision"] == filter_decision]
        elif filter_decision == "SKIP":
            filtered_df = filtered_df[filtered_df["decision"].str.startswith("SKIP")]

        if filter_source == "🔵 Odds API":
            filtered_df = filtered_df[filtered_df["source"] == "🔵 Odds API"]
        elif filter_source == "🟢 FlashScore":
            filtered_df = filtered_df[filtered_df["source"] == "🟢 FlashScore"]

        view_cols = [
            "source",
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
        ]

        if gemini_pick_opinion:
            view_cols.extend(["gemini_stance", "gemini_conf", "gemini_reason"])


        def highlight_rows(row):
            if str(row["decision"]).startswith("BET_"):
                return ["background-color: #dcfce7"] * len(row)
            elif str(row["decision"]) == "NO_BET":
                return ["background-color: #fef3c7"] * len(row)
            else:
                return ["background-color: #fee2e2"] * len(row)


        styled = filtered_df[view_cols].style.apply(highlight_rows, axis=1)
        styled = styled.format({
            "pA": "{:.3f}",
            "edge": "{:+.3f}",
            "oddsA": "{:.2f}",
            "oddsB": "{:.2f}",
        })

        st.dataframe(styled, use_container_width=True, height=500)

        st.markdown("""
        <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 0.85rem;">
            <span>🟢 <b>BET</b> - Рекомендуется ставка</span>
            <span>🟡 <b>NO_BET</b> - Недостаточно edge</span>
            <span>🔴 <b>SKIP</b> - Пропустить</span>
        </div>
        """, unsafe_allow_html=True)

with tab_ledger:
    st.markdown("### 📒 История ставок")

    ledger_path = Path(SETTINGS.bets_ledger_path)
    if not ledger_path.exists():
        st.info(f"📁 Файл ledger не найден: `{ledger_path}`")
    else:
        ledger_df = pd.read_csv(ledger_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Всего записей", len(ledger_df))
        with col2:
            total_stake = float(pd.to_numeric(ledger_df.get("stake", 0), errors="coerce").fillna(0).sum())
            st.metric("💰 Сумма ставок", format_money(total_stake, cfg["currency"]))
        with col3:
            if "timestamp" in ledger_df.columns:
                try:
                    ledger_df["timestamp"] = pd.to_datetime(ledger_df["timestamp"])
                    last_date = ledger_df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
                    st.metric("🕐 Последняя запись", last_date)
                except:
                    st.metric("🕐 Последняя запись", "—")

        st.markdown("---")
        st.dataframe(ledger_df, use_container_width=True, height=420)

        csv = ledger_df.to_csv(index=False)
        st.download_button(
            label="📥 Скачать CSV",
            data=csv,
            file_name="betting_ledger.csv",
            mime="text/csv"
        )

with tab_stats:
    st.markdown("### 📊 Аналитика")

    if pred_df.empty:
        st.info("Запустите анализ для отображения статистики.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Распределение решений")
            decision_chart = pred_df["decision"].value_counts()
            st.bar_chart(decision_chart)

        with col2:
            st.markdown("#### Источники данных")
            source_chart = pred_df["source"].value_counts()
            st.bar_chart(source_chart)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Размеры ставок")
            bet_df = pred_df[pred_df["stake"] > 0].copy()
            if not bet_df.empty:
                st.bar_chart(bet_df.set_index("match")["stake"])
            else:
                st.info("Нет ставок для отображения")

        with col4:
            st.markdown("#### Edge по матчам")
            edge_df = pred_df[pred_df["edge"] != 0].copy()
            if not edge_df.empty:
                st.bar_chart(edge_df.set_index("match")["edge"])
            else:
                st.info("Нет данных edge")

        st.markdown("---")
        st.markdown("#### По турам")

        tour_stats = pred_df.groupby("tour").agg({
            "match": "count",
            "stake": "sum",
            "edge": "mean"
        }).rename(columns={"match": "Матчей", "stake": "Сумма ставок", "edge": "Средний edge"})

        st.dataframe(tour_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    ⚠️ Ставки на спорт несут финансовые риски. Играйте ответственно.
</div>
""", unsafe_allow_html=True)