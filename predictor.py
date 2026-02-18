# predictor.py
# Fully functional predictor module (fixed + safer for betting):
# - Loads trained model bundle (XGB + calibrator + feature_names)
# - Loads warmed StateEngine from state/engine_state.pkl
# - Normalizes OddsAPI player names -> dataset format (e.g., "Carlos Alcaraz" -> "Alcaraz C.")
# - Gemini ALWAYS ON (uses RAW names for news search)
# - Adds market sanity filters (skip illiquid/buggy odds)
# - Adds probability safety clamp to avoid extreme calibrated outputs
# - Produces value/edge + stake sizing (fractional Kelly with cap)
#
# Required files:
#   - state_engine.py
#   - state_persistence.py
#   - gemini_features.py
#   - model/atp_model.pkl or model/wta_model.pkl (bundle dict)
#   - state/engine_state.pkl (created by warmup/wrump)
#
# Usage:
#   from predictor import Predictor
#   p = Predictor("model/atp_model.pkl", bankroll=1000, debug=True)
#   out = p.predict_event("Carlos Alcaraz","Novak Djokovic",1.85,2.05,surface="Hard", level=2, rnd=4)

import math
from datetime import datetime
from typing import Dict, Any, Optional

import joblib
import pandas as pd

from gemini_features import get_gemini_features  # Gemini ALWAYS ON
from state_persistence import load_engine


# =========================================================
# NAME NORMALIZATION (OddsAPI -> dataset)
# =========================================================
def to_dataset_name(full_name: str) -> str:
    """
    Convert 'Carlos Alcaraz' -> 'Alcaraz C.'
    Convert 'Elina Svitolina' -> 'Svitolina E.'
    Works for most ATP/WTA historical datasets where players are stored as "Lastname F."
    """
    if not full_name:
        return full_name

    s = str(full_name).strip()
    if not s:
        return s

    parts = [p for p in s.split() if p]
    if len(parts) == 1:
        return parts[0]

    first = parts[0]
    last = parts[-1]

    initial = first[0].upper() if first else ""
    return f"{last} {initial}."


# =========================================================
# UTILS
# =========================================================
def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds and odds > 1.0 else 0.5


def kelly_fraction(p: float, odds: float) -> float:
    """
    Kelly fraction for decimal odds:
      b = odds - 1
      f* = (p*b - (1-p)) / b
    """
    if odds is None or odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - p
    f = (p * b - q) / b
    return max(0.0, f)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================================================
# PREDICTOR
# =========================================================
class Predictor:
    def __init__(
        self,
        model_path: str,
        bankroll: float = 1000.0,
        max_stake_pct: float = 0.03,         # cap per bet (e.g. 0.03 = 3%)
        kelly_fraction_used: float = 0.5,     # 0.5 = half-Kelly
        min_edge: float = 0.015,              # only bet if edge >= 1.5%
        state_path: str = "state/engine_state.pkl",
        debug: bool = False,

        # --- safety controls ---
        prob_floor: float = 0.03,             # avoid 0.001/0.999 extremes
        prob_ceil: float = 0.97,
        min_odds_allowed: float = 1.10,       # skip ultra-favorites (often bad liquidity)
        max_odds_allowed: float = 15.0,       # skip extreme underdogs (often bad lines)
    ):
        self.debug = bool(debug)

        self.prob_floor = float(prob_floor)
        self.prob_ceil = float(prob_ceil)
        self.min_odds_allowed = float(min_odds_allowed)
        self.max_odds_allowed = float(max_odds_allowed)

        # ---- load model bundle ----
        bundle = joblib.load(model_path)
        if not (isinstance(bundle, dict) and "model" in bundle and "calibrator" in bundle):
            raise RuntimeError(
                "Model bundle must be dict with keys: model, calibrator, feature_names.\n"
                "You are likely loading the wrong file."
            )

        self.model = bundle["model"]
        self.cal = bundle["calibrator"]
        self.feature_names = bundle.get("feature_names", None)

        # ---- bankroll / staking ----
        self.bankroll = float(bankroll)
        self.max_stake_pct = float(max_stake_pct)
        self.kelly_fraction_used = float(kelly_fraction_used)
        self.min_edge = float(min_edge)

        # ---- load warmed state engine ----
        eng = load_engine(state_path)
        if eng is None:
            raise RuntimeError(
                f"StateEngine is not warmed up.\n"
                f"Missing: {state_path}\n"
                f"Run: python wrump.py  (or warmup.py) to create it."
            )

        self.engine = eng
        total_player_matches = sum(st.matches for st in self.engine.players.values())
        inferred_matches = total_player_matches / 2.0
        print(
            f"Loaded StateEngine from {state_path} | "
            f"players: {len(self.engine.players)} | inferred_matches: {inferred_matches:.0f}"
        )

    # -----------------------------------------------------
    # GEMINI FEATURES (raw names)
    # -----------------------------------------------------
    def _add_gemini_diff(self, feats: Dict[str, float], A_raw: str, B_raw: str) -> None:
        g1 = get_gemini_features(A_raw)
        g2 = get_gemini_features(B_raw)

        feats["injury_diff"] = float(g1.get("injury_risk", 0.5) - g2.get("injury_risk", 0.5))
        feats["mental_diff"] = float(g1.get("mental_form", 0.5) - g2.get("mental_form", 0.5))
        feats["motivation_diff"] = float(g1.get("motivation", 0.5) - g2.get("motivation", 0.5))
        feats["fatigue_news_diff"] = float(g1.get("fatigue_risk", 0.5) - g2.get("fatigue_risk", 0.5))
        feats["confidence_diff"] = float(g1.get("confidence", 0.5) - g2.get("confidence", 0.5))

    # -----------------------------------------------------
    # MARKET SANITY
    # -----------------------------------------------------
    def _is_bad_market(self, oddsA: float, oddsB: float) -> bool:
        lo = min(float(oddsA), float(oddsB))
        hi = max(float(oddsA), float(oddsB))
        if lo < self.min_odds_allowed:
            return True
        if hi > self.max_odds_allowed:
            return True
        return False

    # -----------------------------------------------------
    # CORE PREDICT
    # -----------------------------------------------------
    def predict_event(
        self,
        playerA: str,
        playerB: str,
        oddsA: float,
        oddsB: float,
        surface: str = "Hard",
        level: float = 1.0,
        rnd: float = 1.0,
        best_of: float = 3.0,
        indoor: float = 0.0,
        date_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        if date_iso is None:
            date_iso = datetime.utcnow().date().isoformat()

        playerA_raw = str(playerA)
        playerB_raw = str(playerB)

        # normalize to dataset naming for StateEngine
        playerA_ds = to_dataset_name(playerA_raw)
        playerB_ds = to_dataset_name(playerB_raw)

        if self.debug:
            a_in = playerA_ds in self.engine.players
            b_in = playerB_ds in self.engine.players
            print(f"[DEBUG] ODDS: {playerA_raw} vs {playerB_raw}")
            print(f"[DEBUG] DS  : {playerA_ds} ({'FOUND' if a_in else 'NEW'}) vs {playerB_ds} ({'FOUND' if b_in else 'NEW'})")

        # 1) base features from state (dataset names)
        feats = self.engine.build_features(
            A=playerA_ds,
            B=playerB_ds,
            surface=surface,
            level=level,
            rnd=rnd,
            best_of=best_of,
            indoor=indoor,
            date=date_iso,
        )

        # 2) Gemini ALWAYS ON (raw names)
        self._add_gemini_diff(feats, playerA_raw, playerB_raw)

        # 3) align columns
        cols = self.feature_names or sorted(feats.keys())
        X = pd.DataFrame([{c: float(feats.get(c, 0.0)) for c in cols}], columns=cols)

        # 4) predict + calibrate
        raw = float(self.model.predict_proba(X)[:, 1][0])
        pA = float(self.cal.transform([raw])[0])

        # SAFETY clamp (betting)
        pA = clamp(pA, self.prob_floor, self.prob_ceil)

        # 5) market sanity skip (important!)
        if self._is_bad_market(float(oddsA), float(oddsB)):
            return {
                "playerA": playerA_raw,
                "playerB": playerB_raw,
                "playerA_ds": playerA_ds,
                "playerB_ds": playerB_ds,
                "prob_A_win": float(pA),
                "raw_prob_A_win": float(raw),
                "oddsA": float(oddsA),
                "oddsB": float(oddsB),
                "decision": "SKIP_MARKET",
                "reason": "bad_odds_range",
                "stake": 0.0,
                "pick": None,
                "pick_odds": None,
                "pick_edge": 0.0,
                "meta": {
                    "surface": surface,
                    "level": float(level),
                    "round": float(rnd),
                    "best_of": float(best_of),
                    "indoor": float(indoor),
                    "date": date_iso,
                },
            }

        # 6) value / edge / stake sizing
        market_pA = implied_prob(float(oddsA))
        market_pB = implied_prob(float(oddsB))

        edgeA = pA - market_pA
        edgeB = (1.0 - pA) - market_pB

        fA = kelly_fraction(pA, float(oddsA)) * self.kelly_fraction_used
        fB = kelly_fraction(1.0 - pA, float(oddsB)) * self.kelly_fraction_used

        stakeA = self.bankroll * clamp(fA, 0.0, self.max_stake_pct)
        stakeB = self.bankroll * clamp(fB, 0.0, self.max_stake_pct)

        decision = "NO_BET"
        pick = None
        pick_odds = None
        pick_edge = 0.0
        stake = 0.0

        if edgeA >= self.min_edge and stakeA > 0:
            decision = "BET_A"
            pick = playerA_raw
            pick_odds = float(oddsA)
            pick_edge = float(edgeA)
            stake = float(stakeA)

        if edgeB >= self.min_edge and stakeB > 0 and edgeB > pick_edge:
            decision = "BET_B"
            pick = playerB_raw
            pick_odds = float(oddsB)
            pick_edge = float(edgeB)
            stake = float(stakeB)

        return {
            "playerA": playerA_raw,
            "playerB": playerB_raw,
            "playerA_ds": playerA_ds,
            "playerB_ds": playerB_ds,
            "prob_A_win": float(pA),
            "raw_prob_A_win": float(raw),
            "oddsA": float(oddsA),
            "oddsB": float(oddsB),
            "market_prob_A": float(market_pA),
            "market_prob_B": float(market_pB),
            "edgeA": float(edgeA),
            "edgeB": float(edgeB),
            "decision": decision,
            "pick": pick,
            "stake": round(stake, 2),
            "pick_odds": pick_odds,
            "pick_edge": float(pick_edge),
            "meta": {
                "surface": surface,
                "level": float(level),
                "round": float(rnd),
                "best_of": float(best_of),
                "indoor": float(indoor),
                "date": date_iso,
            },
        }
