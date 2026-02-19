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
import re
from datetime import datetime, timezone
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

GEMINI_FEATURE_COLUMNS = {
    "injury_diff",
    "mental_diff",
    "motivation_diff",
    "fatigue_news_diff",
    "confidence_diff",
}

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
        use_calibration: bool = True,

        # --- safety controls ---
        prob_floor: float = 0.08,             # avoid 0.001/0.999 extremes
        prob_ceil: float = 0.95,
        min_odds_allowed: float = 1.15,       # skip ultra-favorites (often bad liquidity)
        max_odds_allowed: float = 10,       # skip extreme underdogs (often bad lines)
        max_overround: float = 1.08,          # skip over-vig markets
        strict_mode_match: bool = False,      # fail if engine/model mode mismatch
        soft_cap_edge: float = 0.25,          # additional dampener above this edge
        soft_cap_factor: float = 0.60,
        clamp_guard_band: float = 0.02,
    ):
        self.debug = bool(debug)
        self.use_calibration = bool(use_calibration)

        self.prob_floor = float(prob_floor)
        self.prob_ceil = float(prob_ceil)
        self.min_odds_allowed = float(min_odds_allowed)
        self.max_odds_allowed = float(max_odds_allowed)
        self.max_overround = float(max_overround)
        self.strict_mode_match = bool(strict_mode_match)
        self.soft_cap_edge = float(soft_cap_edge)
        self.soft_cap_factor = float(soft_cap_factor)
        self.clamp_guard_band = float(clamp_guard_band)

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
        self.model_mode = str(bundle.get("mode", "ATP")).upper()

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
        self._known_players = set(self.engine.players.keys())
        self._exact_name_lookup = {p.lower(): p for p in self._known_players}
        self._surname_initial_lookup = self._build_surname_initial_lookup()
        feature_set = set(self.feature_names or [])
        self.use_gemini_features = bool(feature_set.intersection(GEMINI_FEATURE_COLUMNS))
        total_player_matches = sum(st.matches for st in self.engine.players.values())
        inferred_matches = total_player_matches / 2.0
        engine_mode = str(getattr(self.engine, "mode", "ATP")).upper()
        print(
            f"Loaded StateEngine from {state_path} | "
            f"players: {len(self.engine.players)} | inferred_matches: {inferred_matches:.0f} | "
            f"engine_mode: {engine_mode} | model_mode: {self.model_mode}"
        )
        print(f"Gemini features enabled by model schema: {self.use_gemini_features}")

        if self.strict_mode_match and engine_mode != self.model_mode:
            raise RuntimeError(
                f"Mode mismatch: engine_mode={engine_mode}, model_mode={self.model_mode}. "
                f"Use a matching state_path for this model."
            )

    def _build_surname_initial_lookup(self) -> Dict[tuple[str, str], Optional[str]]:
        lookup: Dict[tuple[str, str], Optional[str]] = {}
        pat = re.compile(r"^(.+)\s+([A-Z])\.$")
        for p in self._known_players:
            m = pat.match(p)
            if not m:
                continue
            key = (m.group(1).strip().lower(), m.group(2))
            if key in lookup and lookup[key] != p:
                lookup[key] = None
            else:
                lookup[key] = p
        return lookup

    def resolve_dataset_name(self, raw_name: str) -> tuple[str, bool]:
        raw = str(raw_name).strip()
        if not raw:
            return raw, False

        if raw in self._known_players:
            return raw, True

        exact = self._exact_name_lookup.get(raw.lower())
        if exact:
            return exact, True

        canonical = to_dataset_name(raw)
        if canonical in self._known_players:
            return canonical, True

        parts = [p for p in raw.split() if p]
        if len(parts) >= 2:
            key = (parts[-1].lower(), parts[0][0].upper())
            found = self._surname_initial_lookup.get(key)
            if found:
                return found, True

        return canonical, False

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
    def _is_bad_market(self, oddsA: float, oddsB: float) -> tuple[bool, str]:
        lo = min(float(oddsA), float(oddsB))
        hi = max(float(oddsA), float(oddsB))
        if lo < self.min_odds_allowed:
            return True, "bad_odds_range"
        if hi > self.max_odds_allowed:
            return True, "bad_odds_range"

        overround = implied_prob(float(oddsA)) + implied_prob(float(oddsB))
        if overround > self.max_overround:
            return True, "high_overround"

        return False, "ok"

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
            date_iso = datetime.now(timezone.utc).date().isoformat()

        playerA_raw = str(playerA)
        playerB_raw = str(playerB)

        # normalize to dataset naming for StateEngine
        playerA_ds, playerA_known = self.resolve_dataset_name(playerA_raw)
        playerB_ds, playerB_known = self.resolve_dataset_name(playerB_raw)

        if self.debug:
            print(f"[DEBUG] ODDS: {playerA_raw} vs {playerB_raw}")
            print(
                f"[DEBUG] DS  : {playerA_ds} ({'FOUND' if playerA_known else 'NEW'}) "
                f"vs {playerB_ds} ({'FOUND' if playerB_known else 'NEW'})"
            )

        if not (playerA_known and playerB_known):
            return {
                "playerA": playerA_raw,
                "playerB": playerB_raw,
                "playerA_ds": playerA_ds,
                "playerB_ds": playerB_ds,
                "prob_A_win": 0.5,
                "raw_prob_A_win": 0.5,
                "oddsA": float(oddsA),
                "oddsB": float(oddsB),
                "decision": "SKIP_UNKNOWN_PLAYERS",
                "reason": "player_not_in_warmup_state",
                "stake": 0.0,
                "pick": None,
                "pick_odds": None,
                "pick_edge": 0.0,
                "edgeA": 0.0,
                "edgeB": 0.0,
                "meta": {
                    "surface": surface,
                    "level": float(level),
                    "round": float(rnd),
                    "best_of": float(best_of),
                    "indoor": float(indoor),
                    "date": date_iso,
                    "playerA_known": bool(playerA_known),
                    "playerB_known": bool(playerB_known),
                },
            }

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

        # 2) Gemini features only if model schema includes them
        if self.use_gemini_features:
            self._add_gemini_diff(feats, playerA_raw, playerB_raw)

        # 3) align columns
        cols = self.feature_names or sorted(feats.keys())
        X = pd.DataFrame([{c: float(feats.get(c, 0.0)) for c in cols}], columns=cols)

        # 4) predict + calibrate
        raw = float(self.model.predict_proba(X)[:, 1][0])
        if self.use_calibration and self.cal is not None:
            pA = float(self.cal.transform([raw])[0])
        else:
            pA = raw

        # SAFETY clamp (betting)
        pA = clamp(pA, self.prob_floor, self.prob_ceil)

        # 5) market sanity skip (important!)
        bad_market, market_reason = self._is_bad_market(float(oddsA), float(oddsB))
        if bad_market:
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
                "reason": market_reason,
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

        near_floor = pA <= (self.prob_floor + self.clamp_guard_band)
        near_ceil = pA >= (self.prob_ceil - self.clamp_guard_band)

        fA = kelly_fraction(pA, float(oddsA)) * self.kelly_fraction_used
        fB = kelly_fraction(1.0 - pA, float(oddsB)) * self.kelly_fraction_used

        stakeA = self.bankroll * clamp(fA, 0.0, self.max_stake_pct)
        stakeB = self.bankroll * clamp(fB, 0.0, self.max_stake_pct)

        if abs(edgeA) > self.soft_cap_edge:
            stakeA *= self.soft_cap_factor
        if abs(edgeB) > self.soft_cap_edge:
            stakeB *= self.soft_cap_factor

        decision = "NO_BET"
        reason = None
        pick = None
        pick_odds = None
        pick_edge = 0.0
        stake = 0.0

        if near_floor or near_ceil:
            reason = "edge_near_clamp"
        elif edgeA >= self.min_edge and stakeA > 0:
            decision = "BET_A"
            pick = playerA_raw
            pick_odds = float(oddsA)
            pick_edge = float(edgeA)
            stake = float(stakeA)

        if reason is None and edgeB >= self.min_edge and stakeB > 0 and edgeB > pick_edge:
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
            "reason": reason,
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
