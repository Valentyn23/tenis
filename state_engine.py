# state_engine.py
# Online state engine for tennis (ATP/WTA)
# Elo + surface Elo + form/fatigue/activity + H2H
# FIX: pickle/joblib safe (NO local lambda in defaultdict)

import math
from collections import defaultdict
from datetime import datetime
from typing import Optional

from config_shared import dynamic_k as shared_dynamic_k, normalize_mode


# =========================================================
# CONFIG
# =========================================================
INITIAL_ELO = 1500.0


# IMPORTANT: must be a TOP-LEVEL function (pickle/joblib safe)
def initial_elo() -> float:
    return INITIAL_ELO


# =========================================================
# HELPERS
# =========================================================
def expected(r1: float, r2: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))


def update_elo(r1: float, r2: float, res: float, K: float):
    e = expected(r1, r2)
    return (
        r1 + K * (res - e),
        r2 + K * ((1.0 - res) - (1.0 - e)),
    )


def dynamic_k(matches: int, level: float, rnd: float) -> float:
    """
    Dynamic K:
      - higher for newcomers
      - scaled by tournament level & round
    """
    K = 32.0

    if matches < 30:
        K *= 1.6
    elif matches < 100:
        K *= 1.2
    else:
        K *= 0.85

    K *= (1.0 + float(level) * 0.15)
    K *= (1.0 + float(rnd) * 0.05)

    return float(K)


def decay(value: float, days: int, half_life: float) -> float:
    if days <= 0:
        return value
    lam = math.log(2.0) / float(half_life)
    return float(value) * math.exp(-lam * float(days))


def surface_transition_penalty(last_surface: Optional[str], current_surface: str) -> float:
    """
    Penalty to BASE Elo when switching surfaces (simple heuristic).
    """
    if last_surface is None or last_surface == current_surface:
        return 0.0
    if last_surface == "Clay" and current_surface == "Grass":
        return -65.0
    return -40.0


def parse_date(x) -> datetime:
    if isinstance(x, datetime):
        return x
    return datetime.fromisoformat(str(x))


# =========================================================
# PLAYER STATE
# =========================================================
class PlayerState:
    def __init__(self):
        self.elo: float = INITIAL_ELO
        self.matches: int = 0
        self.last_date: Optional[datetime] = None
        self.last_surface: Optional[str] = None

        # FIXED: default factory must be pickle-safe (top-level func)
        self.surf_elo = defaultdict(initial_elo)  # surface -> Elo

        # recent results (1/0)
        self.recent: list[int] = []
        self.streak: int = 0

        # simple fatigue/activity counters (with decay)
        self.fatigue: float = 0.0
        self.activity7: float = 0.0
        self.activity14: float = 0.0


# =========================================================
# ENGINE
# =========================================================
class StateEngine:
    def __init__(self, mode: str = "ATP"):
        self.mode = normalize_mode(mode)
        self.players = defaultdict(PlayerState)
        self.h2h = defaultdict(int)           # (A,B) -> wins of A vs B
        self.h2h_surface = defaultdict(int)   # (A,B,surface) -> wins of A vs B on surface

    # -----------------------------------------------------
    # DECAY BEFORE MATCH
    # -----------------------------------------------------
    def _apply_decay(self, st: PlayerState, date: datetime) -> None:
        if st.last_date is None:
            return

        days = (date - st.last_date).days
        if days <= 0:
            return

        st.fatigue *= (0.65 ** days)
        st.activity7 = decay(st.activity7, days, 7.0)
        st.activity14 = decay(st.activity14, days, 14.0)

    # -----------------------------------------------------
    # WINRATE
    # -----------------------------------------------------
    @staticmethod
    def _winrate(st: PlayerState, n: int) -> float:
        r = st.recent[-n:]
        return (sum(r) / len(r)) if r else 0.5

    # -----------------------------------------------------
    # REST DAYS
    # -----------------------------------------------------
    @staticmethod
    def _rest_days(st: PlayerState, date: datetime) -> int:
        if st.last_date is None:
            return 30
        return max(0, (date - st.last_date).days)

    # -----------------------------------------------------
    # BUILD FEATURES (A vs B orientation)
    # -----------------------------------------------------
    def build_features(
        self,
        A: str,
        B: str,
        surface: str,
        level: float,
        rnd: float,
        best_of: float,
        indoor: float,
        date,
    ) -> dict:
        A = str(A)
        B = str(B)
        surface = str(surface)

        date_dt = parse_date(date)

        # IMPORTANT: avoid mutating player store during inference for unseen players.
        # Using defaultdict indexing here would create new PlayerState entries.
        pA = self.players.get(A)
        pB = self.players.get(B)
        if pA is None:
            pA = PlayerState()
        if pB is None:
            pB = PlayerState()

        # decay before feature extraction
        self._apply_decay(pA, date_dt)
        self._apply_decay(pB, date_dt)

        # surface switching penalty (only to base Elo)
        penA = surface_transition_penalty(pA.last_surface, surface)
        penB = surface_transition_penalty(pB.last_surface, surface)

        feats = {
            # skill
            "elo_diff": (pA.elo + penA) - (pB.elo + penB),
            "surface_elo_diff": pA.surf_elo[surface] - pB.surf_elo[surface],

            # form
            "winrate10_diff": self._winrate(pA, 10) - self._winrate(pB, 10),
            "winrate30_diff": self._winrate(pA, 30) - self._winrate(pB, 30),
            "streak_diff": float(pA.streak - pB.streak),

            # fatigue/activity
            "fatigue_diff": float(pA.fatigue - pB.fatigue),
            "activity7_diff": float(pA.activity7 - pB.activity7),
            "activity14_diff": float(pA.activity14 - pB.activity14),

            # experience/rest
            "rest_days_diff": float(self._rest_days(pA, date_dt) - self._rest_days(pB, date_dt)),
            "matches_diff": float(pA.matches - pB.matches),

            # switching signal
            "surface_penalty_diff": float(penA - penB),

            # H2H
            "h2h_diff": float(self.h2h.get((A, B), 0) - self.h2h.get((B, A), 0)),
            "h2h_surface_diff": float(
                self.h2h_surface.get((A, B, surface), 0) - self.h2h_surface.get((B, A, surface), 0)
            ),

            # context
            "round": float(rnd),
            "level": float(level),
            "best_of": float(best_of),
            "indoor": float(indoor),

            # surface one-hot
            "is_hard": 1.0 if surface == "Hard" else 0.0,
            "is_clay": 1.0 if surface == "Clay" else 0.0,
            "is_grass": 1.0 if surface == "Grass" else 0.0,
            "is_carpet": 1.0 if surface == "Carpet" else 0.0,
        }

        return feats

    # -----------------------------------------------------
    # UPDATE AFTER MATCH
    # -----------------------------------------------------
    def update_after_match(
        self,
        winner: str,
        loser: str,
        surface: str,
        level: float,
        rnd: float,
        date,
    ) -> None:
        winner = str(winner)
        loser = str(loser)
        surface = str(surface)

        date_dt = parse_date(date)

        W = self.players[winner]
        L = self.players[loser]

        K = shared_dynamic_k(W.matches, level, rnd, mode=self.mode)

        # base Elo update
        W.elo, L.elo = update_elo(W.elo, L.elo, 1.0, K)

        # surface Elo update
        Ws = W.surf_elo[surface]
        Ls = L.surf_elo[surface]
        Ws, Ls = update_elo(Ws, Ls, 1.0, K)
        W.surf_elo[surface] = Ws
        L.surf_elo[surface] = Ls

        # matches
        W.matches += 1
        L.matches += 1

        # recent results
        W.recent.append(1)
        L.recent.append(0)
        W.recent = W.recent[-40:]
        L.recent = L.recent[-40:]

        # streak
        W.streak = (W.streak + 1) if W.streak >= 0 else 1
        L.streak = (L.streak - 1) if L.streak <= 0 else -1

        # fatigue/activity increments
        W.fatigue += 1.0
        L.fatigue += 1.0

        W.activity7 += 1.0
        W.activity14 += 1.0
        L.activity7 += 1.0
        L.activity14 += 1.0

        # last date/surface
        W.last_date = date_dt
        L.last_date = date_dt
        W.last_surface = surface
        L.last_surface = surface

        # H2H
        self.h2h[(winner, loser)] += 1
        self.h2h_surface[(winner, loser, surface)] += 1
