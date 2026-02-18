from __future__ import annotations

from typing import Optional


def normalize_mode(mode: str) -> str:
    m = (mode or "ATP").strip().upper()
    return "WTA" if m == "WTA" else "ATP"


def dynamic_k(matches: int, level: float, rnd: float, mode: str = "ATP") -> float:
    """Shared Elo K-factor logic for both training and runtime engine."""
    m = normalize_mode(mode)
    k = 32.0 if m == "ATP" else 24.0

    if matches < 30:
        k *= 1.6
    elif matches < 100:
        k *= 1.2
    else:
        k *= 0.85

    k *= (1.0 + float(level) * 0.15)
    k *= (1.0 + float(rnd) * 0.05)
    return float(k)


def tournament_level_from_text(text: Optional[str], default: float = 1.0) -> float:
    s = "" if text is None else str(text).strip().lower()
    if not s:
        return float(default)

    if "grand" in s or "slam" in s:
        return 3.0
    if "masters" in s or "1000" in s:
        return 2.0
    if "500" in s or "premier" in s:
        return 1.7
    if "250" in s or "international" in s:
        return 1.3
    return float(default)


def infer_level_from_sport_key(sport_key: Optional[str], default: float = 1.0) -> tuple[float, bool]:
    """Infer approximate tournament level from Odds API sport_key. Returns (level, used_fallback)."""
    s = "" if sport_key is None else str(sport_key).strip().lower()
    if not s:
        return float(default), True

    # Known patterns in TheOddsAPI tennis sport keys
    if "grand_slam" in s or "grandslam" in s:
        return 3.0, False
    if "atp_1000" in s or "wta_1000" in s or "masters" in s:
        return 2.0, False
    if "atp_500" in s or "wta_500" in s:
        return 1.7, False
    if "atp_250" in s or "wta_250" in s:
        return 1.3, False

    # fallback: try generic text parser
    lvl = tournament_level_from_text(s, default=default)
    used_fallback = abs(lvl - float(default)) < 1e-12
    return lvl, used_fallback
