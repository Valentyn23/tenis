from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSettings:
    regions: str
    max_events: int
    default_surface: str
    default_level: float
    default_round: float
    default_best_of: float
    default_indoor: float

    bankroll: float
    currency: str
    max_stake_pct: float
    kelly_fraction: float
    min_edge: float

    prob_floor: float
    prob_ceil: float
    max_overround: float
    strict_mode_match: bool

    soft_cap_edge: float
    soft_cap_factor: float

    state_path_atp: str
    state_path_wta: str

    report_dir: str
    save_report: bool
    bets_ledger_path: str
    print_top_n: int

    gemini_pick_opinion: bool
    use_calibration: bool

    risk_profile: str

    def summary(self) -> str:
        return (
            f"regions={self.regions} max_events={self.max_events} bankroll={self.bankroll} "
            f"currency={self.currency} "
            f"risk_profile={self.risk_profile} min_edge={self.min_edge} max_stake_pct={self.max_stake_pct} "
            f"kelly_fraction={self.kelly_fraction} prob=[{self.prob_floor},{self.prob_ceil}] "
            f"max_overround={self.max_overround} strict_mode_match={self.strict_mode_match}"
            f" use_calibration={self.use_calibration}"
        )


PROFILE_DEFAULTS = {
    "conservative": {
        "max_stake_pct": 0.015,
        "kelly_fraction": 0.25,
        "min_edge": 0.035,
        "prob_floor": 0.10,
        "prob_ceil": 0.90,
        "max_overround": 1.05,
        "soft_cap_edge": 0.20,
        "soft_cap_factor": 0.50,
    },
    "balanced": {
        "max_stake_pct": 0.02,
        "kelly_fraction": 0.35,
        "min_edge": 0.03,
        "prob_floor": 0.08,
        "prob_ceil": 0.92,
        "max_overround": 1.06,
        "soft_cap_edge": 0.25,
        "soft_cap_factor": 0.60,
    },
    "aggressive": {
        "max_stake_pct": 0.03,
        "kelly_fraction": 0.50,
        "min_edge": 0.02,
        "prob_floor": 0.05,
        "prob_ceil": 0.95,
        "max_overround": 1.08,
        "soft_cap_edge": 0.30,
        "soft_cap_factor": 0.75,
    },
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_settings() -> RuntimeSettings:
    profile = os.getenv("RISK_PROFILE", "balanced").strip().lower()
    defaults = PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS["balanced"])

    return RuntimeSettings(
        regions=os.getenv("ODDS_REGIONS", "eu"),
        max_events=int(os.getenv("MAX_EVENTS", "30")),
        default_surface=os.getenv("DEFAULT_SURFACE", "Hard"),
        default_level=float(os.getenv("DEFAULT_LEVEL", "1")),
        default_round=float(os.getenv("DEFAULT_ROUND", "1")),
        default_best_of=float(os.getenv("DEFAULT_BEST_OF", "3")),
        default_indoor=float(os.getenv("DEFAULT_INDOOR", "0")),
        bankroll=float(os.getenv("BANKROLL", "1000")),
        currency=os.getenv("CURRENCY", "UAH").strip().upper(),
        max_stake_pct=float(os.getenv("MAX_STAKE_PCT", str(defaults["max_stake_pct"]))),
        kelly_fraction=float(os.getenv("KELLY_FRACTION", str(defaults["kelly_fraction"]))),
        min_edge=float(os.getenv("MIN_EDGE", str(defaults["min_edge"]))),
        prob_floor=float(os.getenv("PROB_FLOOR", str(defaults["prob_floor"]))),
        prob_ceil=float(os.getenv("PROB_CEIL", str(defaults["prob_ceil"]))),
        max_overround=float(os.getenv("MAX_OVERROUND", str(defaults["max_overround"]))),
        strict_mode_match=_env_bool("STRICT_MODE_MATCH", True),
        soft_cap_edge=float(os.getenv("SOFT_CAP_EDGE", str(defaults["soft_cap_edge"]))),
        soft_cap_factor=float(os.getenv("SOFT_CAP_FACTOR", str(defaults["soft_cap_factor"]))),
        state_path_atp=os.getenv("STATE_PATH_ATP", "state/engine_state_atp.pkl"),
        state_path_wta=os.getenv("STATE_PATH_WTA", "state/engine_state_wta.pkl"),
        report_dir=os.getenv("REPORT_DIR", "reports"),
        save_report=_env_bool("SAVE_REPORT", True),
        bets_ledger_path=os.getenv("BETS_LEDGER_PATH", "bets/ledger.csv"),
        print_top_n=int(os.getenv("PRINT_TOP_N", "25")),
        gemini_pick_opinion=_env_bool("GEMINI_PICK_OPINION", False),
        use_calibration=_env_bool("USE_CALIBRATION", True),
        risk_profile=profile,
    )
