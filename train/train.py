import os
import glob
import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import joblib


random.seed(42)

# =========================================================
# CONFIG
# =========================================================
MODE = os.getenv("MODE", "WTA").strip().upper()  # "ATP" или "WTA"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_PATH = str(PROJECT_ROOT / "data" / MODE / "*.xls*")
OUT_PATH = PROJECT_ROOT / "model" / f"{MODE.lower()}_model.pkl"

from config_shared import dynamic_k as shared_dynamic_k, tournament_level_from_text

INITIAL_ELO = 1500.0

# включай, только если в проде у тебя будут odds из API
ODDS_FEATURES = False

# =========================================================
# HELPERS
# =========================================================
def safe_float(x, default=None):
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def bounded_signal_diff(x: float, scale: float = 8.0) -> float:
    return math.tanh(float(x) / float(scale))

def normalize_surface(s: str) -> str:
    s = str(s).strip().lower()
    # частые варианты в теннисных датасетах
    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"
    if "carpet" in s:
        return "Carpet"
    # indoor hard / hard / etc
    if "hard" in s:
        return "Hard"
    # если встретится что-то экзотическое — считаем hard-like
    return "Hard"

def is_indoor_from_row(row) -> int:
    # Court бывает: "Indoor", "Outdoor", иногда пусто
    c = str(row.get("Court", "")).lower()
    if "indoor" in c:
        return 1
    return 0

def get_level(row) -> float:
    # ATP: Series, WTA: Tier
    val = row.get("Series", "") if MODE == "ATP" else row.get("Tier", "")
    return tournament_level_from_text(val, default=1.0)

def round_score(r) -> float:
    r = str(r)
    # типичные обозначения
    if "R128" in r: return 0.0
    if "R64"  in r: return 1.0
    if "R32"  in r: return 2.0
    if "R16"  in r: return 3.0
    if "QF" in r or "Quarter" in r: return 4.0
    if "SF" in r or "Semi" in r: return 5.0
    if "Final" in r or r.strip() == "F": return 6.0
    return 0.0

def best_of_value(row) -> float:
    # Best of (3/5), иногда строка
    bo = safe_float(row.get("Best of", None), default=3.0)
    if bo is None:
        return 3.0
    return float(bo)

# =========================================================
# ELO
# =========================================================
def expected(r1, r2):
    return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))

def dynamic_k(matches: int, level: float, rnd: float) -> float:
    return shared_dynamic_k(matches, level, rnd, mode=MODE)

def update_elo(r1, r2, res, K):
    e = expected(r1, r2)
    r1_new = r1 + K * (res - e)
    r2_new = r2 + K * ((1.0 - res) - (1.0 - e))
    return r1_new, r2_new

def surface_transition_penalty(last_surface: Optional[str], cur_surface: str) -> float:
    if last_surface is None or last_surface == cur_surface:
        return 0.0
    # базовый штраф
    pen = -40.0
    # самый неприятный переход
    if last_surface == "Clay" and cur_surface == "Grass":
        pen = -65.0
    return pen

# =========================================================
# ONLINE STATE (players + h2h)
# =========================================================
@dataclass
class PlayerState:
    elo: float = INITIAL_ELO
    matches: int = 0
    last_date: Optional[pd.Timestamp] = None
    last_surface: Optional[str] = None

    # per-surface elo
    surf_elo: Dict[str, float] = None

    # recent results (0/1)
    recent: list = None
    streak: int = 0  # +wins in a row, -losses in a row

    # fatigue + activity (exp decay)
    fatigue: float = 0.0
    activity7: float = 0.0
    activity14: float = 0.0

    def __post_init__(self):
        if self.surf_elo is None:
            self.surf_elo = {}
        if self.recent is None:
            self.recent = []

def winrate(st: PlayerState, n: int) -> float:
    r = st.recent[-n:]
    return (sum(r) / len(r)) if r else 0.5

def decay_activity(value: float, days: int, half_life_days: float) -> float:
    # экспоненциальный распад по half-life
    if days <= 0:
        return value
    lam = math.log(2.0) / max(half_life_days, 1e-6)
    return value * math.exp(-lam * days)

def update_pre_match_decay(st: PlayerState, date: pd.Timestamp):
    if st.last_date is None:
        return
    days = (date - st.last_date).days
    if days <= 0:
        return
    # fatigue восстанавливается быстрее
    st.fatigue *= (0.65 ** days)
    # активность распадается по half-life
    st.activity7 = decay_activity(st.activity7, days, half_life_days=7.0)
    st.activity14 = decay_activity(st.activity14, days, half_life_days=14.0)

def update_post_match(st: PlayerState, won: int, date: pd.Timestamp, surface: str):
    st.matches += 1
    st.recent.append(int(won))
    st.recent = st.recent[-40:]

    # streak
    if won == 1:
        st.streak = st.streak + 1 if st.streak >= 0 else 1
    else:
        st.streak = st.streak - 1 if st.streak <= 0 else -1

    # добавляем “нагрузку” за матч
    st.fatigue += 1.0
    st.activity7 += 1.0
    st.activity14 += 1.0

    st.last_date = date
    st.last_surface = surface

# H2H хранение: wins[A,B] = сколько A выиграл у B
H2H = Dict[Tuple[str, str], int]
H2H_SURF = Dict[Tuple[str, str, str], int]

def h2h_wins(h2h: H2H, a: str, b: str) -> int:
    return h2h.get((a, b), 0)

def h2h_wins_surf(h2h_s: H2H_SURF, a: str, b: str, surf: str) -> int:
    return h2h_s.get((a, b, surf), 0)

# =========================================================
# ODDS (optional)
# =========================================================
ODDS_CANDIDATES = [
    ("AvgW", "AvgL"),
    ("MaxW", "MaxL"),
    ("PSW", "PSL"),
    ("B365W", "B365L"),
    ("CBW", "CBL"),
    ("IWW", "IWL"),
    ("SBW", "SBL"),
    ("GBW", "GBL"),
    ("EXW", "EXL"),
    ("UBW", "UBL"),
]

def pick_odds_columns(df: pd.DataFrame):
    for wcol, lcol in ODDS_CANDIDATES:
        if wcol in df.columns and lcol in df.columns:
            return wcol, lcol
    return None, None

def implied_prob(odds: float) -> float:
    # десят. коэффициент -> implied prob (без маржи-очистки)
    if odds is None or odds <= 1.0:
        return 0.5
    return 1.0 / odds

# =========================================================
# LOAD
# =========================================================
print(f"Loading {MODE} data from: {DATA_PATH}")
files = sorted(glob.glob(DATA_PATH))
if not files:
    raise FileNotFoundError(f"No files found at {DATA_PATH}")

dfs = [pd.read_excel(f) for f in files]
data = pd.concat(dfs, ignore_index=True)

data = data.dropna(subset=["Winner", "Loser", "Date", "Surface"])
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

print("Matches:", len(data))

odds_w_col, odds_l_col = pick_odds_columns(data)
if ODDS_FEATURES:
    if odds_w_col is None:
        raise RuntimeError("ODDS_FEATURES=True but no odds columns found in your data.")
    print(f"Using odds columns: {odds_w_col}, {odds_l_col}")
else:
    print("ODDS_FEATURES=False (odds are not used)")

# =========================================================
# SPLIT BY TIME (matches)
# =========================================================
n = len(data)
split1 = int(n * 0.70)
split2 = int(n * 0.85)

# =========================================================
# STATE
# =========================================================
players: Dict[str, PlayerState] = defaultdict(PlayerState)
h2h: H2H = {}
h2h_surf: H2H_SURF = {}

# =========================================================
# FEATURE BUILDING
# =========================================================
train_rows, valid_rows, test_rows = [], [], []

def surface_onehot(surf: str):
    return {
        "is_hard": 1.0 if surf == "Hard" else 0.0,
        "is_clay": 1.0 if surf == "Clay" else 0.0,
        "is_grass": 1.0 if surf == "Grass" else 0.0,
        "is_carpet": 1.0 if surf == "Carpet" else 0.0,
    }

print("Building features (online, leakage-safe)...")

for i, row in data.iterrows():
    winner = row["Winner"]
    loser = row["Loser"]
    date: pd.Timestamp = row["Date"]

    surf = normalize_surface(row["Surface"])
    indoor = float(is_indoor_from_row(row))
    level = float(get_level(row))
    rnd = float(round_score(row.get("Round", "")))
    bo = float(best_of_value(row))

    # обновляем decay ДО расчёта фич (это pre-match состояние)
    W = players[winner]
    L = players[loser]
    update_pre_match_decay(W, date)
    update_pre_match_decay(L, date)

    # случайная ориентация A/B как в проде
    if random.random() < 0.5:
        A_name, B_name = winner, loser
        A_is_winner = 1.0
    else:
        A_name, B_name = loser, winner
        A_is_winner = 0.0

    A = players[A_name]
    B = players[B_name]
    # важно: decay уже применён на обоих

    # penalties (зависят от игрока -> diff)
    A_pen = surface_transition_penalty(A.last_surface, surf)
    B_pen = surface_transition_penalty(B.last_surface, surf)

    # surface elo init
    A_selo = A.surf_elo.get(surf, INITIAL_ELO)
    B_selo = B.surf_elo.get(surf, INITIAL_ELO)

    # rest days (diff)
    def rest_days(st: PlayerState) -> float:
        if st.last_date is None:
            return 30.0
        return float((date - st.last_date).days)

    # H2H (online)
    h2h_ab = float(h2h_wins(h2h, A_name, B_name) - h2h_wins(h2h, B_name, A_name))
    h2h_ab_s = float(h2h_wins_surf(h2h_surf, A_name, B_name, surf) - h2h_wins_surf(h2h_surf, B_name, A_name, surf))

    # diff features (A-B)
    diff = {
        "elo_diff": (A.elo + A_pen) - (B.elo + B_pen),
        "surface_elo_diff": A_selo - B_selo,
        "winrate10_diff": winrate(A, 10) - winrate(B, 10),
        "winrate30_diff": winrate(A, 30) - winrate(B, 30),
        "streak_diff": float(A.streak - B.streak),
        "fatigue_diff": float(bounded_signal_diff(A.fatigue - B.fatigue, scale=6.0)),
        "activity7_diff": float(bounded_signal_diff(A.activity7 - B.activity7, scale=8.0)),
        "activity14_diff": float(bounded_signal_diff(A.activity14 - B.activity14, scale=10.0)),
        "rest_days_diff": rest_days(A) - rest_days(B),
        "matches_diff": float(A.matches - B.matches),
        "surface_penalty_diff": float(A_pen - B_pen),
        "h2h_diff": h2h_ab,
        "h2h_surface_diff": h2h_ab_s,
    }

    # context features (не зависят от ориентации)
    ctx = {
        "round": rnd,
        "level": level,
        "best_of": bo,
        "indoor": indoor,
        **surface_onehot(surf),
    }

    feats = {**diff, **ctx}

    # odds features (если включено)
    if ODDS_FEATURES:
        ow = safe_float(row.get(odds_w_col, None), default=None)
        ol = safe_float(row.get(odds_l_col, None), default=None)

        # odds в файле идут как Winner/Loser odds, а у нас A/B могут быть swapped
        # определяем odds для A и B корректно:
        if A_name == winner:
            odds_A, odds_B = ow, ol
        else:
            odds_A, odds_B = ol, ow

        ip_A = implied_prob(odds_A)
        ip_B = implied_prob(odds_B)

        # diff-like odds features (A-B)
        feats["implied_prob_diff"] = float(ip_A - ip_B)
        feats["log_odds_diff"] = float(math.log(max(odds_B, 1.01)) - math.log(max(odds_A, 1.01)))
        feats["odds_A"] = float(odds_A if odds_A is not None else 0.0)
        feats["odds_B"] = float(odds_B if odds_B is not None else 0.0)

    out = {**feats, "target": float(A_is_winner)}

    if i < split1:
        train_rows.append(out)
    elif i < split2:
        valid_rows.append(out)
    else:
        test_rows.append(out)

    # =====================================================
    # UPDATE TRUE MATCH RESULT (winner/loser)
    # =====================================================
    # Elo updates всегда по истинному исходу (winner beats loser)
    K = dynamic_k(W.matches, level, rnd)
    W.elo, L.elo = update_elo(W.elo, L.elo, 1.0, K)

    W_selo = W.surf_elo.get(surf, INITIAL_ELO)
    L_selo = L.surf_elo.get(surf, INITIAL_ELO)
    W_selo, L_selo = update_elo(W_selo, L_selo, 1.0, K)
    W.surf_elo[surf] = W_selo
    L.surf_elo[surf] = L_selo

    # post-match updates (form, fatigue, etc)
    update_post_match(W, 1, date, surf)
    update_post_match(L, 0, date, surf)

    # update H2H
    h2h[(winner, loser)] = h2h.get((winner, loser), 0) + 1
    h2h_surf[(winner, loser, surf)] = h2h_surf.get((winner, loser, surf), 0) + 1

# =========================================================
# DATAFRAMES
# =========================================================
train_df = pd.DataFrame(train_rows).astype(float)
valid_df = pd.DataFrame(valid_rows).astype(float)
test_df = pd.DataFrame(test_rows).astype(float)

feature_names = [c for c in train_df.columns if c != "target"]

X_train, y_train = train_df[feature_names], train_df["target"]
X_valid, y_valid = valid_df[feature_names], valid_df["target"]
X_test, y_test = test_df[feature_names], test_df["target"]

print("Train:", X_train.shape, "Valid:", X_valid.shape, "Test:", X_test.shape)

# =========================================================
# TRAIN (stronger params + early stopping)
# =========================================================
model = XGBClassifier(
    n_estimators=5000,
    max_depth=5 if MODE == "WTA" else 6,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_lambda=1.0,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=200   # ← ПЕРЕНЕСЛИ СЮДА
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True
)


# =========================================================
# RAW METRICS
# =========================================================
raw = model.predict_proba(X_test)[:, 1]
print("\nRAW METRICS")
print("AUC:", roc_auc_score(y_test, raw))
print("LogLoss:", log_loss(y_test, raw))
print("Brier:", brier_score_loss(y_test, raw))

# =========================================================
# CALIBRATION (isotonic, sklearn>=1.4 safe)
# =========================================================
print("\nCalibrating probabilities (isotonic)...")
valid_raw = model.predict_proba(X_valid)[:, 1]

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(valid_raw, y_valid)

def predict_calibrated(X):
    r = model.predict_proba(X)[:, 1]
    return iso.transform(r)

pred = predict_calibrated(X_test)

print("\nFINAL METRICS (calibrated)")
print("AUC:", roc_auc_score(y_test, pred))
print("LogLoss:", log_loss(y_test, pred))
print("Brier:", brier_score_loss(y_test, pred))

# =========================================================
# SAVE
# =========================================================
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"model": model, "calibrator": iso, "feature_names": feature_names, "mode": MODE, "odds_features": ODDS_FEATURES},
    str(OUT_PATH)
)
print("\nSaved:", OUT_PATH.resolve())
