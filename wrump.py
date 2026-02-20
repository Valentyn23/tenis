import os
import glob
import re
import pandas as pd
from datetime import datetime

from state_engine import StateEngine
from state_persistence import save_engine
from config_shared import normalize_mode, tournament_level_from_text

# =========================================================
# CONFIG
# =========================================================
MODE = normalize_mode(os.getenv("MODE", "ATP"))
DATA_FOLDER = os.getenv("DATA_FOLDER_OVERRIDE", os.path.join("data", MODE))
EXTENSIONS = ("*.xlsx", "*.xls", "*.csv")
STATE_PATH = os.getenv("STATE_PATH", f"state/engine_state_{MODE.lower()}.pkl")


# =========================================================
# HELPERS
# =========================================================
def norm_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip()).lower()


def find_col(colmap: dict, candidates: list[str]):
    for cand in candidates:
        cand_n = norm_col(cand)
        if cand_n in colmap:
            return colmap[cand_n]
    return None


def parse_date(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and not x.strip()):
        return None

    if isinstance(x, datetime):
        return x

    try:
        if hasattr(x, "to_pydatetime"):
            return x.to_pydatetime()
    except Exception:
        pass

    if isinstance(x, (int, float)) and not pd.isna(x):
        xi = int(x)
        if 19000101 <= xi <= 21001231:
            s = str(xi)
            try:
                return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))
            except Exception:
                pass

    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def level_from_series(series_or_tier):
    return tournament_level_from_text(series_or_tier, default=1.0)


def parse_round(rnd):
    if rnd is None or (isinstance(rnd, float) and pd.isna(rnd)):
        return 1.0
    s = str(rnd).strip().upper()

    mapping = {
        "F": 6.0,
        "SF": 5.0,
        "QF": 4.0,
        "R16": 3.0,
        "4R": 3.0,
        "R32": 2.0,
        "3R": 2.0,
        "R64": 1.5,
        "2R": 1.5,
        "R128": 1.0,
        "1R": 1.0,
        "RR": 2.0,
        "Q1": 0.5,
        "Q2": 0.6,
        "Q3": 0.7,
    }
    if s in mapping:
        return mapping[s]

    try:
        return float(s)
    except Exception:
        return 1.0


def parse_surface(surface):
    if surface is None or (isinstance(surface, float) and pd.isna(surface)):
        return "Hard"
    s = str(surface).strip().capitalize()
    if s.lower() in ("hard", "h"):
        return "Hard"
    if s.lower() in ("clay", "c"):
        return "Clay"
    if s.lower() in ("grass", "g"):
        return "Grass"
    if s.lower() in ("carpet",):
        return "Carpet"
    return s


# =========================================================
# LOAD FILES
# =========================================================
def load_all_files():
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(DATA_FOLDER, "**", ext), recursive=True))

    if not files:
        raise RuntimeError(f"No data files found in folder: {DATA_FOLDER}")

    dfs = []
    for f in sorted(files):
        try:
            if f.endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)

            df["__source"] = os.path.basename(f)
            dfs.append((f, df))
            print("Loaded:", f)
        except Exception as e:
            print("Skip file:", f, e)

    return dfs


# =========================================================
# EXTRACT MATCHES
# =========================================================
def extract_matches(path: str, df: pd.DataFrame):
    if df is None or df.empty:
        return []

    colmap = {norm_col(c): c for c in df.columns}

    col_w = find_col(colmap, ["winner", "w", "player1", "p1", "player 1"])
    col_l = find_col(colmap, ["loser", "l", "player2", "p2", "player 2"])
    col_date = find_col(colmap, ["date", "match date"])
    col_surface = find_col(colmap, ["surface", "court surface"])
    col_round = find_col(colmap, ["round"])
    col_series = find_col(colmap, ["series", "tier"])

    if not col_w or not col_l or not col_date:
        print(
            f"[WARN] {os.path.basename(path)}: missing essential columns "
            f"(winner={col_w}, loser={col_l}, date={col_date}) -> 0 matches"
        )
        return []

    matches = []
    for _, r in df.iterrows():
        w = r.get(col_w)
        l = r.get(col_l)
        if pd.isna(w) or pd.isna(l):
            continue

        d = parse_date(r.get(col_date))
        if d is None:
            continue

        surface = parse_surface(r.get(col_surface, "Hard"))
        rnd = parse_round(r.get(col_round, 1))
        level = level_from_series(r.get(col_series, ""))

        matches.append(
            {
                "winner": str(w).strip(),
                "loser": str(l).strip(),
                "surface": surface,
                "round": float(rnd),
                "level": float(level),
                "date": d,
            }
        )

    return matches


def build_match_history():
    pairs = load_all_files()

    all_matches = []
    per_file = []

    for path, df in pairs:
        ms = extract_matches(path, df)
        per_file.append((path, len(ms)))
        all_matches.extend(ms)

    print("\n=== MATCHES PER FILE (top 30) ===")
    for p, n in sorted(per_file, key=lambda x: x[1], reverse=True)[:30]:
        print(f"{n:7d}  {p}")

    zeros = [p for p, n in per_file if n == 0]
    if zeros:
        print("\n[WARN] Files with 0 parsed matches (first 30):")
        for p in zeros[:30]:
            print("  ", p)

    if not all_matches:
        raise RuntimeError("No matches parsed from datasets (all files resulted in 0).")

    all_matches.sort(key=lambda x: x["date"])

    unique_players = {p for m in all_matches for p in (m["winner"], m["loser"])}
    print("\n=== DATASET SUMMARY ===")
    print(f"Matches parsed: {len(all_matches)}")
    print(f"Unique players: {len(unique_players)}")
    if unique_players:
        print(f"Avg matches per player: {(2.0 * len(all_matches)) / len(unique_players):.2f}")

    return all_matches


# =========================================================
# WARMUP
# =========================================================
def warmup_engine():
    print("\nLoading matches...")
    matches = build_match_history()
    print("\nTotal matches:", len(matches))

    engine = StateEngine(mode=MODE)

    for i, m in enumerate(matches):
        engine.update_after_match(
            winner=m["winner"],
            loser=m["loser"],
            surface=m["surface"],
            level=m["level"],
            rnd=m["round"],
            date=m["date"],
        )
        if i > 0 and i % 100000 == 0:
            print("Processed", i)

    print("Warmup complete.")
    total_player_matches = sum(st.matches for st in engine.players.values())
    inferred_matches = total_player_matches / 2.0
    print(f"Engine players: {len(engine.players)}")
    print(f"Engine inferred matches: {inferred_matches:.0f}")
    return engine


if __name__ == "__main__":
    print(f"Warmup MODE: {MODE}")
    print(f"Warmup DATA_FOLDER: {DATA_FOLDER}")
    print(f"Warmup STATE_PATH: {STATE_PATH}")
    engine = warmup_engine()

    print("\nExample player states:")
    for p in list(engine.players.keys())[:10]:
        print(p, engine.players[p].elo)

    save_engine(engine, STATE_PATH)
    print(f"\nSaved engine state to: {STATE_PATH}")
    print("File exists:", os.path.exists(STATE_PATH))
    print("Players:", len(engine.players))
