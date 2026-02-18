# state_persistence.py
from pathlib import Path
import joblib

DEFAULT_STATE_PATH = Path("state") / "engine_state.pkl"


def _to_path(p) -> Path:
    if isinstance(p, Path):
        return p
    return Path(str(p))


def save_engine(engine, path=DEFAULT_STATE_PATH) -> None:
    path = _to_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(engine, path)


def load_engine(path=DEFAULT_STATE_PATH):
    path = _to_path(path)
    if not path.exists():
        return None
    return joblib.load(path)
