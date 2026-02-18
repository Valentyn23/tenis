"""One-command pipeline for retraining + state rebuild + quick validation.

Usage:
  python retrain_rebuild_validate.py

Requires runtime dependencies installed (pandas, xgboost, sklearn, joblib, etc.).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n=== {name} ===")
    print("$", " ".join(cmd))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(cmd, cwd=ROOT, env=merged_env)
    if proc.returncode != 0:
        raise SystemExit(f"Step failed: {name} (code={proc.returncode})")


def main() -> None:
    python = sys.executable

    # Train ATP and WTA (includes 2026 because DATA_PATH uses *.xls* in data/{MODE})
    run_step("Train ATP", [python, "train/train.py"], env={"MODE": "ATP"})
    run_step("Train WTA", [python, "train/train.py"], env={"MODE": "WTA"})

    # Warm states separately
    run_step("Warmup ATP state", [python, "wrump.py"], env={"MODE": "ATP"})
    run_step("Warmup WTA state", [python, "wrump.py"], env={"MODE": "WTA"})

    # Quick app run preflight
    run_step("App smoke (strict mode)", [python, "app.py"], env={"STRICT_MODE_MATCH": "1", "MAX_EVENTS": "10"})

    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
