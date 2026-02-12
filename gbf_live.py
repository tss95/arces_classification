#!/usr/bin/env python3
"""Deprecated local live entrypoint. Delegates to ml_array_data_classification."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def resolve_inference_repo() -> Path:
    candidates = []

    env_repo = os.environ.get("INFERENCE_REPO_DIR")
    if env_repo:
        candidates.append(Path(env_repo))

    candidates.append(Path("/tf/inference/ml_array_data_classification"))

    # Local development default: sibling repo in the workspace.
    candidates.append(Path(__file__).resolve().parent.parent / "ml_array_data_classification")

    project_dir = os.environ.get("PROJECT_DIR")
    if project_dir:
        candidates.append(Path(project_dir).resolve().parent / "ml_array_data_classification")

    for candidate in candidates:
        if (candidate / "inference.py").exists():
            return candidate

    candidate_list = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate ml_array_data_classification inference repo. "
        f"Tried: {candidate_list}"
    )


def main() -> int:
    script_name = Path(__file__).name
    print(
        f"{script_name} is deprecated in arces_classification. "
        "Delegating to ml_array_data_classification/inference.py instead.",
        file=sys.stderr,
    )

    repo_dir = resolve_inference_repo()
    script_path = repo_dir / "inference.py"

    env = os.environ.copy()
    env.setdefault("INFERENCE_REPO_DIR", str(repo_dir))

    cmd = [sys.executable, str(script_path), *sys.argv[1:]]
    return subprocess.call(cmd, cwd=str(repo_dir), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
