"""
Run manifest for reproducible paper artifacts.

Each experiment run should write a JSON manifest with git commit, time, and config
so external reviewers can tie figures to exact code and hyperparameters.
"""

from __future__ import annotations

import json
import hashlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1.0"


def get_git_commit(short: bool = True) -> str:
    """Return current git HEAD hash, or 'unknown' if not in a git repo."""
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out[:12] if short else out
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def get_git_dirty() -> Optional[bool]:
    """Return whether working tree is dirty; None if git unavailable."""
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return bool(out)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def sha256_file(path: str) -> str:
    """Hash a file for artifact traceability."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    experiment_id: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a manifest dict suitable for JSON serialization.

    Args:
        experiment_id: e.g. 'exp1_kinship', 'exp2_pivot'
        extra: Config snapshot (alpha, layers, vector_method, seeds, paths, etc.)
    """
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "git_commit": get_git_commit(short=True),
        "git_commit_full": get_git_commit(short=False),
        "git_dirty": get_git_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_CACHE",
        "HF_HOME",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "FLORES_DATASET",
    )
    payload["environment"] = {k: os.environ.get(k) for k in env_keys}
    if extra:
        payload["config"] = extra
    return payload


def write_manifest(path: Path, experiment_id: str, **config: Any) -> None:
    """Write manifest JSON to ``path``; creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(experiment_id, extra=config if config else None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {path}")


def merge_manifest_into_metadata(metadata: Dict[str, Any], experiment_id: str, **config: Any) -> Dict[str, Any]:
    """Return a copy of metadata with ``run_manifest`` attached."""
    out = dict(metadata)
    out["run_manifest"] = build_manifest(experiment_id, extra=config if config else None)
    return out
