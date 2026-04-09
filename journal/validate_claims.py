#!/usr/bin/env python3
"""
External validation (light mode): verify frozen assets and optional checksums.

Full causal replication requires GPU + model; this CLI supports a reproducibility
checklist without loading NLLB.

Usage:
  python -m journal.validate_claims --manifest results/manifests/exp4_eng_Latn.json
  python -m journal.validate_claims --stimuli outputs/stimuli/kinship_pairs.json \\
      --vectors-glob 'outputs/vectors/kinship_*.pt' --light-only
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_stimuli(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": f"missing {path}"}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"ok": False, "error": "stimuli JSON must be an object"}
    n_langs = len(data)
    return {"ok": True, "path": str(path), "sha256": _sha256_file(path), "n_top_level_keys": n_langs}


def validate_vectors_glob(pattern: str) -> Dict[str, Any]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return {"ok": False, "error": f"no files match {pattern}"}
    digests = {p: _sha256_file(Path(p)) for p in paths}
    return {"ok": True, "n_files": len(paths), "sha256_by_path": digests}


def validate_config_json(path: Path) -> Dict[str, Any]:
    """Lightweight check for a frozen hyperparameter / paths snapshot."""
    if not path.exists():
        return {"ok": False, "error": f"missing {path}"}
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        return {"ok": False, "error": "config JSON must be an object"}
    suggested = (
        "alpha", "layers", "vector_method", "matching_mode",
        "INTERVENTION_LAYERS", "model_name", "domains",
    )
    keys = set(cfg.keys())
    present = [k for k in suggested if k in keys]
    return {
        "ok": True,
        "path": str(path),
        "sha256": _sha256_file(path),
        "n_keys": len(keys),
        "recognized_keys_present": present,
    }


def validate_manifest_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": f"missing {path}"}
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    required = ("schema_version", "experiment_id", "git_commit", "timestamp_utc")
    missing = [k for k in required if k not in m]
    if missing:
        return {"ok": False, "error": f"manifest missing keys: {missing}"}
    return {"ok": True, "manifest": m}


def main() -> int:
    parser = argparse.ArgumentParser(description="SACRED external validation (light)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to run_manifest JSON")
    parser.add_argument("--stimuli", type=str, default=None, help="Path to *_pairs.json")
    parser.add_argument("--vectors-glob", type=str, default=None, help="Glob for vector .pt files")
    parser.add_argument("--config", type=str, default=None, help="Frozen config JSON (hyperparameters + paths)")
    parser.add_argument("--light-only", action="store_true", help="Only checksums + JSON structure")
    parser.add_argument("--out", type=str, default=None, help="Write validation report JSON")
    args = parser.parse_args()

    report: Dict[str, Any] = {"mode": "light" if args.light_only else "full", "checks": {}}

    if args.manifest:
        report["checks"]["manifest"] = validate_manifest_file(Path(args.manifest))
    if args.stimuli:
        report["checks"]["stimuli"] = validate_stimuli(Path(args.stimuli))
    if args.vectors_glob:
        report["checks"]["vectors"] = validate_vectors_glob(args.vectors_glob)
    if args.config:
        report["checks"]["config"] = validate_config_json(Path(args.config))

    if not report["checks"]:
        parser.print_help()
        print("\nProvide at least one of --manifest, --stimuli, --vectors-glob, --config", file=sys.stderr)
        return 2

    all_ok = all(c.get("ok") for c in report["checks"].values() if isinstance(c, dict))
    report["all_ok"] = bool(all_ok)

    line = json.dumps(report, indent=2)
    print(line)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(line)
        print(f"[validate_claims] wrote {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
