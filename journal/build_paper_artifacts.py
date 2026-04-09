"""
Deterministic paper-artifact builder.

Builds manuscript-facing CSV/JSON tables from experiment JSON outputs so all
numbers in the paper can be traced to machine-readable artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from journal.run_manifest import build_manifest, sha256_file


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _exp2_rows(path: Path) -> List[Dict[str, str]]:
    data = _load_json(path)
    rows: List[Dict[str, str]] = []
    by_pair = data.get("results_by_pair", {})
    for pair_str, row in by_pair.items():
        rows.append(
            {
                "source_target": pair_str,
                "pivot_index": str(row.get("pivot_index")),
                "pivot_index_continuous": str(row.get("pivot_index_continuous")),
                "baseline_deletion": str(row.get("baseline", {}).get("deletion_rate")),
                "english_deletion": str(row.get("condition_C_english", {}).get("deletion_rate")),
                "random_deletion": str(row.get("condition_D_random", {}).get("deletion_rate")),
            }
        )
    return rows


def _exp4_rows(path: Path) -> List[Dict[str, str]]:
    data = _load_json(path)
    summary = data.get("summary", {})
    return [
        {
            "domain": str(data.get("domain")),
            "vector_method": str(data.get("vector_method")),
            "output_lang": str(data.get("output_lang")),
            "mean_off_diagonal_deletion": str(summary.get("mean_off_diagonal_deletion")),
            "best_transfer_pair": str(summary.get("best_transfer_pair")),
            "best_transfer_rate": str(summary.get("best_transfer_rate")),
            "english_hub_score": str(summary.get("english_hub_score")),
            "english_hub_absolute_mean": str(summary.get("english_hub_absolute_mean")),
            "non_english_absolute_mean": str(summary.get("non_english_absolute_mean")),
            "off_diagonal_ceiling_rate": str(summary.get("off_diagonal_ceiling_rate")),
        }
    ]


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build(results_dir: str = "results") -> Path:
    root = Path(results_dir)
    out_dir = root / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp2_paths = sorted((root / "json").glob("exp2_pivot_*.json"))
    exp4_paths = sorted(root.glob("*/json/exp4_transfer_summary_*.json"))

    exp2_rows: List[Dict[str, str]] = []
    for p in exp2_paths:
        exp2_rows.extend(_exp2_rows(p))
    exp4_rows: List[Dict[str, str]] = []
    for p in exp4_paths:
        exp4_rows.extend(_exp4_rows(p))

    exp2_csv = out_dir / "table_exp2_pivot_pairs.csv"
    exp4_csv = out_dir / "table_exp4_transfer_summary.csv"
    _write_csv(exp2_csv, exp2_rows)
    _write_csv(exp4_csv, exp4_rows)

    index = {
        "run_manifest": build_manifest(
            "paper_artifact_build",
            extra={
                "results_dir": results_dir,
                "exp2_json_count": len(exp2_paths),
                "exp4_json_count": len(exp4_paths),
            },
        ),
        "artifacts": {
            "table_exp2_pivot_pairs.csv": {
                "path": str(exp2_csv),
                "sha256": sha256_file(str(exp2_csv)) if exp2_csv.exists() else None,
            },
            "table_exp4_transfer_summary.csv": {
                "path": str(exp4_csv),
                "sha256": sha256_file(str(exp4_csv)) if exp4_csv.exists() else None,
            },
        },
    }
    out_index = out_dir / "artifact_index.json"
    with open(out_index, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return out_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build paper tables from experiment JSON.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    out = build(results_dir=args.results_dir)
    print(f"Paper artifacts written: {out}")
