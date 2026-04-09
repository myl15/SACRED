#!/usr/bin/env python3
"""
Hyperparameter sweep planner: writes a JSON grid and optional shell snippets.

By default only writes the plan JSON. Use ``--execute`` to run each command (GPU heavy).

Example:
  python -m journal.hyperparam_sweep --experiment exp2 --domain kinship \\
      --alphas 0.1,0.25,0.5 --vector-methods mean,pca --out results/journal/sweep_exp2.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_layers(s: Optional[str]) -> Optional[str]:
    if not s or not str(s).strip():
        return None
    return ",".join(x.strip() for x in str(s).split(",") if x.strip())


def build_grid(
    experiment: str,
    domain: str,
    alphas: List[float],
    vector_methods: List[str],
    output_langs: List[str],
    layers_arg: Optional[str],
) -> List[Dict[str, Any]]:
    rows = []
    layers_suffix = ""
    if layers_arg:
        layers_suffix = f" --layers {layers_arg}"

    if experiment == "exp2":
        for a, vm in itertools.product(alphas, vector_methods):
            cmd = (
                f"python experiments/exp2_pivot.py --domain {domain} --alpha {a} "
                f"--vector-method {vm}{layers_suffix}"
            )
            rows.append({
                "experiment": "exp2",
                "domain": domain,
                "alpha": a,
                "vector_method": vm,
                "layers": layers_arg,
                "cmd": cmd,
            })
    elif experiment == "exp4":
        for a, vm, ol in itertools.product(alphas, vector_methods, output_langs):
            cmd = (
                f"python experiments/exp4_transfer_matrix.py --domain {domain} --alpha {a} "
                f"--vector-method {vm} --output-lang {ol}{layers_suffix}"
            )
            rows.append({
                "experiment": "exp4",
                "domain": domain,
                "alpha": a,
                "vector_method": vm,
                "output_lang": ol,
                "layers": layers_arg,
                "cmd": cmd,
            })
    else:
        raise ValueError(f"unknown experiment {experiment}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep planner")
    parser.add_argument("--experiment", required=True, choices=["exp2", "exp4"])
    parser.add_argument("--domain", default="kinship")
    parser.add_argument("--alphas", default="0.25", help="Comma-separated floats")
    parser.add_argument("--vector-methods", default="mean", help="Comma-separated: mean,pca,both")
    parser.add_argument("--output-langs", default="eng_Latn", help="exp4 only, comma-separated")
    parser.add_argument(
        "--layers",
        default=None,
        help="Optional comma-separated encoder layer indices (passed to exp2/exp4 if supported)",
    )
    parser.add_argument("--out", type=str, default="results/journal/hyperparam_sweep_plan.json")
    parser.add_argument("--execute", action="store_true", help="Run each cmd (GPU heavy)")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    vms = [x.strip() for x in args.vector_methods.split(",") if x.strip()]
    ols = [x.strip() for x in args.output_langs.split(",") if x.strip()]
    layers_arg = _parse_layers(args.layers)

    grid = build_grid(args.experiment, args.domain, alphas, vms, ols, layers_arg)
    payload = {
        "n_runs": len(grid),
        "saturation_note": (
            "After each run, check exp4 mean deletion < 0.9 or reduce alpha; "
            "exp2 random-control separation. Match INTERVENTION_LAYERS to calibration."
        ),
        "n_per_concept_note": "Vary n_per_concept in exp1 (ContrastivePairGenerator) and re-run the pipeline.",
        "runs": grid,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[hyperparam_sweep] wrote {out} ({len(grid)} runs)")

    if args.execute:
        for i, row in enumerate(grid):
            print(f"--- [{i+1}/{len(grid)}] {row['cmd']}")
            ret = subprocess.call(row["cmd"], shell=True)
            if ret != 0:
                print(f"FAILED with exit {ret}", file=sys.stderr)
                return ret
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
