#!/usr/bin/env python3
"""
Thin ablation launcher: wrong-domain vectors, optional layer override via env-style args.

Does not duplicate experiment logic — delegates to ``experiments.exp2_pivot`` /
``experiments.exp4_transfer_matrix``.

Examples:
  python -m journal.ablation_runner exp2 --domain sacred --vector-source-domain kinship --alpha 0.25
  python -m journal.ablation_runner exp4 --domain kinship --layers 12,13,14,15
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _parse_layers(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="SACRED ablation runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p2 = sub.add_parser("exp2", help="Pivot diagnosis ablation")
    p2.add_argument("--domain", default="kinship", choices=["kinship", "sacred"])
    p2.add_argument("--vector-source-domain", default=None,
                    help="Load vectors from this domain's .pt files (wrong-domain ablation)")
    p2.add_argument("--alpha", type=float, default=0.25)
    p2.add_argument("--vector-method", default="mean", choices=["mean", "pca", "both"])
    p2.add_argument("--layers", default=None, help="Comma-separated encoder layers, e.g. 10,11,12,13,14,15")
    p2.add_argument("--vectors-dir", default="outputs/vectors")
    p2.add_argument("--results-dir", default="results")
    p2.add_argument("--n-random-controls", type=int, default=20)
    p2.add_argument("--random-seed", type=int, default=42)
    p2.add_argument("--matching-mode", default="hybrid",
                    choices=["substring", "word_boundary", "token_only", "hybrid"])

    p4 = sub.add_parser("exp4", help="Transfer matrix ablation")
    p4.add_argument("--domain", default="kinship", choices=["kinship", "sacred"])
    p4.add_argument("--vector-source-domain", default=None)
    p4.add_argument("--alpha", type=float, default=0.25)
    p4.add_argument("--vector-method", default="mean", choices=["mean", "pca", "both"])
    p4.add_argument("--layers", default=None)
    p4.add_argument("--output-lang", default="eng_Latn")
    p4.add_argument("--vectors-dir", default="outputs/vectors")
    p4.add_argument("--results-dir", default="results")
    p4.add_argument("--matching-mode", default="hybrid",
                    choices=["substring", "word_boundary", "token_only", "hybrid"])

    args = parser.parse_args()
    layers = _parse_layers(getattr(args, "layers", None))

    if args.cmd == "exp2":
        from experiments.exp2_pivot import run_exp2
        run_exp2(
            domain=args.domain,
            vectors_dir=args.vectors_dir,
            layers=layers,
            alpha=args.alpha,
            results_dir=args.results_dir,
            vector_method=args.vector_method,
            n_random_controls=args.n_random_controls,
            random_seed=args.random_seed,
            vector_source_domain=args.vector_source_domain,
            matching_mode=args.matching_mode,
        )
        return 0

    if args.cmd == "exp4":
        from experiments.exp4_transfer_matrix import run_exp4
        run_exp4(
            domain=args.domain,
            vectors_dir=args.vectors_dir,
            layers=layers,
            alpha=args.alpha,
            results_dir=args.results_dir,
            vector_method=args.vector_method,
            output_lang=args.output_lang,
            vector_source_domain=args.vector_source_domain,
            matching_mode=args.matching_mode,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
