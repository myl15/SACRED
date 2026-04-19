"""
Experiment 5 (supplementary): post-hoc cosine deletion diagnostics.

Builds a single supplementary CSV table that compares token-matching deletion
rates (from Exp2/Exp4 artifacts) to embedding-cosine deletion rates.
"""

import argparse

from config import DEFAULT_DEVICE
from analysis.cosine_deletion_supplement import run_cosine_supplement


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Cosine similarity supplement table"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--vectors-dir", default="outputs/vectors")
    parser.add_argument("--stimuli-dir", default="outputs/stimuli")
    parser.add_argument(
        "--output-csv",
        default="results/paper/table_cosine_deletion_supplement.csv",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--cosine-change-threshold",
        type=float,
        default=0.90,
        help="A sample counts as deleted if cos(baseline, ablated) <= this value.",
    )
    parser.add_argument(
        "--anchor-presence-threshold",
        type=float,
        default=0.20,
        help="Anchor crossing threshold for concept-anchor deletion criterion.",
    )
    parser.add_argument(
        "--max-sentences-per-pair",
        type=int,
        default=0,
        help="Optional cap per pair; 0 uses all available positives.",
    )
    args = parser.parse_args()

    out = run_cosine_supplement(
        results_dir=args.results_dir,
        vectors_dir=args.vectors_dir,
        stimuli_dir=args.stimuli_dir,
        output_csv=args.output_csv,
        device=args.device,
        cosine_change_threshold=args.cosine_change_threshold,
        anchor_presence_threshold=args.anchor_presence_threshold,
        max_sentences_per_pair=args.max_sentences_per_pair,
    )
    print(f"Cosine supplement table written: {out}")


if __name__ == "__main__":
    main()

