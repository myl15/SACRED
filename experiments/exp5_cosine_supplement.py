"""
Experiment 5: cosine concept-deletion table (domain/target anchor, gated).

Writes a primary CSV (Exp2 conditions A–D + Exp4 transfer cells) and optionally
Exp1 English validation CSV. See analysis/cosine_concept_deletion.py.
"""

import argparse

from config import DEFAULT_DEVICE
from analysis.cosine_concept_deletion import run_cosine_concept_deletion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Cosine concept-deletion (anchor-matched, gated)"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--vectors-dir", default="outputs/vectors")
    parser.add_argument("--stimuli-dir", default="outputs/stimuli")
    parser.add_argument(
        "--exp1-json-dir",
        default="outputs",
        help="Directory containing exp1_{domain}_deletion.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/paper/table_cosine_concept_deletion.csv",
    )
    parser.add_argument(
        "--validation-output-csv",
        default="results/paper/table_cosine_exp1_validation.csv",
    )
    parser.add_argument(
        "--write-intervention-divergence-csv",
        default=None,
        metavar="PATH",
        help="Optional appendix CSV: pooled baseline vs ablated cosine (intervention effect only).",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.5,
        help="Gate: score sentence only if cos(baseline_embedding, anchor) >= this.",
    )
    parser.add_argument(
        "--deletion-threshold",
        type=float,
        default=0.5,
        help="When deletion-mode=absolute: deletion if a_anchor < this.",
    )
    parser.add_argument(
        "--deletion-mode",
        choices=["absolute", "relative_drop"],
        default="absolute",
    )
    parser.add_argument(
        "--deletion-relative-margin",
        type=float,
        default=0.1,
        help="When deletion-mode=relative_drop: deletion if a_anchor < b_anchor - margin.",
    )
    parser.add_argument(
        "--low-gate-threshold",
        type=float,
        default=0.3,
        help="Mark low_gate in CSV when gate_pass_rate < this.",
    )
    parser.add_argument(
        "--divergence-tau",
        type=float,
        default=0.90,
        help="Intervention appendix: fraction of sentences with cos(baseline,ablated) <= tau.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=0,
        help="Cap sentences per pair/cell; 0 = all.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Progress logging frequency for loops (1 = every item, 10 = every 10th item).",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=8,
        help="Batch size for model.generate calls (higher is faster but uses more GPU memory).",
    )
    parser.add_argument(
        "--max-random-trials",
        type=int,
        default=0,
        help="Optional cap for Exp2 Condition D Monte Carlo trials (0 = use JSON metadata).",
    )
    parser.add_argument(
        "--validate-exp1-only",
        action="store_true",
        help="Only run Exp1 English validation and exit.",
    )
    parser.add_argument(
        "--skip-exp1-validation",
        action="store_true",
        help="Skip Exp1 validation CSV (main table still runs).",
    )
    args = parser.parse_args()

    val_path, main_path, div_path = run_cosine_concept_deletion(
        results_dir=args.results_dir,
        vectors_dir=args.vectors_dir,
        stimuli_dir=args.stimuli_dir,
        output_csv=args.output_csv,
        validation_csv=args.validation_output_csv,
        exp1_json_dir=args.exp1_json_dir,
        device=args.device,
        presence_threshold=args.presence_threshold,
        deletion_threshold=args.deletion_threshold,
        deletion_mode=args.deletion_mode,
        deletion_relative_margin=args.deletion_relative_margin,
        low_gate_threshold=args.low_gate_threshold,
        max_sentences=args.max_sentences,
        validate_exp1_only=args.validate_exp1_only,
        skip_exp1_validation=args.skip_exp1_validation,
        write_divergence_csv=args.write_intervention_divergence_csv,
        divergence_tau=args.divergence_tau,
        log_every=args.log_every,
        generation_batch_size=args.generation_batch_size,
        max_random_trials=args.max_random_trials,
    )
    if val_path:
        print(f"Exp1 validation CSV: {val_path}")
    if main_path:
        print(f"Primary cosine table: {main_path}")
    if div_path:
        print(f"Intervention divergence appendix: {div_path}")


if __name__ == "__main__":
    main()
