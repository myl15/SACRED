"""Experiment 5: token-level max-cosine concept deletion."""

import argparse

from config import DEFAULT_DEVICE
from analysis.cosine_concept_deletion import run_token_max_cosine


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
        default="results/paper/table_cosine_token_max.csv",
    )
    parser.add_argument(
        "--calibration-output-csv",
        default="results/paper/table_cosine_token_max_calibration.csv",
    )
    parser.add_argument(
        "--coherence-output-csv",
        default="results/paper/table_cosine_token_max_coherence.csv",
    )
    parser.add_argument(
        "--calibration-debug-output-csv",
        default="results/paper/table_cosine_token_max_calibration_debug.csv",
    )
    parser.add_argument(
        "--calibration-debug-summary-output-csv",
        default="results/paper/table_cosine_token_max_calibration_debug_summary.csv",
    )
    parser.add_argument(
        "--pivot-comparison-output-csv",
        default="results/paper/table_cosine_token_max_pivot_comparison.csv",
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
        default=0.4,
        help="Deletion decision threshold on a_max for gate-passing pairs.",
    )
    parser.add_argument(
        "--low-gate-threshold",
        type=float,
        default=0.3,
        help="Mark low_gate in CSV when gate_pass_rate < this.",
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
        "--random-trials",
        type=int,
        default=20,
        help="Number of Monte Carlo trials for Exp2 condition D.",
    )
    parser.add_argument(
        "--min-valid-random-trials",
        type=int,
        default=10,
        help="Minimum valid D_random trials required to aggregate condition D.",
    )
    parser.add_argument(
        "--validate-exp1-only",
        action="store_true",
        help="Only run calibration/coherence checks and exit.",
    )
    parser.add_argument(
        "--allow-calibration-fail",
        action="store_true",
        help="Proceed with full run even if calibration checks fail.",
    )
    parser.add_argument(
        "--debug-calibration",
        action="store_true",
        help="Write per-sentence max-token diagnostics for calibration to debug threshold issues.",
    )
    parser.add_argument(
        "--debug-sentence-cap",
        type=int,
        default=100,
        help="Max number of calibration sentences per concept to include in debug output.",
    )
    parser.add_argument(
        "--blocked-token-ids",
        default="",
        help="Comma-separated token IDs to exclude from token-max scoring.",
    )
    parser.add_argument(
        "--blocked-token-strings",
        default="",
        help="Comma-separated tokenizer token strings to exclude (e.g. ▁The,▁the).",
    )
    parser.add_argument(
        "--disable-content-token-filter",
        action="store_true",
        help="Disable content-token gating and score all non-special/non-blocked tokens.",
    )
    args = parser.parse_args()

    out = run_token_max_cosine(
        results_dir=args.results_dir,
        vectors_dir=args.vectors_dir,
        stimuli_dir=args.stimuli_dir,
        output_csv=args.output_csv,
        calibration_csv=args.calibration_output_csv,
        coherence_csv=args.coherence_output_csv,
        calibration_debug_csv=args.calibration_debug_output_csv,
        calibration_debug_summary_csv=args.calibration_debug_summary_output_csv,
        pivot_comparison_csv=args.pivot_comparison_output_csv,
        exp1_json_dir=args.exp1_json_dir,
        device=args.device,
        presence_threshold=args.presence_threshold,
        deletion_threshold=args.deletion_threshold,
        low_gate_threshold=args.low_gate_threshold,
        max_sentences=args.max_sentences,
        calibration_only=args.validate_exp1_only,
        require_calibration_pass=not args.allow_calibration_fail,
        log_every=args.log_every,
        generation_batch_size=args.generation_batch_size,
        random_trials=args.random_trials,
        min_valid_random_trials=args.min_valid_random_trials,
        debug_calibration=args.debug_calibration,
        debug_sentence_cap=args.debug_sentence_cap,
        blocked_token_ids_csv=args.blocked_token_ids,
        blocked_token_strs_csv=args.blocked_token_strings,
        require_content_tokens=(not args.disable_content_token_filter),
    )
    print(f"Calibration passed: {out['calibration_passed']}")
    if out["coherence_csv"]:
        print(f"Coherence CSV: {out['coherence_csv']}")
    if out["calibration_csv"]:
        print(f"Calibration CSV: {out['calibration_csv']}")
    if out.get("calibration_debug_csv"):
        print(f"Calibration debug CSV: {out['calibration_debug_csv']}")
    if out.get("calibration_debug_summary_csv"):
        print(f"Calibration debug summary CSV: {out['calibration_debug_summary_csv']}")
    if out["main_csv"]:
        print(f"Primary token-max cosine CSV: {out['main_csv']}")
    if out["pivot_csv"]:
        print(f"Pivot comparison CSV: {out['pivot_csv']}")


if __name__ == "__main__":
    main()
