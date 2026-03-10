"""
Download and cache all models and datasets needed for SACRED experiments.

Run this script on a login node (internet access required) before submitting
any SLURM jobs. Compute nodes run in fully offline mode.

Usage:
    python scripts/download_models.py [--cache-dir .hf_cache] [--skip-flores]
                                      [--hf-token <token>]

The HuggingFace token is required for gated datasets (e.g. flores_plus).
You can also set the HF_TOKEN environment variable instead of passing --hf-token.
Get your token at: https://huggingface.co/settings/tokens
"""

import argparse
import os
import sys
from pathlib import Path


def download_nllb(model_name: str, cache_dir: str):
    print(f"\n[1/2] Downloading NLLB model: {model_name}")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("  Downloading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"  Tokenizer vocab size: {tok.vocab_size}")

    print("  Downloading model weights (this may take several minutes)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {n_params:.0f}M")
    print(f"  Saved to: {cache_dir}")


def download_flores(cache_dir: str, token: str = None):
    print(f"\n[2/2] Downloading FLORES+ dataset (for Experiment 3)")
    try:
        from datasets import load_dataset

        # openlanguagedata/flores_plus is a Parquet-based mirror of FLORES+
        # that works with current datasets versions (no dataset scripts needed).
        # The config name is the language code; devtest split has 1012 sentences.
        # A HuggingFace token is required — accept the dataset terms at:
        #   https://huggingface.co/datasets/openlanguagedata/flores_plus
        if not token:
            print("  WARNING: no HuggingFace token provided (--hf-token or HF_TOKEN).")
            print("  If flores_plus is gated, this will fail.")

        # Use FLORES+ config names (cmn_Hant, not zho_Hant — FLORES uses ISO 639-3)
        langs = ["eng_Latn", "arb_Arab", "cmn_Hant", "spa_Latn"]
        failed = []
        for lang in langs:
            print(f"  Fetching FLORES+ devtest — {lang}...")
            try:
                load_dataset(
                    "openlanguagedata/flores_plus",
                    lang,
                    split="devtest",
                    cache_dir=cache_dir,
                    token=token,
                )
                print(f"    OK")
            except Exception as lang_err:
                print(f"    WARNING: {lang} failed: {lang_err}")
                failed.append(lang)
        if failed:
            print(f"  WARNING: {len(failed)} language(s) failed: {failed}")
            print("  Re-run this script to retry failed languages.")
        else:
            print(f"  FLORES+ saved to: {cache_dir}")
    except Exception as e:
        print(f"  WARNING: FLORES+ download setup failed: {e}")
        print("  Experiment 3 will fall back to built-in sample sentences.")


def main():
    parser = argparse.ArgumentParser(description="Download models and datasets for SACRED")
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent.parent / ".hf_cache"),
        help="HuggingFace cache directory (default: <project>/.hf_cache)",
    )
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-1.3B",
        help="NLLB model ID to download",
    )
    parser.add_argument(
        "--skip-flores",
        action="store_true",
        help="Skip FLORES+ dataset download",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace access token for gated datasets (overrides HF_TOKEN env var)",
    )
    args = parser.parse_args()

    # Resolve token: CLI flag > HF_TOKEN env var
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        # Let HF libraries pick it up automatically as well
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token  # legacy alias

    # config.sh sets HF_HOME=<cache_dir> and HF_HUB_CACHE=<cache_dir>/hub.
    # We must write into <cache_dir>/hub so the SLURM jobs find files there.
    hf_home = args.cache_dir
    hub_dir = str(Path(hf_home) / "hub")
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hub_dir
    os.environ["TRANSFORMERS_CACHE"] = hub_dir

    Path(hub_dir).mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME   : {hf_home}")
    print(f"Hub cache : {hub_dir}")

    download_nllb(args.model, hub_dir)

    if not args.skip_flores:
        download_flores(hub_dir, token=token)

    print("\nAll downloads complete. You can now submit SLURM jobs in offline mode.")


if __name__ == "__main__":
    main()
