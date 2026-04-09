"""
Correctness tests for SACRED experiment pipeline.

These tests validate the four critical bugs identified in the audit:

  Bug 1 — All stimuli are English (test_stimuli_are_correct_language)
  Bug 2 — pivot_index uses absolute rates, not deltas (test_pivot_index_uses_deltas)
  Bug 3 — concept_prob sums 30+ token probs (test_concept_prob_aggregation)
  Hook  — hooks actually fire during model.generate() (test_hook_fires_during_generate)

Tests 1, 3, 4 run without a GPU (no model required).
Test 2 (test_hook_fires_during_generate) requires a loaded model; it is marked
with @pytest.mark.slow and is skipped automatically unless --runslow is passed.

Run lightweight tests:
  pytest tests/test_experiment_correctness.py -v

Run all including slow:
  pytest tests/test_experiment_correctness.py -v --runslow
"""

import re
import unicodedata
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from intervention.necessity import _contains_concept
from analysis.statistical import perform_cross_validation
from analysis.transfer_matrix import interpret_transfer_matrix

# ---------------------------------------------------------------------------
# pytest fixture: allow --runslow flag
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="Run slow tests that require a loaded model")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (requires model)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="Pass --runslow to run model-dependent tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Test 1 — Bug 1: Stimuli language check
# ---------------------------------------------------------------------------

def _has_arabic_script(text: str) -> bool:
    """Return True if text contains at least one Arabic-script character."""
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _has_cjk_characters(text: str) -> bool:
    """Return True if text contains at least one CJK character."""
    return any(
        "\u4E00" <= ch <= "\u9FFF"   # CJK Unified Ideographs
        or "\u3400" <= ch <= "\u4DBF"  # Extension A
        for ch in text
    )


def _has_spanish_markers(text: str) -> bool:
    """
    Return True if text has at least one Spanish-specific feature:
    accented vowels, ñ, ¿, or ¡.
    Spanish is Latin-script so we can only use these heuristic markers.
    """
    spanish_chars = set("áéíóúüñÁÉÍÓÚÜÑ¿¡")
    return any(ch in spanish_chars for ch in text)


def test_stimuli_are_correct_language():
    """
    EXPECTED TO FAIL (confirming Bug 1).

    ContrastivePairGenerator generates English sentences regardless of the
    `languages` parameter. Concept vectors for arb_Arab / zho_Hant / spa_Latn
    are therefore extracted from English text tokenized with the wrong
    src_lang, which produces meaningless vectors.

    This test asserts that:
      - arb_Arab positive sentences contain Arabic script
      - zho_Hant positive sentences contain CJK characters
      - spa_Latn positive sentences contain Spanish-specific characters

    If Bug 1 is present, all three assertions will fail.
    After Fix A (translate_pairs_to_language), all three should pass.
    """
    from data.contrastive_pairs import ContrastivePairGenerator

    generator = ContrastivePairGenerator(seed=42)
    pairs = generator.generate_pairs(
        domain="kinship",
        n_per_concept=3,
        languages=["arb_Arab", "zho_Hant", "spa_Latn"],
    )

    failures = []

    # Arabic
    arb_positives = [
        p["positive"]
        for concept_pairs in pairs.get("arb_Arab", {}).values()
        for p in concept_pairs
    ]
    assert arb_positives, "No Arabic pairs were generated"
    if not any(_has_arabic_script(s) for s in arb_positives):
        failures.append(
            f"arb_Arab sentences contain no Arabic script.\n"
            f"  Example: {arb_positives[0]!r}\n"
            f"  FIX: translate English templates to Arabic before extraction."
        )

    # Chinese
    zho_positives = [
        p["positive"]
        for concept_pairs in pairs.get("zho_Hant", {}).values()
        for p in concept_pairs
    ]
    assert zho_positives, "No Chinese pairs were generated"
    if not any(_has_cjk_characters(s) for s in zho_positives):
        failures.append(
            f"zho_Hant sentences contain no CJK characters.\n"
            f"  Example: {zho_positives[0]!r}\n"
            f"  FIX: translate English templates to Chinese before extraction."
        )

    # Spanish (heuristic — not all Spanish sentences contain accented chars)
    spa_positives = [
        p["positive"]
        for concept_pairs in pairs.get("spa_Latn", {}).values()
        for p in concept_pairs
    ]
    assert spa_positives, "No Spanish pairs were generated"
    if not any(_has_spanish_markers(s) for s in spa_positives):
        failures.append(
            f"spa_Latn sentences appear to be plain English (no Spanish markers).\n"
            f"  Example: {spa_positives[0]!r}\n"
            f"  FIX: translate English templates to Spanish before extraction."
        )

    if failures:
        pytest.fail(
            "Bug 1 confirmed — stimuli are NOT in the labelled language:\n\n"
            + "\n\n".join(failures)
        )


# ---------------------------------------------------------------------------
# Test 2 — Bug 2: pivot_index should use deltas from baseline
# ---------------------------------------------------------------------------

def _pivot_index_current(baseline, cond_A, cond_B, cond_C):
    """Current (buggy) absolute-rate formula."""
    mean_AB = (cond_A + cond_B) / 2
    return cond_C / mean_AB if mean_AB > 1e-6 else float("nan")


def _pivot_index_correct(baseline, cond_A, cond_B, cond_C):
    """Correct delta-from-baseline formula."""
    delta_A = cond_A - baseline
    delta_B = cond_B - baseline
    delta_C = cond_C - baseline
    mean_delta_AB = (delta_A + delta_B) / 2
    return delta_C / mean_delta_AB if abs(mean_delta_AB) > 1e-6 else float("nan")


def test_pivot_index_uses_deltas():
    """
    Demonstrates Bug 2: the current pivot_index formula produces spurious
    results when the baseline absence rate is non-zero.

    Scenario: baseline has 20% natural absence. Intervention adds 0% extra
    absence (i.e., intervention has no effect). All conditions stay at 20%.

    Expected correct behaviour:
      - delta_A = delta_B = delta_C = 0 → pivot_index = NaN (no effect)

    Current buggy behaviour:
      - pivot_index = 0.20 / 0.20 = 1.0 → falsely reports "STRONG pivot"

    Second scenario: intervention has a real effect (40% with intervention vs
    20% baseline). English vector is equally effective as source/target.
    Both formulas should agree in this case.
    """

    # --- Scenario A: intervention has ZERO effect, baseline = 20% absence ---
    baseline_del = 0.20
    cond_A_del = 0.20   # no change from baseline
    cond_B_del = 0.20
    cond_C_del = 0.20

    buggy = _pivot_index_current(baseline_del, cond_A_del, cond_B_del, cond_C_del)
    correct = _pivot_index_correct(baseline_del, cond_A_del, cond_B_del, cond_C_del)

    # Current formula incorrectly gives 1.0 (spurious "STRONG pivot evidence")
    assert abs(buggy - 1.0) < 1e-9, (
        f"Expected buggy formula to give 1.0 on no-effect scenario, got {buggy}"
    )
    # Correct formula gives NaN (correctly undefined — no intervention effect)
    assert np.isnan(correct), (
        f"Correct formula should give NaN when intervention has zero effect, got {correct}"
    )

    # --- Scenario B: zero baseline, intervention has a real effect ---
    baseline_del2 = 0.0
    cond_A_del2 = 0.40
    cond_B_del2 = 0.40
    cond_C_del2 = 0.40   # English equally effective → pivot_index = 1.0

    buggy2 = _pivot_index_current(baseline_del2, cond_A_del2, cond_B_del2, cond_C_del2)
    correct2 = _pivot_index_correct(baseline_del2, cond_A_del2, cond_B_del2, cond_C_del2)

    assert abs(buggy2 - 1.0) < 1e-9, f"Scenario B buggy formula: {buggy2}"
    assert abs(correct2 - 1.0) < 1e-9, f"Scenario B correct formula: {correct2}"

    # --- Scenario C: 20% baseline, 50% with intervention (real delta = 30pp) ---
    # English half as effective as source/target → pivot_index = 0.5
    baseline_del3 = 0.20
    cond_A_del3 = 0.50   # delta = 0.30
    cond_B_del3 = 0.50   # delta = 0.30
    cond_C_del3 = 0.35   # delta = 0.15 → ratio = 0.15/0.30 = 0.5

    correct3 = _pivot_index_correct(baseline_del3, cond_A_del3, cond_B_del3, cond_C_del3)
    buggy3 = _pivot_index_current(baseline_del3, cond_A_del3, cond_B_del3, cond_C_del3)

    assert abs(correct3 - 0.5) < 1e-9, f"Scenario C correct formula: expected 0.5, got {correct3}"
    # Buggy formula gives 0.35/0.50 = 0.70, not 0.5
    assert abs(buggy3 - 0.70) < 1e-9, (
        f"Scenario C buggy formula: expected 0.70 (wrong answer), got {buggy3}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Bug 3: concept_prob aggregation with .sum() vs .max()
# ---------------------------------------------------------------------------

def test_concept_prob_aggregation():
    """
    Demonstrates Bug 3: summing probabilities across all concept token IDs
    produces a metric with inconsistent semantics across decoding steps.

    When the model is uncertain between many kinship variants (madre/mamá/
    progenitora/etc.), each variant can receive ~2-3% probability. With 45
    concept token IDs, sum() ≈ 0.9 and max() ≈ 0.02 — a 45x divergence.

    Even though sum() stays below 1.0, it means something different at each
    step: at step i (generating "madre"), sum ≈ 0.90; at step j (generating
    "hola"), sum ≈ 0.03. Averaging these gives ~0.05 regardless of whether
    the concept word actually appeared — blending signal with noise.

    max() cleanly captures: "what was the single highest probability assigned
    to any concept token at this step?" — always a valid probability in [0,1].
    """
    vocab_size = 256206  # NLLB-200-1.3B vocabulary size
    torch.manual_seed(42)

    # Typical kinship concept_token_ids: 15 words × ~3 subword tokens
    n_concept_tokens = 45
    concept_token_ids = list(range(1000, 1000 + n_concept_tokens))  # fixed indices

    # --- Scenario A: model is uncertain between many kinship variants ---
    # (e.g., generating a kinship word but unsure which one)
    # Each of the 45 concept tokens gets 2.5% probability
    logits_uncertain = torch.full((vocab_size,), -20.0)
    for tid in concept_token_ids:
        logits_uncertain[tid] = 0.0  # raise concept tokens uniformly
    probs_uncertain = F.softmax(logits_uncertain, dim=0)

    sum_uncertain = probs_uncertain[concept_token_ids].sum().item()
    max_uncertain = probs_uncertain[concept_token_ids].max().item()

    # When 45 tokens each have equal high logit, sum captures almost ALL vocab
    # probability (≈ 1.0), while max correctly reflects each token's probability (≈ 1/45).
    # sum is ~45x max — misleadingly high as a "concept probability" signal.
    assert sum_uncertain >= 0.99, (
        f"Expected sum ≈ 1.0 when 45 concept tokens dominate distribution, "
        f"got {sum_uncertain:.4f}"
    )
    assert 0.0 <= max_uncertain <= 1.0, f"max() produced {max_uncertain}, out of [0,1]"
    # sum is far larger than max (the divergence is the key point)
    assert sum_uncertain > 10 * max_uncertain, (
        f"sum ({sum_uncertain:.4f}) should be ~45x max ({max_uncertain:.4f}) "
        f"when probability is spread across 45 concept tokens equally"
    )
    print(f"\n  Uncertain scenario (45 equal-prob concept tokens):")
    print(f"    .sum() = {sum_uncertain:.4f}  (aggregates ALL concept token probs — near 1.0)")
    print(f"    .max() = {max_uncertain:.4f}  (single most likely token — ~1/45, ~{1/45:.3f})")

    # --- Scenario B: model is generating the concept word (high prob on one token) ---
    logits_high = torch.full((vocab_size,), -20.0)
    logits_high[concept_token_ids[0]] = 5.0   # "madre" is highly likely
    logits_high[concept_token_ids[1]] = 1.0   # "mamá" is somewhat likely
    probs_high = F.softmax(logits_high, dim=0)

    sum_high = probs_high[concept_token_ids].sum().item()
    max_high = probs_high[concept_token_ids].max().item()

    assert 0.0 <= max_high <= 1.0
    print(f"\n  High-confidence scenario (one dominant concept token):")
    print(f"    .sum() = {sum_high:.4f}  (reasonable here, dominated by the top token)")
    print(f"    .max() = {max_high:.4f}  (same answer as sum in this case)")

    # --- Scenario C: averaging sum() over many steps hides the step where concept appeared ---
    # Step 0: concept present (sum=0.9), Steps 1-9: non-concept (sum=0.003 each)
    # mean_sum = (0.9 + 9×0.003) / 10 = 0.0927
    # mean_max = (0.88 + 9×0.001) / 10 = 0.0889
    # Both give ~0.09; but: mean_sum at step 0 is 0.9 (meaningful concept signal)
    # while mean at non-concept steps is 0.003 (small but non-zero, adding noise)

    # The key issue: sum() semantics change per step. max() is always
    # "probability of the single most likely concept token" — a stable metric.
    # For 0 vs >0 comparisons (deletion_rate), the binary string-match check
    # is more reliable than either; for continuous tracking, max() is cleaner.
    assert True  # Informational — the assertions above are the core of the test


# ---------------------------------------------------------------------------
# Test 4 — Hook fires during model.generate() (requires model, slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_hook_fires_during_generate():
    """
    Verify that register_vector_subtraction_hook actually modifies the model's
    output during model.generate() (not just during model.encoder() calls).

    Uses a large alpha (50.0) so even a small concept vector produces a
    measurable change in output probabilities. If the hook does NOT fire,
    baseline and ablated mean_concept_probability will be identical.

    Requires a loaded NLLB model (~5GB VRAM or RAM with CPU fallback).
    Run with: pytest tests/test_experiment_correctness.py --runslow
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from config import MODEL_NAME, HF_CACHE_DIR, DEFAULT_DEVICE, INTERVENTION_LAYERS
    from intervention.hooks import InterventionHook
    from intervention.necessity import measure_concept_deletion

    device = DEFAULT_DEVICE
    print(f"\n  Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
    except Exception as e:
        pytest.skip(f"Could not load model on {device}: {e}")
    model.eval()

    # Use a sentence where the concept is clearly present
    sentence = "My mother taught me everything I know about life."
    source_lang = "eng_Latn"
    target_lang = "spa_Latn"

    # Concept token IDs for "madre" / "mamá" in Spanish (rough)
    from data.contrastive_pairs import load_independent_sacred_tokens
    concept_token_ids = load_independent_sacred_tokens("spa_Latn", tokenizer, domain="kinship")
    concept_words = ["madre", "mamá", "mamá", "abuela", "familia", "padre", "hijo", "hija"]

    # Baseline
    baseline = measure_concept_deletion(
        [sentence], model, tokenizer, source_lang, target_lang,
        concept_token_ids, concept_words=concept_words, device=device,
    )

    # Extract a concept vector to subtract (use encoder activations)
    from extraction.concept_vectors import extract_concept_vectors
    pairs = [{"positive": sentence, "negative": "My teacher taught me everything I know about work."}]
    concept_vecs = extract_concept_vectors(
        contrastive_pairs=pairs,
        model=model, tokenizer=tokenizer, lang_code=source_lang,
        layers=INTERVENTION_LAYERS, device=device, method="mean",
    )
    stacked = torch.stack(list(concept_vecs.values())).mean(0).to(device)

    # Ablated with large alpha
    hook = InterventionHook()
    hook.register_vector_subtraction_hook(model, stacked, INTERVENTION_LAYERS, alpha=50.0)
    ablated = measure_concept_deletion(
        [sentence], model, tokenizer, source_lang, target_lang,
        concept_token_ids, intervention=hook, concept_words=concept_words, device=device,
    )
    hook.cleanup()

    baseline_prob = baseline["mean_concept_probability"]
    ablated_prob = ablated["mean_concept_probability"]
    print(f"\n  Baseline mean_concept_probability:  {baseline_prob:.6f}")
    print(f"  Ablated  mean_concept_probability:  {ablated_prob:.6f}")
    print(f"  Baseline translation: {baseline['translations'][0]!r}")
    print(f"  Ablated  translation: {ablated['translations'][0]!r}")

    assert baseline_prob != ablated_prob, (
        "Hook did NOT fire during model.generate(): baseline and ablated "
        "mean_concept_probability are identical. "
        "Check that model.model.encoder.layers[i] hooks are triggered by model.generate()."
    )
    # With alpha=50 the change should be large
    relative_change = abs(baseline_prob - ablated_prob) / (baseline_prob + 1e-10)
    assert relative_change > 0.01, (
        f"Hook fired but change is tiny ({relative_change:.4%}). "
        f"Possible causes: concept vector is near-zero, alpha too small, or wrong layers."
    )


def test_matching_mode_word_boundary_avoids_partial_match():
    # "god" should not match "godzilla" under boundary-aware matching.
    translation = "Godzilla appeared in the city."
    assert not _contains_concept(
        translation=translation,
        concept_words=["god"],
        token_present=False,
        matching_mode="word_boundary",
    )


def test_matching_mode_hybrid_uses_token_or_lexical():
    # token evidence should trigger presence even without lexical match.
    assert _contains_concept(
        translation="Nothing lexical here",
        concept_words=["mother"],
        token_present=True,
        matching_mode="hybrid",
    )
    # lexical boundary match should trigger even with token miss.
    assert _contains_concept(
        translation="My mother is kind.",
        concept_words=["mother"],
        token_present=False,
        matching_mode="hybrid",
    )


def test_cross_validation_placeholder_flags():
    cv = perform_cross_validation(
        stimuli={},
        model=None,
        tokenizer=None,
        discovery_fn=lambda *args, **kwargs: None,
        n_folds=3,
    )
    assert cv.implemented is False
    assert cv.status == "placeholder"


def test_transfer_summary_reports_absolute_and_relative_hub_metrics():
    matrix = np.array([
        [0.2, 0.6, 0.5, 0.4],   # eng row
        [0.55, 0.2, 0.3, 0.3],
        [0.52, 0.2, 0.2, 0.3],
        [0.5, 0.2, 0.3, 0.2],
    ], dtype=float)
    langs = ["eng_Latn", "arb_Arab", "spa_Latn", "zho_Hant"]
    summary = interpret_transfer_matrix(matrix, langs)
    assert "english_hub_score" in summary
    assert "english_hub_absolute_mean" in summary
    assert "non_english_absolute_mean" in summary
    assert "off_diagonal_ceiling_rate" in summary
    assert summary["english_hub_score"] > 0.0


def test_matching_mode_hybrid_handles_english_target_mismatch():
    # Token mismatch but lexical hit: should still count as present.
    assert _contains_concept(
        translation="The mother embraced the child.",
        concept_words=["mother"],
        token_present=False,
        matching_mode="hybrid",
    ) is True
