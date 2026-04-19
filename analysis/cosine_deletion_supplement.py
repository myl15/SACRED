"""
Post-hoc cosine-similarity deletion diagnostics for Exp2/Exp4 outputs.

This module regenerates baseline + ablated translations for selected pairs,
embeds generated outputs using the model shared embedding table, and computes:
  1) baseline-vs-ablated cosine deletion rate
  2) concept-anchor crossing deletion rate

It then joins these with token-matching deletion rates from existing results
JSON and writes one supplementary CSV table.
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import EXPERIMENT_LANGUAGES, HF_CACHE_DIR, INTERVENTION_LAYERS, MODEL_NAME
from data.concept_vocabularies import CONCEPT_VOCABULARIES
from intervention.hooks import InterventionHook


@dataclass
class PairEvalSpec:
    experiment: str
    domain: str
    vector_method: str
    pair_label: str
    source_lang: str
    target_lang: str
    vector_lang: str
    token_deletion_rate: float
    alpha: float
    matching_mode: str
    layers: List[int]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool_from_rate(rate: float, eps: float = 1e-8) -> bool:
    return bool(rate > eps)


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_stimuli(stimuli_path: Path) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    with open(stimuli_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_pair_key(pair_key: str) -> Tuple[str, str]:
    parsed = ast.literal_eval(pair_key)
    if not isinstance(parsed, tuple) or len(parsed) != 2:
        raise ValueError(f"Unexpected pair key format: {pair_key}")
    return str(parsed[0]), str(parsed[1])


def _average_concept_vectors(raw: Dict) -> Dict[int, torch.Tensor]:
    layer_vecs: Dict[int, List[torch.Tensor]] = {}
    for _, layer_dict in raw.items():
        for layer_str, vec in layer_dict.items():
            layer = int(layer_str)
            layer_vecs.setdefault(layer, []).append(vec)
    return {layer: torch.stack(vecs).mean(0) for layer, vecs in layer_vecs.items()}


def _load_language_vectors(
    vectors_dir: Path,
    domain: str,
    method: str,
    device: str,
) -> Dict[str, Dict[int, torch.Tensor]]:
    suffix = "_pca" if method == "pca" else ""
    out: Dict[str, Dict[int, torch.Tensor]] = {}
    for lang in EXPERIMENT_LANGUAGES:
        vec_path = vectors_dir / f"{domain}_{lang}{suffix}.pt"
        if not vec_path.exists():
            continue
        raw = torch.load(vec_path, map_location=device)
        out[lang] = _average_concept_vectors(raw)
    return out


def _sentence_embedding_from_ids(
    input_ids: torch.Tensor,
    model,
    pad_id: Optional[int],
    eos_id: Optional[int],
) -> torch.Tensor:
    emb_table = model.model.shared
    token_emb = emb_table(input_ids)  # [1, seq, dim]
    mask = torch.ones_like(input_ids, dtype=torch.float32)
    if pad_id is not None:
        mask = mask * (input_ids != pad_id).float()
    if eos_id is not None:
        mask = mask * (input_ids != eos_id).float()
    if mask.sum() <= 0:
        return token_emb.mean(dim=1)[0]
    weighted = token_emb * mask.unsqueeze(-1)
    return weighted.sum(dim=1)[0] / (mask.sum() + 1e-8)


def _build_anchor_embedding(tokenizer, model, device: str) -> torch.Tensor:
    words = CONCEPT_VOCABULARIES["kinship"]["eng_Latn"]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    vecs: List[torch.Tensor] = []
    for w in words:
        ids = tokenizer(
            w,
            return_tensors="pt",
            src_lang="eng_Latn",
        )["input_ids"].to(device)
        vecs.append(_sentence_embedding_from_ids(ids, model, pad_id, eos_id))
    return torch.stack(vecs, dim=0).mean(dim=0)


def _run_translations(
    model,
    tokenizer,
    source_lang: str,
    target_lang: str,
    sentences: List[str],
    intervention: Optional[InterventionHook],
    device: str,
) -> List[torch.Tensor]:
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
    generated: List[torch.Tensor] = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", src_lang=source_lang).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                forced_bos_token_id=target_lang_id,
                max_length=50,
                num_beams=1,
            )
        generated.append(outputs[0])
    return generated


def _evaluate_cosine_deletion(
    baseline_ids: List[torch.Tensor],
    ablated_ids: List[torch.Tensor],
    tokenizer,
    model,
    anchor_vec: torch.Tensor,
    cosine_change_threshold: float,
    anchor_presence_threshold: float,
) -> Tuple[float, float]:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    changed = 0
    anchor_deleted = 0
    n = min(len(baseline_ids), len(ablated_ids))
    if n == 0:
        return 0.0, 0.0
    for i in range(n):
        b_vec = _sentence_embedding_from_ids(baseline_ids[i].unsqueeze(0), model, pad_id, eos_id)
        a_vec = _sentence_embedding_from_ids(ablated_ids[i].unsqueeze(0), model, pad_id, eos_id)
        cos_ba = F.cosine_similarity(b_vec.unsqueeze(0), a_vec.unsqueeze(0), dim=1).item()
        if cos_ba <= cosine_change_threshold:
            changed += 1

        b_anchor = F.cosine_similarity(b_vec.unsqueeze(0), anchor_vec.unsqueeze(0), dim=1).item()
        a_anchor = F.cosine_similarity(a_vec.unsqueeze(0), anchor_vec.unsqueeze(0), dim=1).item()
        if b_anchor >= anchor_presence_threshold and a_anchor < anchor_presence_threshold:
            anchor_deleted += 1

    return changed / n, anchor_deleted / n


def _build_eval_specs_from_exp2(path: Path) -> List[PairEvalSpec]:
    data = _load_json(path)
    md = data.get("metadata", {})
    domain = md.get("domain", "kinship")
    method = md.get("vector_method", "mean")
    alpha = _safe_float(md.get("alpha"), 0.25)
    matching_mode = md.get("matching_mode", "hybrid")
    layers = md.get("layers", INTERVENTION_LAYERS)
    specs: List[PairEvalSpec] = []
    for pair_key, row in data.get("results_by_pair", {}).items():
        src, tgt = _parse_pair_key(pair_key)
        token_rate = _safe_float(row.get("condition_C_english", {}).get("deletion_rate"), 0.0)
        specs.append(
            PairEvalSpec(
                experiment="exp2",
                domain=domain,
                vector_method=method,
                pair_label=f"{src.split('_')[0]}→{tgt.split('_')[0]}",
                source_lang=src,
                target_lang=tgt,
                vector_lang="eng_Latn",
                token_deletion_rate=token_rate,
                alpha=alpha,
                matching_mode=matching_mode,
                layers=[int(l) for l in layers],
            )
        )
    return specs


def _build_eval_specs_from_exp4(path: Path) -> List[PairEvalSpec]:
    data = _load_json(path)
    domain = str(data.get("domain", "kinship"))
    method = str(data.get("vector_method", "mean"))
    alpha = _safe_float(data.get("alpha"), 0.25)
    matching_mode = str(data.get("matching_mode", "hybrid"))
    layers = [int(l) for l in data.get("run_manifest", {}).get("config_snapshot", {}).get("layers", INTERVENTION_LAYERS)]
    langs = [str(x) for x in data.get("languages", [])]
    if not langs:
        return []

    results_root = path.parents[2]
    matrix_path = results_root / "vectors" / f"exp4_deletion_matrix_{domain}_{method}_output_lang_{data.get('output_lang', 'eng_Latn')}.npy"
    if not matrix_path.exists():
        return []
    import numpy as np
    matrix = np.load(str(matrix_path))

    specs: List[PairEvalSpec] = []
    try:
        eng_idx = langs.index("eng_Latn")
    except ValueError:
        return specs

    for i, src in enumerate(langs):
        if src == "eng_Latn":
            continue
        token_rate = float(matrix[i, eng_idx]) if i < matrix.shape[0] and eng_idx < matrix.shape[1] else 0.0
        specs.append(
            PairEvalSpec(
                experiment="exp4",
                domain=domain,
                vector_method=method,
                pair_label=f"{src.split('_')[0]}→eng",
                source_lang=src,
                target_lang="eng_Latn",
                vector_lang=src,
                token_deletion_rate=token_rate,
                alpha=alpha,
                matching_mode=matching_mode,
                layers=layers,
            )
        )
    return specs


def _collect_specs(results_dir: Path) -> List[PairEvalSpec]:
    specs: List[PairEvalSpec] = []
    for p in sorted((results_dir / "json").glob("exp2_pivot_*.json")):
        specs.extend(_build_eval_specs_from_exp2(p))
    for p in sorted((results_dir / "eng_Latn" / "json").glob("exp4_transfer_summary_*.json")):
        specs.extend(_build_eval_specs_from_exp4(p))
    return specs


def _iter_positive_sentences(
    stimuli: Dict[str, Dict[str, List[Dict[str, str]]]],
    lang: str,
) -> List[str]:
    out: List[str] = []
    for concept_pairs in stimuli.get(lang, {}).values():
        out.extend([p["positive"] for p in concept_pairs if "positive" in p])
    return out


def _group_label(spec: PairEvalSpec) -> str:
    if (
        spec.experiment == "exp4"
        and spec.domain == "kinship"
        and spec.target_lang == "eng_Latn"
        and abs(spec.token_deletion_rate) <= 1e-8
    ):
        return "x_to_eng_kinship_zero_token"
    return "token_matching_valid"


def run_cosine_supplement(
    results_dir: str = "results",
    vectors_dir: str = "outputs/vectors",
    stimuli_dir: str = "outputs/stimuli",
    output_csv: str = "results/paper/table_cosine_deletion_supplement.csv",
    device: str = "cuda",
    cosine_change_threshold: float = 0.90,
    anchor_presence_threshold: float = 0.20,
    max_sentences_per_pair: int = 0,
) -> Path:
    results_path = Path(results_dir)
    vectors_path = Path(vectors_dir)
    stimuli_path = Path(stimuli_dir)

    specs = _collect_specs(results_path)
    if not specs:
        raise FileNotFoundError(f"No Exp2/Exp4 specs found under: {results_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
    model.eval()
    anchor_vec = _build_anchor_embedding(tokenizer, model, device).to(device)

    vector_cache: Dict[Tuple[str, str], Dict[str, Dict[int, torch.Tensor]]] = {}
    stimuli_cache: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]] = {}
    rows: List[Dict[str, str]] = []

    for spec in specs:
        if spec.domain not in stimuli_cache:
            stim_file = stimuli_path / f"{spec.domain}_pairs.json"
            if not stim_file.exists():
                continue
            stimuli_cache[spec.domain] = _load_stimuli(stim_file)
        sentences = _iter_positive_sentences(stimuli_cache[spec.domain], spec.source_lang)
        if max_sentences_per_pair > 0:
            sentences = sentences[:max_sentences_per_pair]
        if not sentences:
            continue

        vec_key = (spec.domain, spec.vector_method)
        if vec_key not in vector_cache:
            vector_cache[vec_key] = _load_language_vectors(vectors_path, spec.domain, spec.vector_method, device)
        method_vectors = vector_cache[vec_key]

        vecs_lang = method_vectors.get(spec.vector_lang, {})
        layer_vecs = {l: vecs_lang[l] for l in spec.layers if l in vecs_lang}
        if not layer_vecs:
            continue

        baseline_ids = _run_translations(
            model=model,
            tokenizer=tokenizer,
            source_lang=spec.source_lang,
            target_lang=spec.target_lang,
            sentences=sentences,
            intervention=None,
            device=device,
        )
        hook = InterventionHook()
        hook.register_vector_subtraction_hook(model, layer_vecs, spec.layers, alpha=spec.alpha)
        ablated_ids = _run_translations(
            model=model,
            tokenizer=tokenizer,
            source_lang=spec.source_lang,
            target_lang=spec.target_lang,
            sentences=sentences,
            intervention=hook,
            device=device,
        )
        hook.cleanup()

        cos_rate, anchor_rate = _evaluate_cosine_deletion(
            baseline_ids=baseline_ids,
            ablated_ids=ablated_ids,
            tokenizer=tokenizer,
            model=model,
            anchor_vec=anchor_vec,
            cosine_change_threshold=cosine_change_threshold,
            anchor_presence_threshold=anchor_presence_threshold,
        )

        token_nonzero = _safe_bool_from_rate(spec.token_deletion_rate)
        row = {
            "experiment": spec.experiment,
            "domain": spec.domain,
            "vector_method": spec.vector_method,
            "pair_label": spec.pair_label,
            "token_deletion_rate": f"{spec.token_deletion_rate:.6f}",
            "cosine_deletion_rate_baseline_ablated": f"{cos_rate:.6f}",
            "cosine_deletion_rate_anchor": f"{anchor_rate:.6f}",
            "directional_agreement_baseline_ablated": str(token_nonzero == _safe_bool_from_rate(cos_rate)).lower(),
            "directional_agreement_anchor": str(token_nonzero == _safe_bool_from_rate(anchor_rate)).lower(),
            "group": _group_label(spec),
        }
        rows.append(row)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "domain",
        "vector_method",
        "pair_label",
        "token_deletion_rate",
        "cosine_deletion_rate_baseline_ablated",
        "cosine_deletion_rate_anchor",
        "directional_agreement_baseline_ablated",
        "directional_agreement_anchor",
        "group",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path

