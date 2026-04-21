"""
Domain- and target-language matched cosine concept-deletion metrics (Exp5).

Anchor: mean pooled shared embedding over CONCEPT_VOCABULARIES[domain][target_lang].
Intervention vectors match Exp2 (layer-mean tensor) and Exp4 (per-layer dict).
"""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from analysis.transfer_matrix import _get_layer_mean_vector
from config import EXPERIMENT_LANGUAGES, HF_CACHE_DIR, INTERVENTION_LAYERS, MODEL_NAME
from data.concept_vocabularies import CONCEPT_VOCABULARIES
from intervention.hooks import InterventionHook


def _log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[exp5][{ts}] {message}", flush=True)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Dict[str, Any]:
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


def _average_concept_vectors(raw: Dict[str, Any]) -> Dict[int, torch.Tensor]:
    layer_vecs: Dict[int, List[torch.Tensor]] = {}
    for _, layer_dict in raw.items():
        if not isinstance(layer_dict, dict):
            continue
        for layer_str, vec in layer_dict.items():
            layer = int(layer_str)
            layer_vecs.setdefault(layer, []).append(vec)
    return {layer: torch.stack(vecs).mean(0) for layer, vecs in layer_vecs.items()}


def load_concept_vectors_by_lang(
    vectors_dir: Path,
    domain: str,
    method: str,
    device: str,
    vector_source_domain: Optional[str] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    file_domain = vector_source_domain if vector_source_domain is not None else domain
    suffix = "_pca" if method == "pca" else ""
    out: Dict[str, Dict[int, torch.Tensor]] = {}
    for lang in EXPERIMENT_LANGUAGES:
        vec_path = vectors_dir / f"{file_domain}_{lang}{suffix}.pt"
        if not vec_path.exists():
            continue
        raw = torch.load(vec_path, map_location=device)
        out[lang] = _average_concept_vectors(raw)
    return out


def sentence_embedding_from_ids(
    input_ids: torch.Tensor,
    model: Any,
    pad_id: Optional[int],
    eos_id: Optional[int],
) -> torch.Tensor:
    emb_table = model.model.shared
    token_emb = emb_table(input_ids)
    mask = torch.ones_like(input_ids, dtype=torch.float32)
    if pad_id is not None:
        mask = mask * (input_ids != pad_id).float()
    if eos_id is not None:
        mask = mask * (input_ids != eos_id).float()
    if mask.sum() <= 0:
        return token_emb.mean(dim=1)[0]
    weighted = token_emb * mask.unsqueeze(-1)
    return weighted.sum(dim=1)[0] / (mask.sum() + 1e-8)


def build_concept_anchor(
    domain: str,
    target_lang: str,
    tokenizer: Any,
    model: Any,
    device: str,
    cache: Dict[Tuple[str, str], torch.Tensor],
) -> torch.Tensor:
    key = (domain, target_lang)
    if key in cache:
        return cache[key]
    words = CONCEPT_VOCABULARIES.get(domain, {}).get(target_lang)
    if not words:
        raise KeyError(f"No concept vocabulary for domain={domain} target_lang={target_lang}")
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    vecs: List[torch.Tensor] = []
    for w in words:
        ids = tokenizer(w, return_tensors="pt", src_lang=target_lang)["input_ids"].to(device)
        vecs.append(sentence_embedding_from_ids(ids, model, pad_id, eos_id))
    anchor = torch.stack(vecs, dim=0).mean(dim=0)
    cache[key] = anchor
    return anchor


def run_translations(
    model: Any,
    tokenizer: Any,
    source_lang: str,
    target_lang: str,
    sentences: List[str],
    intervention: Optional[InterventionHook],
    device: str,
    batch_size: int = 8,
) -> List[torch.Tensor]:
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
    generated: List[torch.Tensor] = []
    for i in range(0, len(sentences), max(1, batch_size)):
        batch_sentences = sentences[i : i + max(1, batch_size)]
        inputs = tokenizer(
            batch_sentences,
            return_tensors="pt",
            src_lang=source_lang,
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                forced_bos_token_id=target_lang_id,
                max_length=50,
                num_beams=1,
            )
        for j in range(outputs.shape[0]):
            generated.append(outputs[j])
    return generated


def baseline_anchor_gate_stats(
    baseline_ids: List[torch.Tensor],
    tokenizer: Any,
    model: Any,
    anchor: torch.Tensor,
    presence_threshold: float,
    low_gate_threshold: float,
) -> Dict[str, Any]:
    """Gate rate and mean baseline–anchor cosine (independent of ablation)."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    n = len(baseline_ids)
    if n == 0:
        return {
            "gate_pass_rate": float("nan"),
            "mean_b_anchor": float("nan"),
            "low_gate_warning": "low_gate",
        }
    b_list: List[float] = []
    n_pass = 0
    for i in range(n):
        b_vec = sentence_embedding_from_ids(baseline_ids[i].unsqueeze(0), model, pad_id, eos_id)
        b_anchor = F.cosine_similarity(b_vec.unsqueeze(0), anchor.unsqueeze(0), dim=1).item()
        b_list.append(b_anchor)
        if b_anchor >= presence_threshold:
            n_pass += 1
    gpr = n_pass / n
    low = "low_gate" if gpr < low_gate_threshold else ""
    return {
        "gate_pass_rate": gpr,
        "mean_b_anchor": float(np.mean(b_list)),
        "low_gate_warning": low,
    }


def score_anchor_metric(
    baseline_ids: List[torch.Tensor],
    ablated_ids: List[torch.Tensor],
    tokenizer: Any,
    model: Any,
    anchor: torch.Tensor,
    presence_threshold: float,
    deletion_mode: str,
    deletion_threshold: float,
    deletion_relative_margin: float,
    low_gate_threshold: float,
) -> Dict[str, Any]:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    n = min(len(baseline_ids), len(ablated_ids))
    if n == 0:
        return {
            "gate_pass_rate": float("nan"),
            "cosine_deletion_rate": float("nan"),
            "mean_b_anchor": float("nan"),
            "mean_a_anchor": float("nan"),
            "low_gate_warning": "low_gate",
        }
    b_list: List[float] = []
    a_list: List[float] = []
    gated_deletions: List[int] = []
    for i in range(n):
        b_vec = sentence_embedding_from_ids(baseline_ids[i].unsqueeze(0), model, pad_id, eos_id)
        a_vec = sentence_embedding_from_ids(ablated_ids[i].unsqueeze(0), model, pad_id, eos_id)
        b_anchor = F.cosine_similarity(b_vec.unsqueeze(0), anchor.unsqueeze(0), dim=1).item()
        a_anchor = F.cosine_similarity(a_vec.unsqueeze(0), anchor.unsqueeze(0), dim=1).item()
        b_list.append(b_anchor)
        a_list.append(a_anchor)
        if b_anchor >= presence_threshold:
            if deletion_mode == "relative_drop":
                thr = b_anchor - deletion_relative_margin
                gated_deletions.append(1 if a_anchor < thr else 0)
            else:
                gated_deletions.append(1 if a_anchor < deletion_threshold else 0)
    n_gate = len(gated_deletions)
    gate_pass_rate = n_gate / n
    if n_gate == 0:
        cos_del = float("nan")
    else:
        cos_del = sum(gated_deletions) / n_gate
    low_gate = "low_gate" if gate_pass_rate < low_gate_threshold else ""
    return {
        "gate_pass_rate": gate_pass_rate,
        "cosine_deletion_rate": cos_del,
        "mean_b_anchor": float(np.mean(b_list)),
        "mean_a_anchor": float(np.mean(a_list)),
        "low_gate_warning": low_gate,
    }


def intervention_divergence_rate(
    baseline_ids: List[torch.Tensor],
    ablated_ids: List[torch.Tensor],
    tokenizer: Any,
    model: Any,
    tau: float,
) -> float:
    """Fraction of sentences with cos(pooled_baseline, pooled_ablated) <= tau."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    n = min(len(baseline_ids), len(ablated_ids))
    if n == 0:
        return float("nan")
    changed = 0
    for i in range(n):
        b_vec = sentence_embedding_from_ids(baseline_ids[i].unsqueeze(0), model, pad_id, eos_id)
        a_vec = sentence_embedding_from_ids(ablated_ids[i].unsqueeze(0), model, pad_id, eos_id)
        cos_ba = F.cosine_similarity(b_vec.unsqueeze(0), a_vec.unsqueeze(0), dim=1).item()
        if cos_ba <= tau:
            changed += 1
    return changed / n


def _fmt_csv_float(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.6f}"


def _primary_row(
    experiment: str,
    domain: str,
    vector_method: str,
    pair_label: str,
    condition: str,
    stats: Dict[str, Any],
) -> Dict[str, str]:
    return {
        "experiment": experiment,
        "domain": domain,
        "vector_method": vector_method,
        "pair_label": pair_label,
        "condition": condition,
        "gate_pass_rate": _fmt_csv_float(stats["gate_pass_rate"]),
        "cosine_deletion_rate": _fmt_csv_float(stats["cosine_deletion_rate"]),
        "mean_b_anchor": _fmt_csv_float(stats["mean_b_anchor"]),
        "mean_a_anchor": _fmt_csv_float(stats["mean_a_anchor"]),
        "low_gate_warning": stats["low_gate_warning"],
    }


def _iter_positive_sentences(
    stimuli: Dict[str, Dict[str, List[Dict[str, str]]]],
    lang: str,
) -> List[str]:
    out: List[str] = []
    for concept_pairs in stimuli.get(lang, {}).values():
        out.extend([p["positive"] for p in concept_pairs if "positive" in p])
    return out


def sample_random_pivot_vector(
    concept_vectors: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    hidden_dim: int,
    random_seed: int,
    trial_index: int,
    device: torch.device,
) -> torch.Tensor:
    """Mirrors pivot_diagnosis: vec_C from English, vec_D ~ random with matched norm."""
    vec_c = _get_layer_mean_vector(concept_vectors, "eng_Latn", layers).to(device)
    gen_cpu = torch.Generator()
    gen_cpu.manual_seed(int(random_seed) + int(trial_index) * 1000003)
    vec_d = torch.randn(hidden_dim, generator=gen_cpu, device="cpu").to(device)
    vec_d = vec_d / vec_d.norm() * vec_c.norm()
    return vec_d


def run_exp1_validation(
    vectors_dir: Path,
    stimuli_dir: Path,
    exp1_json_dir: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    presence_threshold: float,
    deletion_mode: str,
    deletion_threshold: float,
    deletion_relative_margin: float,
    low_gate_threshold: float,
    max_sentences_per_concept: int,
    domains: List[str],
    log_every: int = 1,
    generation_batch_size: int = 8,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    anchor_cache: Dict[Tuple[str, str], torch.Tensor] = {}
    source_lang = "eng_Latn"
    target_lang = "eng_Latn"

    for domain_idx, domain in enumerate(domains, start=1):
        _log(f"Exp1 validation domain {domain_idx}/{len(domains)}: {domain}")
        exp1_path = exp1_json_dir / f"exp1_{domain}_deletion.json"
        if not exp1_path.exists():
            _log(f"Exp1 validation skipped for domain={domain}: missing {exp1_path}")
            continue
        data = _load_json(exp1_path)
        deletion_results = data.get("deletion_results", {})
        eng_block = deletion_results.get(source_lang, {})
        stim_path = stimuli_dir / f"{domain}_pairs.json"
        if not stim_path.exists():
            continue
        stimuli = _load_stimuli(stim_path)
        lang_pairs = stimuli.get(source_lang, {})

        raw_vecs_path = vectors_dir / f"{domain}_{source_lang}.pt"
        if not raw_vecs_path.exists():
            continue
        raw_by_concept = torch.load(raw_vecs_path, map_location=device)

        anchor = build_concept_anchor(domain, target_lang, tokenizer, model, device, anchor_cache)

        concept_items = list(eng_block.items())
        n_concepts = len(concept_items)
        for concept_idx, (concept, metrics) in enumerate(concept_items, start=1):
            if concept_idx == 1 or concept_idx % max(1, log_every) == 0 or concept_idx == n_concepts:
                _log(
                    f"Exp1 validation {domain}: concept {concept_idx}/{n_concepts} ({concept})"
                )
            token_dr = _safe_float(metrics.get("deletion_rate"), 0.0)
            pairs = lang_pairs.get(concept, [])
            sentences = [p["positive"] for p in pairs if "positive" in p]
            if max_sentences_per_concept > 0:
                sentences = sentences[:max_sentences_per_concept]
            if not sentences:
                continue
            concept_layer_vecs = raw_by_concept.get(concept, {})
            if not concept_layer_vecs:
                continue
            layer_keys = sorted(concept_layer_vecs.keys(), key=lambda k: int(str(k)))
            stacked = torch.stack([concept_layer_vecs[k] for k in layer_keys], dim=0)
            mean_vec = stacked.mean(dim=0).to(device)

            baseline_ids = run_translations(
                model, tokenizer, source_lang, target_lang, sentences, None, device,
                batch_size=generation_batch_size,
            )
            hook = InterventionHook()
            hook.register_vector_subtraction_hook(
                model, mean_vec, list(INTERVENTION_LAYERS), alpha=1.0,
            )
            ablated_ids = run_translations(
                model, tokenizer, source_lang, target_lang, sentences, hook, device,
                batch_size=generation_batch_size,
            )
            hook.cleanup()

            stats = score_anchor_metric(
                baseline_ids, ablated_ids, tokenizer, model, anchor,
                presence_threshold, deletion_mode, deletion_threshold,
                deletion_relative_margin, low_gate_threshold,
            )
            rows.append({
                "experiment": "exp1",
                "domain": domain,
                "vector_method": "mean",
                "pair_label": "eng→eng",
                "condition": f"concept_{concept}",
                "gate_pass_rate": _fmt_csv_float(stats["gate_pass_rate"]),
                "cosine_deletion_rate": _fmt_csv_float(stats["cosine_deletion_rate"]),
                "mean_b_anchor": _fmt_csv_float(stats["mean_b_anchor"]),
                "mean_a_anchor": _fmt_csv_float(stats["mean_a_anchor"]),
                "low_gate_warning": stats["low_gate_warning"],
                "token_deletion_rate": f"{token_dr:.6f}",
            })
    return rows


def run_exp2_rows(
    results_dir: Path,
    vectors_dir: Path,
    stimuli_dir: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    presence_threshold: float,
    deletion_mode: str,
    deletion_threshold: float,
    deletion_relative_margin: float,
    low_gate_threshold: float,
    max_sentences_per_pair: int,
    hidden_dim: int,
    divergence_tau: float,
    log_every: int = 1,
    generation_batch_size: int = 8,
    max_random_trials: int = 0,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    primary: List[Dict[str, str]] = []
    diverge: List[Dict[str, str]] = []
    anchor_cache: Dict[Tuple[str, str], torch.Tensor] = {}
    dev = torch.device(device)

    exp2_jsons = sorted((results_dir / "json").glob("exp2_pivot_*.json"))
    _log(f"Exp2: found {len(exp2_jsons)} pivot JSON files")
    for json_idx, json_path in enumerate(exp2_jsons, start=1):
        _log(f"Exp2 file {json_idx}/{len(exp2_jsons)}: {json_path.name}")
        data = _load_json(json_path)
        md = data.get("metadata", {})
        domain = str(md.get("domain", "kinship"))
        method = str(md.get("vector_method", "mean"))
        alpha = _safe_float(md.get("alpha"), 0.25)
        layers = [int(x) for x in md.get("layers", INTERVENTION_LAYERS)]
        n_random = int(md.get("n_random_controls", 20))
        if max_random_trials > 0:
            orig_n_random = n_random
            n_random = min(n_random, max_random_trials)
            if n_random != orig_n_random:
                _log(
                    f"Exp2 {json_path.name}: capping D_random trials "
                    f"from {orig_n_random} to {n_random}"
                )
        random_seed = int(md.get("random_seed", 42))
        vsd = md.get("vector_source_domain")

        concept_vectors = load_concept_vectors_by_lang(
            vectors_dir, domain, method, device,
            vector_source_domain=str(vsd) if vsd else None,
        )
        if not concept_vectors:
            continue

        stim_path = stimuli_dir / f"{domain}_pairs.json"
        if not stim_path.exists():
            continue
        stimuli = _load_stimuli(stim_path)

        pair_items = list(data.get("results_by_pair", {}).items())
        _log(f"Exp2 {json_path.name}: {len(pair_items)} language pairs")
        for pair_idx, (pair_key, _pair_res) in enumerate(pair_items, start=1):
            src, tgt = _parse_pair_key(pair_key)
            label = f"{src.split('_')[0]}→{tgt.split('_')[0]}"
            if pair_idx == 1 or pair_idx % max(1, log_every) == 0 or pair_idx == len(pair_items):
                _log(f"Exp2 {json_path.name}: pair {pair_idx}/{len(pair_items)} ({label})")
            sentences = _iter_positive_sentences(stimuli, src)
            if max_sentences_per_pair > 0:
                sentences = sentences[:max_sentences_per_pair]
            if not sentences:
                continue
            try:
                anchor = build_concept_anchor(domain, tgt, tokenizer, model, device, anchor_cache)
            except KeyError:
                continue

            baseline_ids = run_translations(
                model, tokenizer, src, tgt, sentences, None, device,
                batch_size=generation_batch_size,
            )
            base_gate = baseline_anchor_gate_stats(
                baseline_ids, tokenizer, model, anchor,
                presence_threshold, low_gate_threshold,
            )

            def run_cond(vec: Optional[torch.Tensor]) -> List[torch.Tensor]:
                hook = InterventionHook()
                if vec is not None:
                    hook.register_vector_subtraction_hook(model, vec, layers, alpha=alpha)
                out = run_translations(
                    model,
                    tokenizer,
                    src,
                    tgt,
                    sentences,
                    hook,
                    device,
                    batch_size=generation_batch_size,
                )
                # Batch generation is the primary runtime lever.
                hook.cleanup()
                return out

            vec_a = _get_layer_mean_vector(concept_vectors, src, layers)
            vec_b = _get_layer_mean_vector(concept_vectors, tgt, layers)
            vec_c = _get_layer_mean_vector(concept_vectors, "eng_Latn", layers).to(device)

            for cond_name, vec in (
                ("A_source", vec_a),
                ("B_target", vec_b),
                ("C_english", vec_c),
            ):
                _log(f"Exp2 {json_path.name} {label}: running condition {cond_name}")
                ablated = run_cond(vec)
                stats = score_anchor_metric(
                    baseline_ids, ablated, tokenizer, model, anchor,
                    presence_threshold, deletion_mode, deletion_threshold,
                    deletion_relative_margin, low_gate_threshold,
                )
                primary.append(_primary_row("exp2", domain, method, label, cond_name, stats))
                diverge.append({
                    "experiment": "exp2",
                    "domain": domain,
                    "vector_method": method,
                    "pair_label": label,
                    "condition": cond_name,
                    "intervention_divergence_rate": _fmt_csv_float(
                        intervention_divergence_rate(
                            baseline_ids, ablated, tokenizer, model, divergence_tau,
                        )
                    ),
                })

            d_cos_rates: List[float] = []
            d_mean_a: List[float] = []
            div_d: List[float] = []
            n_trials = max(1, n_random)
            for t in range(n_trials):
                if t == 0 or (t + 1) % max(1, log_every) == 0 or (t + 1) == n_trials:
                    _log(
                        f"Exp2 {json_path.name} {label}: D_random trial {t + 1}/{n_trials}"
                    )
                vec_d = sample_random_pivot_vector(
                    concept_vectors, layers, hidden_dim, random_seed, t, dev,
                )
                ablated_d = run_cond(vec_d)
                st = score_anchor_metric(
                    baseline_ids, ablated_d, tokenizer, model, anchor,
                    presence_threshold, deletion_mode, deletion_threshold,
                    deletion_relative_margin, low_gate_threshold,
                )
                cdr = st["cosine_deletion_rate"]
                if not (isinstance(cdr, float) and math.isnan(cdr)):
                    d_cos_rates.append(cdr)
                d_mean_a.append(st["mean_a_anchor"])
                div_d.append(
                    intervention_divergence_rate(
                        baseline_ids, ablated_d, tokenizer, model, divergence_tau,
                    )
                )
            agg_cos = float(np.nanmean(d_cos_rates)) if d_cos_rates else float("nan")
            agg_mean_a = float(np.mean(d_mean_a)) if d_mean_a else float("nan")
            d_row_stats = {
                "gate_pass_rate": base_gate["gate_pass_rate"],
                "cosine_deletion_rate": agg_cos,
                "mean_b_anchor": base_gate["mean_b_anchor"],
                "mean_a_anchor": agg_mean_a,
                "low_gate_warning": base_gate["low_gate_warning"],
            }
            primary.append(_primary_row("exp2", domain, method, label, "D_random", d_row_stats))
            diverge.append({
                "experiment": "exp2",
                "domain": domain,
                "vector_method": method,
                "pair_label": label,
                "condition": "D_random",
                "intervention_divergence_rate": _fmt_csv_float(
                    float(np.mean(div_d)) if div_d else float("nan"),
                ),
            })

    return primary, diverge


def run_exp4_rows(
    results_dir: Path,
    vectors_dir: Path,
    stimuli_dir: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    presence_threshold: float,
    deletion_mode: str,
    deletion_threshold: float,
    deletion_relative_margin: float,
    low_gate_threshold: float,
    max_sentences_per_cell: int,
    divergence_tau: float,
    log_every: int = 1,
    generation_batch_size: int = 8,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    primary: List[Dict[str, str]] = []
    diverge: List[Dict[str, str]] = []
    anchor_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    exp4_jsons = sorted(results_dir.glob("*/json/exp4_transfer_summary_*.json"))
    _log(f"Exp4: found {len(exp4_jsons)} transfer summary JSON files")
    for json_idx, json_path in enumerate(exp4_jsons, start=1):
        _log(f"Exp4 file {json_idx}/{len(exp4_jsons)}: {json_path}")
        data = _load_json(json_path)
        cfg = data.get("run_manifest", {}).get("config", {})
        domain = str(data.get("domain", cfg.get("domain", "kinship")))
        method = str(data.get("vector_method", cfg.get("vector_method", "mean")))
        alpha = _safe_float(data.get("alpha", cfg.get("alpha")), 0.25)
        layers = [int(x) for x in (cfg.get("layers") or data.get("layers") or INTERVENTION_LAYERS)]
        output_lang = str(data.get("output_lang", cfg.get("output_lang", "eng_Latn")))
        vsd = data.get("vector_source_domain", cfg.get("vector_source_domain"))
        langs = [str(x) for x in data.get("languages", [])]
        if len(langs) < 2:
            continue

        concept_vectors = load_concept_vectors_by_lang(
            vectors_dir, domain, method, device,
            vector_source_domain=str(vsd) if vsd else None,
        )
        stim_path = stimuli_dir / f"{domain}_pairs.json"
        if not stim_path.exists():
            continue
        stimuli = _load_stimuli(stim_path)

        try:
            anchor = build_concept_anchor(domain, output_lang, tokenizer, model, device, anchor_cache)
        except KeyError:
            continue

        baseline_cache: Dict[str, List[torch.Tensor]] = {}
        n_cells = len(langs) * len(langs)
        cell_idx = 0
        for vec_lang in langs:
            lang_vecs = concept_vectors.get(vec_lang, {})
            vecs_i = {l: lang_vecs[l] for l in layers if l in lang_vecs}
            if not vecs_i:
                continue
            for stim_lang in langs:
                cell_idx += 1
                sentences = _iter_positive_sentences(stimuli, stim_lang)
                if max_sentences_per_cell > 0:
                    sentences = sentences[:max_sentences_per_cell]
                if not sentences:
                    continue
                pair_label = f"{stim_lang.split('_')[0]}→{output_lang.split('_')[0]}"
                cond = f"vector_{vec_lang}"
                if cell_idx == 1 or cell_idx % max(1, log_every) == 0 or cell_idx == n_cells:
                    _log(
                        f"Exp4 {json_path.name}: cell {cell_idx}/{n_cells} ({cond}, {pair_label})"
                    )

                if stim_lang not in baseline_cache:
                    baseline_cache[stim_lang] = run_translations(
                        model,
                        tokenizer,
                        stim_lang,
                        output_lang,
                        sentences,
                        None,
                        device,
                        batch_size=generation_batch_size,
                    )
                baseline_ids = baseline_cache[stim_lang]
                hook = InterventionHook()
                hook.register_vector_subtraction_hook(model, vecs_i, layers, alpha=alpha)
                ablated_ids = run_translations(
                    model,
                    tokenizer,
                    stim_lang,
                    output_lang,
                    sentences,
                    hook,
                    device,
                    batch_size=generation_batch_size,
                )
                hook.cleanup()

                stats = score_anchor_metric(
                    baseline_ids, ablated_ids, tokenizer, model, anchor,
                    presence_threshold, deletion_mode, deletion_threshold,
                    deletion_relative_margin, low_gate_threshold,
                )
                primary.append(_primary_row("exp4", domain, method, pair_label, cond, stats))
                diverge.append({
                    "experiment": "exp4",
                    "domain": domain,
                    "vector_method": method,
                    "pair_label": pair_label,
                    "condition": cond,
                    "intervention_divergence_rate": _fmt_csv_float(
                        intervention_divergence_rate(
                            baseline_ids, ablated_ids, tokenizer, model, divergence_tau,
                        )
                    ),
                })

    return primary, diverge


PRIMARY_FIELDS = [
    "experiment",
    "domain",
    "vector_method",
    "pair_label",
    "condition",
    "gate_pass_rate",
    "cosine_deletion_rate",
    "mean_b_anchor",
    "mean_a_anchor",
    "low_gate_warning",
]

VALIDATION_FIELDS = PRIMARY_FIELDS + ["token_deletion_rate"]

DIVERGENCE_FIELDS = [
    "experiment",
    "domain",
    "vector_method",
    "pair_label",
    "condition",
    "intervention_divergence_rate",
]


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def run_cosine_concept_deletion(
    results_dir: str,
    vectors_dir: str,
    stimuli_dir: str,
    output_csv: str,
    validation_csv: str,
    exp1_json_dir: str,
    device: str,
    presence_threshold: float,
    deletion_threshold: float,
    deletion_mode: str,
    deletion_relative_margin: float,
    low_gate_threshold: float,
    max_sentences: int,
    validate_exp1_only: bool,
    skip_exp1_validation: bool,
    write_divergence_csv: Optional[str],
    divergence_tau: float,
    log_every: int = 1,
    generation_batch_size: int = 8,
    max_random_trials: int = 0,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    results_path = Path(results_dir)
    vectors_path = Path(vectors_dir)
    stimuli_path = Path(stimuli_dir)
    exp1_dir = Path(exp1_json_dir)

    _log("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
    model.eval()
    hidden_dim = int(model.config.d_model)
    _log(f"Model ready on device={device}; hidden_dim={hidden_dim}")

    val_path: Optional[Path] = None
    if not skip_exp1_validation:
        domains = ["sacred", "kinship"]
        _log("Starting Exp1 validation pass")
        val_rows = run_exp1_validation(
            vectors_path, stimuli_path, exp1_dir, tokenizer, model, device,
            presence_threshold, deletion_mode, deletion_threshold,
            deletion_relative_margin, low_gate_threshold,
            max_sentences, domains, log_every=log_every, generation_batch_size=generation_batch_size,
        )
        val_path = write_csv(Path(validation_csv), val_rows, VALIDATION_FIELDS)
        if validate_exp1_only:
            return val_path, None, None
        if val_rows:
            print(f"Exp1 validation rows written: {val_path} (n={len(val_rows)})")
    elif validate_exp1_only:
        raise ValueError("--validate-exp1-only requires Exp1 validation (do not combine with --skip-exp1-validation)")

    _log("Starting Exp2 cosine rows")
    exp2_primary, exp2_div = run_exp2_rows(
        results_path, vectors_path, stimuli_path, tokenizer, model, device,
        presence_threshold, deletion_mode, deletion_threshold,
        deletion_relative_margin, low_gate_threshold,
        max_sentences, hidden_dim, divergence_tau, log_every=log_every,
        generation_batch_size=generation_batch_size, max_random_trials=max_random_trials,
    )
    _log("Starting Exp4 cosine rows")
    exp4_primary, exp4_div = run_exp4_rows(
        results_path, vectors_path, stimuli_path, tokenizer, model, device,
        presence_threshold, deletion_mode, deletion_threshold,
        deletion_relative_margin, low_gate_threshold,
        max_sentences, divergence_tau, log_every=log_every,
        generation_batch_size=generation_batch_size,
    )
    all_primary = exp2_primary + exp4_primary
    if not all_primary:
        raise FileNotFoundError(
            "No Exp2/Exp4 cosine rows produced. Check results/json and results/*/json/exp4_transfer_summary_*.json",
        )
    out_main = write_csv(Path(output_csv), all_primary, PRIMARY_FIELDS)
    _log(f"Wrote primary CSV with {len(all_primary)} rows: {out_main}")

    out_div: Optional[Path] = None
    if write_divergence_csv:
        all_div = exp2_div + exp4_div
        out_div = write_csv(Path(write_divergence_csv), all_div, DIVERGENCE_FIELDS)
        _log(f"Wrote divergence CSV with {len(all_div)} rows: {out_div}")

    return val_path, out_main, out_div
