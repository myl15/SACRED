"""
Token-level max-cosine concept deletion metrics for Exp5.

This module replaces sentence-mean embedding cosine with a token-position
max cosine metric against domain/language anchors built from shared embeddings.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _get_sacred_vocabulary_by_language(lang: str) -> List[str]:
    base = CONCEPT_VOCABULARIES.get("sacred", {})
    if lang in base:
        return list(base[lang])
    return list(base.get("eng_Latn", []))


def _get_kinship_vocabulary_by_language(lang: str) -> List[str]:
    # Includes required concepts and common morphological variants.
    vocab = {
        "eng_Latn": [
            "mother", "mothers", "maternal", "mom", "mum",
            "father", "fathers", "paternal", "dad",
            "family", "families", "familial", "household",
            "child", "children", "kid", "kids", "offspring",
            "ancestor", "ancestors", "ancestral",
            "parent", "parents", "parental",
        ],
        "spa_Latn": [
            "madre", "madres", "materno", "materna", "mamá",
            "padre", "padres", "paterno", "paterna", "papá",
            "familia", "familias", "familiar",
            "niño", "niña", "niños", "niñas", "hijo", "hija", "hijos", "hijas",
            "antepasado", "antepasados", "ancestral",
            "progenitor", "progenitora", "progenitores", "padres",
        ],
        "arb_Arab": [
            "أم", "الأم", "أمهات", "أمومة",
            "أب", "الأب", "آباء", "أبوة",
            "عائلة", "الأسرة", "أسر", "عائلي",
            "طفل", "أطفال", "ابن", "ابنة", "أبناء", "بنات",
            "سلف", "أسلاف", "الأجداد", "جد", "جدة", "أجداد",
            "والد", "والدة", "والدان", "والدين",
        ],
        "zho_Hant": [
            "母親", "母亲", "媽媽", "母系",
            "父親", "父亲", "爸爸", "父系",
            "家庭", "家族", "家人",
            "孩子", "兒童", "小孩", "子女", "兒子", "女兒",
            "祖先", "先祖", "祖輩", "祖辈",
            "父母", "雙親", "家長",
        ],
    }
    if lang in vocab:
        return vocab[lang]
    base = CONCEPT_VOCABULARIES.get("kinship", {})
    return list(base.get(lang, base.get("eng_Latn", [])))


def get_concept_vocabulary(domain: str, lang: str) -> List[str]:
    domain = str(domain)
    if domain == "sacred":
        return _get_sacred_vocabulary_by_language(lang)
    if domain == "kinship":
        return _get_kinship_vocabulary_by_language(lang)
    # fallback to configured vocab if any
    return list(CONCEPT_VOCABULARIES.get(domain, {}).get(lang, []))


def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    return F.normalize(vec, dim=-1, p=2)


def _tokenize_term_ids(term: str, tokenizer: Any) -> List[int]:
    ids = tokenizer.encode(term, add_special_tokens=False)
    return [int(i) for i in ids]


@dataclass
class AnchorBundle:
    anchor: torch.Tensor
    term_vectors: torch.Tensor  # [n_terms, hidden_dim], normalized
    terms_used: List[str]


def build_concept_anchor(
    domain: str,
    language: str,
    tokenizer: Any,
    model: Any,
    cache: Dict[Tuple[str, str], AnchorBundle],
) -> AnchorBundle:
    key = (domain, language)
    if key in cache:
        return cache[key]

    words = get_concept_vocabulary(domain, language)
    if not words:
        raise KeyError(f"No vocabulary for domain={domain} language={language}")

    emb = model.model.shared.weight  # [vocab, hidden]
    term_vecs: List[torch.Tensor] = []
    terms_used: List[str] = []
    for w in words:
        ids = _tokenize_term_ids(w, tokenizer)
        if not ids:
            continue
        term_vec = emb[torch.tensor(ids, device=emb.device)].mean(dim=0)
        term_vecs.append(term_vec)
        terms_used.append(w)
    if not term_vecs:
        raise ValueError(f"All vocab terms tokenized to empty ids for domain={domain}, lang={language}")
    term_stack = torch.stack(term_vecs, dim=0)
    term_stack = _l2_normalize(term_stack)
    anchor = _l2_normalize(term_stack.mean(dim=0))
    bundle = AnchorBundle(anchor=anchor, term_vectors=term_stack, terms_used=terms_used)
    cache[key] = bundle
    return bundle


def mean_pairwise_cosine(term_vectors: torch.Tensor) -> float:
    n = int(term_vectors.shape[0])
    if n < 2:
        return float("nan")
    sims = term_vectors @ term_vectors.T
    mask = ~torch.eye(n, dtype=torch.bool, device=sims.device)
    vals = sims[mask]
    return float(vals.mean().item()) if vals.numel() else float("nan")


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
        batch = sentences[i : i + max(1, batch_size)]
        inputs = tokenizer(
            batch,
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
            generated.append(outputs[j].detach())
    return generated


def token_max_similarity(
    token_ids: torch.Tensor,
    anchor: torch.Tensor,
    emb_weight: torch.Tensor,
    special_ids: set[int],
    tokenizer: Any,
    blocked_token_ids: set[int],
    blocked_token_strs: set[str],
    require_content_tokens: bool,
) -> float:
    ids = _filter_scored_token_ids(
        token_ids=token_ids,
        tokenizer=tokenizer,
        special_ids=special_ids,
        blocked_token_ids=blocked_token_ids,
        blocked_token_strs=blocked_token_strs,
        require_content_tokens=require_content_tokens,
    )
    if not ids:
        return float("-inf")
    token_vecs = emb_weight[torch.tensor(ids, device=emb_weight.device)]
    token_vecs = _l2_normalize(token_vecs)
    sims = token_vecs @ anchor
    return float(sims.max().item())


def _looks_like_content_token(tok: str) -> bool:
    t = tok.replace("▁", "").strip()
    if not t:
        return False
    has_letter = any(ch.isalpha() for ch in t)
    has_digit = any(ch.isdigit() for ch in t)
    return has_letter and not has_digit


def _normalize_blocked_token_str(tok: str) -> str:
    return str(tok).strip()


def _filter_scored_token_ids(
    token_ids: torch.Tensor,
    tokenizer: Any,
    special_ids: set[int],
    blocked_token_ids: set[int],
    blocked_token_strs: set[str],
    require_content_tokens: bool,
) -> List[int]:
    out: List[int] = []
    for tok in token_ids.tolist():
        tid = int(tok)
        if tid in special_ids or tid in blocked_token_ids:
            continue
        tstr = tokenizer.convert_ids_to_tokens([tid])[0]
        if _normalize_blocked_token_str(tstr) in blocked_token_strs:
            continue
        if require_content_tokens and (not _looks_like_content_token(tstr)):
            continue
        out.append(tid)
    return out


def token_max_similarity_details(
    token_ids: torch.Tensor,
    anchor: torch.Tensor,
    emb_weight: torch.Tensor,
    special_ids: set[int],
    tokenizer: Any,
    blocked_token_ids: set[int],
    blocked_token_strs: set[str],
    require_content_tokens: bool,
) -> Dict[str, Any]:
    pairs: List[Tuple[int, int]] = []
    for idx, tok in enumerate(token_ids.tolist()):
        tid = int(tok)
        if tid in special_ids or tid in blocked_token_ids:
            continue
        tstr = tokenizer.convert_ids_to_tokens([tid])[0]
        if _normalize_blocked_token_str(tstr) in blocked_token_strs:
            continue
        if require_content_tokens and (not _looks_like_content_token(tstr)):
            continue
        pairs.append((idx, tid))
    if not pairs:
        return {
            "max_sim": float("-inf"),
            "max_pos": -1,
            "max_token_id": -1,
            "max_token_str": "",
            "top2_token_id": -1,
            "top2_token_str": "",
            "top2_sim": float("nan"),
            "top3_token_id": -1,
            "top3_token_str": "",
            "top3_sim": float("nan"),
            "n_scored_tokens": 0,
        }
    kept_ids = [tok for _, tok in pairs]
    kept_pos = [pos for pos, _ in pairs]
    vecs = emb_weight[torch.tensor(kept_ids, device=emb_weight.device)]
    vecs = _l2_normalize(vecs)
    sims = vecs @ anchor
    order = torch.argsort(sims, descending=True)
    max_idx = int(order[0].item())
    max_sim = float(sims[max_idx].item())
    max_pos = int(kept_pos[max_idx])
    max_token_id = int(kept_ids[max_idx])
    tok_str = tokenizer.convert_ids_to_tokens([max_token_id])[0]
    if len(order) > 1:
        i2 = int(order[1].item())
        top2_id = int(kept_ids[i2])
        top2_str = tokenizer.convert_ids_to_tokens([top2_id])[0]
        top2_sim = float(sims[i2].item())
    else:
        top2_id, top2_str, top2_sim = -1, "", float("nan")
    if len(order) > 2:
        i3 = int(order[2].item())
        top3_id = int(kept_ids[i3])
        top3_str = tokenizer.convert_ids_to_tokens([top3_id])[0]
        top3_sim = float(sims[i3].item())
    else:
        top3_id, top3_str, top3_sim = -1, "", float("nan")
    return {
        "max_sim": max_sim,
        "max_pos": max_pos,
        "max_token_id": max_token_id,
        "max_token_str": str(tok_str),
        "top2_token_id": top2_id,
        "top2_token_str": str(top2_str),
        "top2_sim": top2_sim,
        "top3_token_id": top3_id,
        "top3_token_str": str(top3_str),
        "top3_sim": top3_sim,
        "n_scored_tokens": len(kept_ids),
    }


def score_max_cosine_metric(
    baseline_ids: List[torch.Tensor],
    ablated_ids: List[torch.Tensor],
    anchor: torch.Tensor,
    emb_weight: torch.Tensor,
    special_ids: set[int],
    tokenizer: Any,
    blocked_token_ids: set[int],
    blocked_token_strs: set[str],
    require_content_tokens: bool,
    presence_threshold: float,
    deletion_threshold: float,
    low_gate_threshold: float,
) -> Dict[str, float | bool]:
    n = min(len(baseline_ids), len(ablated_ids))
    if n == 0:
        return {
            "gate_pass_rate": float("nan"),
            "n_gate_passed": 0,
            "cosine_deletion_rate": float("nan"),
            "mean_b_max": float("nan"),
            "mean_a_max": float("nan"),
            "low_gate": True,
        }
    b_vals: List[float] = []
    a_vals: List[float] = []
    gated: List[int] = []
    for i in range(n):
        b_max = token_max_similarity(
            baseline_ids[i],
            anchor,
            emb_weight,
            special_ids,
            tokenizer,
            blocked_token_ids,
            blocked_token_strs,
            require_content_tokens,
        )
        a_max = token_max_similarity(
            ablated_ids[i],
            anchor,
            emb_weight,
            special_ids,
            tokenizer,
            blocked_token_ids,
            blocked_token_strs,
            require_content_tokens,
        )
        b_vals.append(b_max)
        a_vals.append(a_max)
        if b_max >= presence_threshold:
            gated.append(1 if a_max < deletion_threshold else 0)
    n_gate = len(gated)
    gate_pass_rate = n_gate / n
    low_gate = bool(gate_pass_rate < low_gate_threshold)
    if n_gate == 0 or low_gate:
        cos_del = float("nan")
    else:
        cos_del = float(np.mean(gated))
    return {
        "gate_pass_rate": float(gate_pass_rate),
        "n_gate_passed": int(n_gate),
        "cosine_deletion_rate": cos_del,
        "mean_b_max": float(np.mean(b_vals)),
        "mean_a_max": float(np.mean(a_vals)),
        "low_gate": low_gate,
    }


def _fmt_csv_float(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.6f}"


def _iter_positive_sentences(
    stimuli: Dict[str, Dict[str, List[Dict[str, str]]]],
    lang: str,
) -> List[str]:
    out: List[str] = []
    for concept_pairs in stimuli.get(lang, {}).values():
        out.extend([p["positive"] for p in concept_pairs if "positive" in p])
    return out


def _primary_row(
    experiment: str,
    domain: str,
    vector_method: str,
    pair_label: str,
    condition: str,
    stats: Dict[str, float | bool],
) -> Dict[str, str]:
    return {
        "experiment": experiment,
        "domain": domain,
        "vector_method": vector_method,
        "pair_label": pair_label,
        "condition": condition,
        "gate_pass_rate": _fmt_csv_float(float(stats["gate_pass_rate"])),
        "n_gate_passed": str(int(stats["n_gate_passed"])),
        "cosine_deletion_rate": _fmt_csv_float(float(stats["cosine_deletion_rate"])),
        "mean_b_max": _fmt_csv_float(float(stats["mean_b_max"])),
        "mean_a_max": _fmt_csv_float(float(stats["mean_a_max"])),
        "low_gate": str(bool(stats["low_gate"])).lower(),
    }


def sample_random_pivot_vector(
    concept_vectors: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    hidden_dim: int,
    random_seed: int,
    trial_index: int,
    device: torch.device,
) -> torch.Tensor:
    vec_c = _get_layer_mean_vector(concept_vectors, "eng_Latn", layers).to(device)
    gen_cpu = torch.Generator()
    gen_cpu.manual_seed(int(random_seed) + int(trial_index) * 1000003)
    vec_d = torch.randn(hidden_dim, generator=gen_cpu, device="cpu").to(device)
    vec_d = vec_d / vec_d.norm() * vec_c.norm()
    return vec_d


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


PRIMARY_FIELDS = [
    "experiment",
    "domain",
    "vector_method",
    "pair_label",
    "condition",
    "gate_pass_rate",
    "n_gate_passed",
    "cosine_deletion_rate",
    "mean_b_max",
    "mean_a_max",
    "low_gate",
]

COHERENCE_FIELDS = [
    "domain",
    "language",
    "n_terms",
    "mean_pairwise_cosine",
    "coherence_flag",
]

CALIBRATION_FIELDS = [
    "dataset",
    "domain",
    "pair_label",
    "condition",
    "gate_pass_rate",
    "cosine_deletion_rate",
    "mean_b_max",
    "mean_a_max",
    "token_deletion_rate",
    "directional_agreement",
]

CALIBRATION_DEBUG_FIELDS = [
    "domain",
    "concept",
    "sentence_idx",
    "baseline_max_sim",
    "baseline_max_pos",
    "baseline_max_token_id",
    "baseline_max_token_str",
    "baseline_n_scored_tokens",
    "ablated_max_sim",
    "ablated_max_pos",
    "ablated_max_token_id",
    "ablated_max_token_str",
    "ablated_top2_token_id",
    "ablated_top2_token_str",
    "ablated_top2_sim",
    "ablated_top3_token_id",
    "ablated_top3_token_str",
    "ablated_top3_sim",
    "ablated_n_scored_tokens",
    "gate_pass",
    "deleted",
]

CALIBRATION_DEBUG_SUMMARY_FIELDS = [
    "side",
    "token_id",
    "token_str",
    "count",
    "fraction",
]

PI_FIELDS = [
    "domain",
    "vector_method",
    "pair_label",
    "token_pi",
    "cosine_pi",
    "token_underpowered",
    "cosine_underpowered",
]


def run_vocabulary_coherence(
    tokenizer: Any,
    model: Any,
    domains: List[str],
    languages: List[str],
) -> List[Dict[str, str]]:
    cache: Dict[Tuple[str, str], AnchorBundle] = {}
    rows: List[Dict[str, str]] = []
    for domain in domains:
        for lang in languages:
            try:
                bundle = build_concept_anchor(domain, lang, tokenizer, model, cache)
            except Exception:
                continue
            mp = mean_pairwise_cosine(bundle.term_vectors)
            if math.isnan(mp):
                flag = "nan"
            elif mp > 0.5:
                flag = "tight"
            elif mp < 0.3:
                flag = "loose"
            else:
                flag = "mid"
            rows.append(
                {
                    "domain": domain,
                    "language": lang,
                    "n_terms": str(len(bundle.terms_used)),
                    "mean_pairwise_cosine": _fmt_csv_float(mp),
                    "coherence_flag": flag,
                }
            )
    return rows


def run_exp1_calibration(
    vectors_dir: Path,
    stimuli_dir: Path,
    exp1_json_dir: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    presence_threshold: float,
    deletion_threshold: float,
    low_gate_threshold: float,
    generation_batch_size: int,
    max_sentences: int,
    log_every: int,
    debug_mode: bool = False,
    debug_sentence_cap: int = 100,
    blocked_token_ids: Optional[set[int]] = None,
    blocked_token_strs: Optional[set[str]] = None,
    require_content_tokens: bool = True,
) -> Tuple[List[Dict[str, str]], bool, List[Dict[str, str]], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    debug_rows: List[Dict[str, str]] = []
    anchor_cache: Dict[Tuple[str, str], AnchorBundle] = {}
    emb = model.model.shared.weight
    special_ids = set(int(x) for x in tokenizer.all_special_ids if x is not None)
    blocked_token_ids = blocked_token_ids or set()
    blocked_token_strs = blocked_token_strs or set()
    baseline_tok_counts: Dict[Tuple[int, str], int] = {}
    ablated_tok_counts: Dict[Tuple[int, str], int] = {}

    domain = "sacred"
    src = "eng_Latn"
    tgt = "eng_Latn"
    exp1_path = exp1_json_dir / "exp1_sacred_deletion.json"
    if not exp1_path.exists():
        _log(f"Calibration skipped: missing {exp1_path}")
        return rows, False, debug_rows, []
    data = _load_json(exp1_path)
    eng_block = data.get("deletion_results", {}).get(src, {})
    stim_path = stimuli_dir / "sacred_pairs.json"
    if not stim_path.exists():
        _log(f"Calibration skipped: missing {stim_path}")
        return rows, False, debug_rows, []
    stimuli = _load_stimuli(stim_path).get(src, {})
    vec_path = vectors_dir / "sacred_eng_Latn.pt"
    if not vec_path.exists():
        _log(f"Calibration skipped: missing {vec_path}")
        return rows, False, debug_rows, []
    raw_vecs = torch.load(vec_path, map_location=device)

    anchor = build_concept_anchor(domain, tgt, tokenizer, model, anchor_cache).anchor
    items = list(eng_block.items())
    n_items = len(items)
    for idx, (concept, token_metrics) in enumerate(items, start=1):
        if idx == 1 or idx % max(1, log_every) == 0 or idx == n_items:
            _log(f"Calibration concept {idx}/{n_items}: {concept}")
        concept_pairs = stimuli.get(concept, [])
        sentences = [p["positive"] for p in concept_pairs if "positive" in p]
        if max_sentences > 0:
            sentences = sentences[:max_sentences]
        if not sentences:
            continue
        layer_dict = raw_vecs.get(concept, {})
        if not layer_dict:
            continue
        keys = sorted(layer_dict.keys(), key=lambda k: int(str(k)))
        mean_vec = torch.stack([layer_dict[k] for k in keys]).mean(dim=0).to(device)

        baseline = run_translations(model, tokenizer, src, tgt, sentences, None, device, generation_batch_size)
        hook = InterventionHook()
        hook.register_vector_subtraction_hook(model, mean_vec, list(INTERVENTION_LAYERS), alpha=1.0)
        ablated = run_translations(model, tokenizer, src, tgt, sentences, hook, device, generation_batch_size)
        hook.cleanup()

        st = score_max_cosine_metric(
            baseline, ablated, anchor, emb, special_ids,
            tokenizer, blocked_token_ids, blocked_token_strs, require_content_tokens,
            presence_threshold, deletion_threshold, low_gate_threshold,
        )
        token_del = _safe_float(token_metrics.get("deletion_rate"), 0.0)
        cos_del = float(st["cosine_deletion_rate"])
        directional = not math.isnan(cos_del) and ((cos_del > 0) == (token_del > 0))
        rows.append(
            {
                "dataset": "exp1",
                "domain": domain,
                "pair_label": "eng->eng",
                "condition": f"concept_{concept}",
                "gate_pass_rate": _fmt_csv_float(float(st["gate_pass_rate"])),
                "cosine_deletion_rate": _fmt_csv_float(cos_del),
                "mean_b_max": _fmt_csv_float(float(st["mean_b_max"])),
                "mean_a_max": _fmt_csv_float(float(st["mean_a_max"])),
                "token_deletion_rate": _fmt_csv_float(token_del),
                "directional_agreement": str(directional).lower(),
            }
        )
        if debug_mode:
            n_debug = min(len(baseline), len(ablated), max(0, debug_sentence_cap))
            for s_idx in range(n_debug):
                b_det = token_max_similarity_details(
                    baseline[s_idx],
                    anchor,
                    emb,
                    special_ids,
                    tokenizer,
                    blocked_token_ids,
                    blocked_token_strs,
                    require_content_tokens,
                )
                a_det = token_max_similarity_details(
                    ablated[s_idx],
                    anchor,
                    emb,
                    special_ids,
                    tokenizer,
                    blocked_token_ids,
                    blocked_token_strs,
                    require_content_tokens,
                )
                gate_pass = bool(b_det["max_sim"] >= presence_threshold)
                deleted = bool(gate_pass and (a_det["max_sim"] < deletion_threshold))
                b_key = (int(b_det["max_token_id"]), str(b_det["max_token_str"]))
                a_key = (int(a_det["max_token_id"]), str(a_det["max_token_str"]))
                baseline_tok_counts[b_key] = baseline_tok_counts.get(b_key, 0) + 1
                ablated_tok_counts[a_key] = ablated_tok_counts.get(a_key, 0) + 1
                debug_rows.append(
                    {
                        "domain": domain,
                        "concept": concept,
                        "sentence_idx": str(s_idx),
                        "baseline_max_sim": _fmt_csv_float(float(b_det["max_sim"])),
                        "baseline_max_pos": str(int(b_det["max_pos"])),
                        "baseline_max_token_id": str(int(b_det["max_token_id"])),
                        "baseline_max_token_str": str(b_det["max_token_str"]),
                        "baseline_n_scored_tokens": str(int(b_det["n_scored_tokens"])),
                        "ablated_max_sim": _fmt_csv_float(float(a_det["max_sim"])),
                        "ablated_max_pos": str(int(a_det["max_pos"])),
                        "ablated_max_token_id": str(int(a_det["max_token_id"])),
                        "ablated_max_token_str": str(a_det["max_token_str"]),
                        "ablated_top2_token_id": str(int(a_det["top2_token_id"])),
                        "ablated_top2_token_str": str(a_det["top2_token_str"]),
                        "ablated_top2_sim": _fmt_csv_float(float(a_det["top2_sim"])),
                        "ablated_top3_token_id": str(int(a_det["top3_token_id"])),
                        "ablated_top3_token_str": str(a_det["top3_token_str"]),
                        "ablated_top3_sim": _fmt_csv_float(float(a_det["top3_sim"])),
                        "ablated_n_scored_tokens": str(int(a_det["n_scored_tokens"])),
                        "gate_pass": str(gate_pass).lower(),
                        "deleted": str(deleted).lower(),
                    }
                )

    sacred_gate = [
        float(r["gate_pass_rate"])
        for r in rows
        if r["domain"] == "sacred" and r["pair_label"] == "eng->eng" and r["gate_pass_rate"] != "nan"
    ]
    mean_gate = float(np.mean(sacred_gate)) if sacred_gate else float("nan")
    passed = bool((not math.isnan(mean_gate)) and mean_gate >= 0.8)
    _log(f"Calibration sacred eng diagonal mean gate_pass_rate={mean_gate:.4f}; pass={passed}")
    summary_rows: List[Dict[str, str]] = []
    if debug_mode and debug_rows:
        b_total = sum(baseline_tok_counts.values())
        a_total = sum(ablated_tok_counts.values())
        for (tid, tstr), c in sorted(baseline_tok_counts.items(), key=lambda x: -x[1]):
            summary_rows.append(
                {
                    "side": "baseline",
                    "token_id": str(tid),
                    "token_str": tstr,
                    "count": str(c),
                    "fraction": _fmt_csv_float(c / b_total if b_total else float("nan")),
                }
            )
        for (tid, tstr), c in sorted(ablated_tok_counts.items(), key=lambda x: -x[1]):
            summary_rows.append(
                {
                    "side": "ablated",
                    "token_id": str(tid),
                    "token_str": tstr,
                    "count": str(c),
                    "fraction": _fmt_csv_float(c / a_total if a_total else float("nan")),
                }
            )
        top_baseline = sorted(baseline_tok_counts.items(), key=lambda x: -x[1])[:5]
        top_ablated = sorted(ablated_tok_counts.items(), key=lambda x: -x[1])[:5]
        _log(f"Calibration debug top baseline max tokens: {top_baseline}")
        _log(f"Calibration debug top ablated max tokens: {top_ablated}")
    return rows, passed, debug_rows, summary_rows


def run_exp2_rows(
    results_dir: Path,
    vectors_dir: Path,
    stimuli_dir: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    presence_threshold: float,
    deletion_threshold: float,
    low_gate_threshold: float,
    max_sentences: int,
    generation_batch_size: int,
    log_every: int,
    random_trials: int,
    min_valid_random_trials: int,
    blocked_token_ids: set[int],
    blocked_token_strs: set[str],
    require_content_tokens: bool,
) -> Tuple[List[Dict[str, str]], Dict[Tuple[str, str, str], Dict[str, float]], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    pi_accum: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    token_pi_rows: List[Dict[str, str]] = []
    anchor_cache: Dict[Tuple[str, str], AnchorBundle] = {}
    emb = model.model.shared.weight
    special_ids = set(int(x) for x in tokenizer.all_special_ids if x is not None)
    hidden_dim = int(model.config.d_model)
    dev = torch.device(device)

    exp2_jsons = sorted((results_dir / "json").glob("exp2_pivot_*.json"))
    _log(f"Exp2 files: {len(exp2_jsons)}")
    for j_idx, json_path in enumerate(exp2_jsons, start=1):
        _log(f"Exp2 file {j_idx}/{len(exp2_jsons)}: {json_path.name}")
        data = _load_json(json_path)
        md = data.get("metadata", {})
        domain = str(md.get("domain", "kinship"))
        method = str(md.get("vector_method", "mean"))
        alpha = _safe_float(md.get("alpha"), 0.25)
        layers = [int(x) for x in md.get("layers", INTERVENTION_LAYERS)]
        vsd = md.get("vector_source_domain")
        n_random = max(1, int(random_trials))
        random_seed = int(md.get("random_seed", 42))

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
        for p_idx, (pair_key, pair_res) in enumerate(pair_items, start=1):
            src, tgt = _parse_pair_key(pair_key)
            pair_label = f"{src.split('_')[0]}->{tgt.split('_')[0]}"
            if p_idx == 1 or p_idx % max(1, log_every) == 0 or p_idx == len(pair_items):
                _log(f"Exp2 {json_path.name}: pair {p_idx}/{len(pair_items)} {pair_label}")
            sentences = _iter_positive_sentences(stimuli, src)
            if max_sentences > 0:
                sentences = sentences[:max_sentences]
            if not sentences:
                continue
            try:
                anchor = build_concept_anchor(domain, tgt, tokenizer, model, anchor_cache).anchor
            except Exception:
                continue

            baseline = run_translations(model, tokenizer, src, tgt, sentences, None, device, generation_batch_size)

            def run_cond(vec: torch.Tensor) -> List[torch.Tensor]:
                hook = InterventionHook()
                hook.register_vector_subtraction_hook(model, vec, layers, alpha=alpha)
                out = run_translations(model, tokenizer, src, tgt, sentences, hook, device, generation_batch_size)
                hook.cleanup()
                return out

            vec_a = _get_layer_mean_vector(concept_vectors, src, layers)
            vec_b = _get_layer_mean_vector(concept_vectors, tgt, layers)
            vec_c = _get_layer_mean_vector(concept_vectors, "eng_Latn", layers).to(device)

            cond_to_vec = {
                "A_source": vec_a,
                "B_target": vec_b,
                "C_english": vec_c,
            }
            cond_scores: Dict[str, float] = {}
            token_scores: Dict[str, float] = {}

            token_scores["A_source"] = _safe_float(pair_res.get("condition_A_source", {}).get("deletion_rate"), float("nan"))
            token_scores["B_target"] = _safe_float(pair_res.get("condition_B_target", {}).get("deletion_rate"), float("nan"))
            token_scores["C_english"] = _safe_float(pair_res.get("condition_C_english", {}).get("deletion_rate"), float("nan"))
            token_scores["D_random"] = _safe_float(pair_res.get("condition_D_random", {}).get("deletion_rate"), float("nan"))

            for cond, vec in cond_to_vec.items():
                ablated = run_cond(vec)
                st = score_max_cosine_metric(
                    baseline, ablated, anchor, emb, special_ids,
                    tokenizer, blocked_token_ids, blocked_token_strs, require_content_tokens,
                    presence_threshold, deletion_threshold, low_gate_threshold,
                )
                rows.append(_primary_row("exp2", domain, method, pair_label, cond, st))
                cond_scores[cond] = float(st["cosine_deletion_rate"])

            d_rates: List[float] = []
            d_b: List[float] = []
            d_a: List[float] = []
            d_gate: List[float] = []
            d_n_gate: List[int] = []
            n_valid_trials = 0
            for t in range(n_random):
                if t == 0 or (t + 1) % max(1, log_every) == 0 or t + 1 == n_random:
                    _log(f"Exp2 {pair_label} D_random trial {t+1}/{n_random}")
                vec_d = sample_random_pivot_vector(concept_vectors, layers, hidden_dim, random_seed, t, dev)
                ablated_d = run_cond(vec_d)
                st = score_max_cosine_metric(
                    baseline, ablated_d, anchor, emb, special_ids,
                    tokenizer, blocked_token_ids, blocked_token_strs, require_content_tokens,
                    presence_threshold, deletion_threshold, low_gate_threshold,
                )
                d_gate.append(float(st["gate_pass_rate"]))
                d_n_gate.append(int(st["n_gate_passed"]))
                d_b.append(float(st["mean_b_max"]))
                d_a.append(float(st["mean_a_max"]))
                cdr = float(st["cosine_deletion_rate"])
                trial_valid = (not bool(st["low_gate"])) and (not math.isnan(cdr))
                if trial_valid:
                    d_rates.append(cdr)
                    n_valid_trials += 1
            enough_valid_trials = n_valid_trials >= max(1, int(min_valid_random_trials))
            if not enough_valid_trials:
                _log(
                    f"Exp2 {pair_label} D_random underpowered: "
                    f"valid_trials={n_valid_trials}/{n_random} < min_valid={min_valid_random_trials}"
                )
            d_stats = {
                "gate_pass_rate": float(np.mean(d_gate)) if d_gate else float("nan"),
                "n_gate_passed": int(round(float(np.mean(d_n_gate)))) if d_n_gate else 0,
                "cosine_deletion_rate": float(np.mean(d_rates)) if d_rates and enough_valid_trials else float("nan"),
                "mean_b_max": float(np.mean(d_b)) if d_b else float("nan"),
                "mean_a_max": float(np.mean(d_a)) if d_a else float("nan"),
                "low_gate": (not enough_valid_trials) or ((float(np.mean(d_gate)) if d_gate else 0.0) < low_gate_threshold),
            }
            rows.append(_primary_row("exp2", domain, method, pair_label, "D_random", d_stats))
            cond_scores["D_random"] = float(d_stats["cosine_deletion_rate"])

            key = (domain, method, pair_label)
            pi_accum[key] = cond_scores

            denom_token = 0.5 * (token_scores["A_source"] + token_scores["B_target"]) - token_scores["D_random"]
            token_underpowered = bool(math.isnan(denom_token) or denom_token < 0.05)
            if token_underpowered:
                token_pi = float("nan")
            else:
                token_pi = (token_scores["C_english"] - token_scores["D_random"]) / denom_token
            token_pi_rows.append(
                {
                    "domain": domain,
                    "vector_method": method,
                    "pair_label": pair_label,
                    "token_pi": _fmt_csv_float(token_pi),
                    "cosine_pi": "nan",
                    "token_underpowered": str(token_underpowered).lower(),
                    "cosine_underpowered": "true",
                }
            )

    return rows, pi_accum, token_pi_rows


def run_exp4_rows(
    results_dir: Path,
    vectors_dir: Path,
    stimuli_dir: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    presence_threshold: float,
    deletion_threshold: float,
    low_gate_threshold: float,
    max_sentences: int,
    generation_batch_size: int,
    log_every: int,
    blocked_token_ids: set[int],
    blocked_token_strs: set[str],
    require_content_tokens: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    anchor_cache: Dict[Tuple[str, str], AnchorBundle] = {}
    emb = model.model.shared.weight
    special_ids = set(int(x) for x in tokenizer.all_special_ids if x is not None)

    exp4_jsons = sorted(results_dir.glob("*/json/exp4_transfer_summary_*.json"))
    _log(f"Exp4 files: {len(exp4_jsons)}")
    for f_idx, json_path in enumerate(exp4_jsons, start=1):
        _log(f"Exp4 file {f_idx}/{len(exp4_jsons)}: {json_path.name}")
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
        if not concept_vectors:
            continue
        stim_path = stimuli_dir / f"{domain}_pairs.json"
        if not stim_path.exists():
            continue
        stimuli = _load_stimuli(stim_path)
        try:
            anchor = build_concept_anchor(domain, output_lang, tokenizer, model, anchor_cache).anchor
        except Exception:
            continue

        baseline_cache: Dict[str, List[torch.Tensor]] = {}
        n_cells = len(langs) * len(langs)
        c_idx = 0
        for vec_lang in langs:
            lang_vecs = concept_vectors.get(vec_lang, {})
            vecs = {l: lang_vecs[l] for l in layers if l in lang_vecs}
            if not vecs:
                continue
            for stim_lang in langs:
                c_idx += 1
                if c_idx == 1 or c_idx % max(1, log_every) == 0 or c_idx == n_cells:
                    _log(f"Exp4 {json_path.name}: cell {c_idx}/{n_cells}")
                sentences = _iter_positive_sentences(stimuli, stim_lang)
                if max_sentences > 0:
                    sentences = sentences[:max_sentences]
                if not sentences:
                    continue
                pair_label = f"{stim_lang.split('_')[0]}->{output_lang.split('_')[0]}"
                cond = f"vector_{vec_lang}"
                if stim_lang not in baseline_cache:
                    baseline_cache[stim_lang] = run_translations(
                        model, tokenizer, stim_lang, output_lang, sentences, None, device, generation_batch_size
                    )
                baseline = baseline_cache[stim_lang]
                hook = InterventionHook()
                hook.register_vector_subtraction_hook(model, vecs, layers, alpha=alpha)
                ablated = run_translations(
                    model, tokenizer, stim_lang, output_lang, sentences, hook, device, generation_batch_size
                )
                hook.cleanup()
                st = score_max_cosine_metric(
                    baseline, ablated, anchor, emb, special_ids,
                    tokenizer, blocked_token_ids, blocked_token_strs, require_content_tokens,
                    presence_threshold, deletion_threshold, low_gate_threshold,
                )
                rows.append(_primary_row("exp4", domain, method, pair_label, cond, st))
    return rows


def build_cosine_pi_rows(
    token_pi_rows: List[Dict[str, str]],
    cosine_scores: Dict[Tuple[str, str, str], Dict[str, float]],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    token_map = {(r["domain"], r["vector_method"], r["pair_label"]): r for r in token_pi_rows}
    keys = sorted(set(token_map.keys()) | set(cosine_scores.keys()))
    for key in keys:
        domain, method, pair_label = key
        tok = token_map.get(key, {})
        conds = cosine_scores.get(key, {})
        a = conds.get("A_source", float("nan"))
        b = conds.get("B_target", float("nan"))
        c = conds.get("C_english", float("nan"))
        d = conds.get("D_random", float("nan"))
        denom = 0.5 * (a + b) - d
        cosine_underpowered = bool(math.isnan(denom) or denom < 0.05)
        if cosine_underpowered:
            cosine_pi = float("nan")
        else:
            cosine_pi = (c - d) / denom
        out.append(
            {
                "domain": domain,
                "vector_method": method,
                "pair_label": pair_label,
                "token_pi": tok.get("token_pi", "nan"),
                "cosine_pi": _fmt_csv_float(cosine_pi),
                "token_underpowered": tok.get("token_underpowered", "true"),
                "cosine_underpowered": str(cosine_underpowered).lower(),
            }
        )
    return out


def run_token_max_cosine(
    results_dir: str,
    vectors_dir: str,
    stimuli_dir: str,
    output_csv: str,
    calibration_csv: str,
    coherence_csv: str,
    calibration_debug_csv: str,
    calibration_debug_summary_csv: str,
    pivot_comparison_csv: str,
    exp1_json_dir: str,
    device: str,
    presence_threshold: float,
    deletion_threshold: float,
    low_gate_threshold: float,
    max_sentences: int,
    generation_batch_size: int,
    random_trials: int,
    min_valid_random_trials: int,
    calibration_only: bool,
    require_calibration_pass: bool,
    log_every: int,
    debug_calibration: bool,
    debug_sentence_cap: int,
    blocked_token_ids_csv: str,
    blocked_token_strs_csv: str,
    require_content_tokens: bool,
) -> Dict[str, Optional[Path] | bool]:
    results_path = Path(results_dir)
    vectors_path = Path(vectors_dir)
    stimuli_path = Path(stimuli_dir)
    exp1_path = Path(exp1_json_dir)

    _log("Loading model/tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR).to(device)
    model.eval()
    _log(f"Model ready on {device}")

    blocked_token_ids: set[int] = set()
    blocked_token_strs: set[str] = {"▁The", "▁the"}
    if blocked_token_ids_csv.strip():
        blocked_token_ids.update(
            int(x.strip()) for x in blocked_token_ids_csv.split(",") if x.strip()
        )
    if blocked_token_strs_csv.strip():
        blocked_token_strs.update(
            _normalize_blocked_token_str(x) for x in blocked_token_strs_csv.split(",") if x.strip()
        )
    _log(
        f"Token filter config: blocked_ids={sorted(blocked_token_ids)} "
        f"blocked_strs={sorted(blocked_token_strs)} "
        f"require_content_tokens={require_content_tokens}"
    )

    coherence_rows = run_vocabulary_coherence(
        tokenizer=tokenizer,
        model=model,
        domains=["sacred", "kinship"],
        languages=EXPERIMENT_LANGUAGES,
    )
    coherence_out = write_csv(Path(coherence_csv), coherence_rows, COHERENCE_FIELDS)
    _log(f"Wrote coherence diagnostics: {coherence_out}")

    cal_rows, cal_pass, cal_debug_rows, cal_debug_summary = run_exp1_calibration(
        vectors_dir=vectors_path,
        stimuli_dir=stimuli_path,
        exp1_json_dir=exp1_path,
        tokenizer=tokenizer,
        model=model,
        device=device,
        presence_threshold=presence_threshold,
        deletion_threshold=deletion_threshold,
        low_gate_threshold=low_gate_threshold,
        generation_batch_size=generation_batch_size,
        max_sentences=max_sentences,
        log_every=log_every,
        debug_mode=debug_calibration,
        debug_sentence_cap=debug_sentence_cap,
        blocked_token_ids=blocked_token_ids,
        blocked_token_strs=blocked_token_strs,
        require_content_tokens=require_content_tokens,
    )
    cal_out = write_csv(Path(calibration_csv), cal_rows, CALIBRATION_FIELDS)
    _log(f"Wrote calibration rows: {cal_out}")
    cal_debug_out: Optional[Path] = None
    cal_debug_summary_out: Optional[Path] = None
    if debug_calibration:
        cal_debug_out = write_csv(
            Path(calibration_debug_csv), cal_debug_rows, CALIBRATION_DEBUG_FIELDS
        )
        cal_debug_summary_out = write_csv(
            Path(calibration_debug_summary_csv),
            cal_debug_summary,
            CALIBRATION_DEBUG_SUMMARY_FIELDS,
        )
        _log(f"Wrote calibration debug rows: {cal_debug_out}")
        _log(f"Wrote calibration debug summary: {cal_debug_summary_out}")
    if calibration_only:
        return {
            "coherence_csv": coherence_out,
            "calibration_csv": cal_out,
            "calibration_debug_csv": cal_debug_out,
            "calibration_debug_summary_csv": cal_debug_summary_out,
            "main_csv": None,
            "pivot_csv": None,
            "calibration_passed": cal_pass,
        }
    if require_calibration_pass and not cal_pass:
        raise RuntimeError(
            "Calibration failed: sacred eng diagonal gate_pass_rate below threshold. "
            "Lower presence threshold before full evaluation."
        )

    exp2_rows, cosine_scores, token_pi_rows = run_exp2_rows(
        results_dir=results_path,
        vectors_dir=vectors_path,
        stimuli_dir=stimuli_path,
        tokenizer=tokenizer,
        model=model,
        device=device,
        presence_threshold=presence_threshold,
        deletion_threshold=deletion_threshold,
        low_gate_threshold=low_gate_threshold,
        max_sentences=max_sentences,
        generation_batch_size=generation_batch_size,
        log_every=log_every,
        random_trials=random_trials,
        min_valid_random_trials=min_valid_random_trials,
        blocked_token_ids=blocked_token_ids,
        blocked_token_strs=blocked_token_strs,
        require_content_tokens=require_content_tokens,
    )
    exp4_rows = run_exp4_rows(
        results_dir=results_path,
        vectors_dir=vectors_path,
        stimuli_dir=stimuli_path,
        tokenizer=tokenizer,
        model=model,
        device=device,
        presence_threshold=presence_threshold,
        deletion_threshold=deletion_threshold,
        low_gate_threshold=low_gate_threshold,
        max_sentences=max_sentences,
        generation_batch_size=generation_batch_size,
        log_every=log_every,
        blocked_token_ids=blocked_token_ids,
        blocked_token_strs=blocked_token_strs,
        require_content_tokens=require_content_tokens,
    )
    all_rows = exp2_rows + exp4_rows
    if not all_rows:
        raise FileNotFoundError("No Exp2/Exp4 rows produced for token-max cosine metric")
    out_main = write_csv(Path(output_csv), all_rows, PRIMARY_FIELDS)
    _log(f"Wrote main token-max cosine CSV: {out_main} (rows={len(all_rows)})")

    pi_rows = build_cosine_pi_rows(token_pi_rows, cosine_scores)
    out_pi = write_csv(Path(pivot_comparison_csv), pi_rows, PI_FIELDS)
    _log(f"Wrote pivot comparison CSV: {out_pi}")

    return {
        "coherence_csv": coherence_out,
        "calibration_csv": cal_out,
        "calibration_debug_csv": cal_debug_out,
        "calibration_debug_summary_csv": cal_debug_summary_out,
        "main_csv": out_main,
        "pivot_csv": out_pi,
        "calibration_passed": cal_pass,
    }

