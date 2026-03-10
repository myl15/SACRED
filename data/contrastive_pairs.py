"""
Contrastive pair generation for SACRED experiments.

Replaces the original StimulusGenerator with a domain-agnostic
ContrastivePairGenerator that outputs paired (positive, negative)
sentence dicts instead of the old {lang: {condition: [sentences]}} format.

Output format:
    {
        domain: {
            lang: {
                concept: [
                    {"positive": str, "negative": str, "concept_token_pos": int},
                    ...
                ]
            }
        }
    }

For backward compatibility the old StimulusGenerator is kept as an alias.
"""

import json
import random
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from collections import defaultdict

from data.concept_vocabularies import CONCEPT_DOMAINS, CONCEPT_VOCABULARIES


class ContrastivePairGenerator:
    """
    Generates (positive, negative) contrastive sentence pairs for concept
    vector extraction experiments.

    Each pair shares the same template but swaps the concept token for a
    matched control, isolating the concept signal in the activation difference.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def generate_pairs(
        self,
        domain: str,
        n_per_concept: int = 15,
        languages: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Generate contrastive pairs for a concept domain.

        Args:
            domain: Domain name (e.g. "sacred", "kinship")
            n_per_concept: Number of pairs per concept per language
            languages: Language codes to generate for (default: English only)
            output_path: Optional path to save JSON

        Returns:
            {lang: {concept: [{"positive": str, "negative": str,
                               "concept_token_pos": int}]}}
        """
        if domain not in CONCEPT_DOMAINS:
            raise ValueError(f"Unknown domain '{domain}'. Available: {list(CONCEPT_DOMAINS.keys())}")

        if languages is None:
            languages = ["eng_Latn"]

        domain_cfg = CONCEPT_DOMAINS[domain]
        concepts = domain_cfg["concepts"]
        controls = domain_cfg["controls"]
        templates = domain_cfg["templates"]

        result: Dict[str, Dict[str, List[Dict]]] = {}

        for lang in languages:
            result[lang] = {}
            for concept in concepts:
                pairs = []
                for _ in range(n_per_concept):
                    control = random.choice(controls)
                    pos_tmpl, neg_tmpl = random.choice(templates)

                    positive = pos_tmpl.format(concept=concept, control=control)
                    negative = neg_tmpl.format(concept=concept, control=control)

                    # Estimate concept token position (word index of concept)
                    pos_words = positive.split()
                    concept_pos = next(
                        (i for i, w in enumerate(pos_words) if concept.lower() in w.lower()),
                        0,
                    )

                    pairs.append({
                        "positive": positive,
                        "negative": negative,
                        "concept_token_pos": concept_pos,
                        "concept": concept,
                        "control": control,
                        "lang": lang,
                    })

                result[lang][concept] = pairs

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Contrastive pairs saved to {output_path}")

        return result

    def generate_all_domains(
        self,
        domains: Optional[List[str]] = None,
        n_per_concept: int = 15,
        languages: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """Generate pairs for multiple domains."""
        if domains is None:
            domains = list(CONCEPT_DOMAINS.keys())

        all_pairs = {}
        for domain in domains:
            pairs = self.generate_pairs(
                domain=domain,
                n_per_concept=n_per_concept,
                languages=languages,
                output_path=f"{output_dir}/{domain}_pairs.json" if output_dir else None,
            )
            all_pairs[domain] = pairs

        return all_pairs


def get_concept_words(lang: str, domain: str = "kinship") -> List[str]:
    """
    Return the vocabulary word strings for a language/domain combination.

    Use this to pass concept_words= to measure_concept_deletion for reliable
    string-based presence checking (avoids SentencePiece tokenization issues).
    Falls back to English if the language has no entry.
    """
    vocab = CONCEPT_VOCABULARIES.get(domain, {})
    return vocab.get(lang, vocab.get("eng_Latn", []))


def load_independent_sacred_tokens(
    lang: str,
    tokenizer,
    external_data_path: Optional[str] = None,
    domain: str = "sacred",
) -> List[int]:
    """
    Load concept token IDs from external sources (no circular dependency).

    Args:
        lang: Language code (e.g., "eng_Latn")
        tokenizer: NLLB tokenizer
        external_data_path: Optional path to pre-validated token JSON
        domain: Concept domain (default "sacred")

    Returns:
        List of token IDs corresponding to the concept
    """
    if external_data_path and Path(external_data_path).exists():
        with open(external_data_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)
            if lang in token_data:
                return token_data[lang]

    vocab = CONCEPT_VOCABULARIES.get(domain, {})
    words = vocab.get(lang, vocab.get("eng_Latn", []))

    token_ids = []
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        token_ids.extend(tokens)

    return list(set(token_ids))


def create_train_test_split(
    stimuli: Dict,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Create train/test splits for cross-validation.

    Works on old-format stimuli {lang: {condition: [sentences]}}.
    """
    random.seed(seed)

    train_stimuli: Dict = {}
    test_stimuli: Dict = {}

    for lang in stimuli:
        train_stimuli[lang] = {}
        test_stimuli[lang] = {}

        for condition in stimuli[lang]:
            sentences = stimuli[lang][condition]
            n_test = int(len(sentences) * test_size)
            indices = list(range(len(sentences)))
            random.shuffle(indices)

            test_indices = indices[:n_test]
            train_indices = indices[n_test:]

            train_stimuli[lang][condition] = [sentences[i] for i in train_indices]
            test_stimuli[lang][condition] = [sentences[i] for i in test_indices]

    return train_stimuli, test_stimuli


# ---------------------------------------------------------------------------
# Backward-compatible StimulusGenerator (sacred-specific three-way design)
# ---------------------------------------------------------------------------

class StimulusGenerator:
    """
    Legacy stimulus generator (three-way sacred/secular/inanimate design).
    Use ContrastivePairGenerator for new experiments.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.templates = self._load_templates()
        self.concepts = self._load_concepts()

    def _load_templates(self) -> List[Dict[str, str]]:
        return [
            {"pattern": "{subject} {verb} {object}.", "type": "declarative_svo"},
            {"pattern": "{subject} {verb} {complement}.", "type": "declarative_svc"},
            {"pattern": "{subject} {verb} {object} with {attribute}.", "type": "complex_declarative"},
            {"pattern": "{subject}, who {relative_clause}, {verb} {object}.", "type": "relative_clause"},
            {"pattern": "{object} is {verb_passive} by {subject}.", "type": "passive"},
            {"pattern": "Does {subject} {verb} {object}?", "type": "question"},
            {"pattern": "What does {subject} {verb}?", "type": "wh_question"},
            {"pattern": "{subject} is {adjective}.", "type": "copular"},
            {"pattern": "{subject} remains {adjective}.", "type": "copular_remain"},
            {"pattern": "There is {subject} that {verb} {object}.", "type": "existential"},
            {"pattern": "{subject} {past_verb} {object}.", "type": "past_tense"},
            {"pattern": "{subject} will {future_verb} {object}.", "type": "future_tense"},
        ]

    def _load_concepts(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "sacred": {
                "subjects": [
                    "God", "Allah", "the Divine", "the Creator", "the Almighty",
                    "the Lord", "Providence", "the Supreme Being", "Yahweh",
                    "the Holy One", "the Eternal", "the Heavenly Father",
                    "the Deity", "the Divine Being", "the Sacred One",
                ],
                "verbs": [
                    "creates", "guides", "watches", "protects", "blesses",
                    "loves", "commands", "reveals", "ordains", "sanctifies",
                    "redeems", "judges", "forgives", "saves",
                ],
                "adjectives": [
                    "holy", "divine", "sacred", "eternal", "omnipotent",
                    "merciful", "righteous", "almighty", "infinite", "glorious",
                ],
                "objects": [
                    "the universe", "humanity", "the world", "all beings",
                    "the faithful", "creation", "the souls", "the righteous",
                    "believers", "the earth",
                ],
            },
            "secular": {
                "subjects": [
                    "the king", "the president", "the judge", "the general",
                    "the leader", "the minister", "the senator", "the chief",
                    "the commander", "the governor", "the mayor", "the emperor",
                    "the prime minister", "the chancellor", "the ruler",
                ],
                "verbs": [
                    "leads", "directs", "controls", "manages", "governs",
                    "commands", "orders", "decides", "rules", "supervises",
                    "administers", "regulates", "presides", "oversees",
                ],
                "adjectives": [
                    "powerful", "authoritative", "commanding", "influential",
                    "respected", "dominant", "strong", "capable", "wise",
                    "experienced",
                ],
                "objects": [
                    "the nation", "the people", "the country", "the citizens",
                    "the state", "the government", "the territory", "the empire",
                    "the realm", "the population",
                ],
            },
            "inanimate": {
                "subjects": [
                    "the stone", "the rock", "the mountain", "the river",
                    "the tree", "the table", "the chair", "the building",
                    "the bridge", "the wall", "the tower", "the monument",
                    "the statue", "the boulder", "the pillar",
                ],
                "verbs": [
                    "stands", "remains", "sits", "rests", "lies",
                    "exists", "stays", "occupies", "fills", "covers",
                ],
                "adjectives": [
                    "still", "motionless", "stationary", "fixed", "immobile",
                    "static", "unmoving", "rigid", "solid", "stable",
                ],
                "objects": [
                    "the ground", "the space", "the area", "the location",
                    "the place", "the spot", "the site", "the position",
                    "the territory", "the land",
                ],
            },
        }

    def generate_diverse_stimuli(
        self,
        n_per_condition: int = 50,
        languages: List[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, List[Dict]]]:
        if languages is None:
            languages = ["eng_Latn", "spa_Latn", "arb_Arab", "zho_Hant", "qul_Latn"]

        print(f"Generating {n_per_condition} stimuli per condition for {len(languages)} languages...")
        stimuli: Dict[str, Dict[str, List[Dict]]] = {}

        for lang in languages:
            stimuli[lang] = {}
            for condition in ["sacred", "secular", "inanimate"]:
                stimuli[lang][condition] = []
                for _ in range(n_per_condition):
                    template = random.choice(self.templates)
                    sentence_data = self._fill_template(template, condition, lang)
                    if sentence_data:
                        stimuli[lang][condition].append(sentence_data)

        validation_report = self.validate_confound_control(stimuli)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(stimuli, f, ensure_ascii=False, indent=2)
            print(f"Stimuli saved to {output_path}")

        total = sum(len(v) for s in stimuli.values() for v in s.values())
        print(f"Generated {total} total sentences")
        return stimuli

    def _fill_template(self, template: Dict, condition: str, lang: str) -> Optional[Dict]:
        pattern = template["pattern"]
        concept_pool = self.concepts[condition]
        placeholders = re.findall(r'\{(\w+)\}', pattern)
        mapping = {
            "subject": "subjects", "verb": "verbs", "verbs": "verbs",
            "past_verb": "verbs", "future_verb": "verbs",
            "verb_passive": "verbs", "adjective": "adjectives",
            "object": "objects", "objects": "objects",
            "complement": "objects", "attribute": "adjectives",
            "relative_clause": "verbs",
        }
        filled = pattern
        selected = {}
        for ph in placeholders:
            concept_type = mapping.get(ph, ph + "s")
            if concept_type in concept_pool:
                concept = random.choice(concept_pool[concept_type])
                filled = filled.replace(f"{{{ph}}}", concept)
                selected[ph] = concept
            else:
                return None

        words = filled.replace(".", "").replace("?", "").replace(",", "").split()
        word_freqs = [np.log(len(w) + 1) for w in words]
        return {
            "text": filled,
            "concepts": selected,
            "template": template["type"],
            "word_freq": float(np.mean(word_freqs)) if word_freqs else 0.0,
            "length": len(words),
            "lang": lang,
        }

    def validate_confound_control(self, stimuli: Dict) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "passed": True,
            "warnings": [],
            "statistics": defaultdict(dict),
        }
        for lang in stimuli:
            lang_stats: Dict[str, List] = defaultdict(list)
            for condition in ["sacred", "secular", "inanimate"]:
                if condition in stimuli[lang]:
                    sents = stimuli[lang][condition]
                    lang_stats[f"{condition}_length"].extend([s["length"] for s in sents])
                    lang_stats[f"{condition}_freq"].extend([s["word_freq"] for s in sents])

            length_means = {
                c: float(np.mean(lang_stats[f"{c}_length"])) if lang_stats[f"{c}_length"] else 0.0
                for c in ["sacred", "secular", "inanimate"]
            }
            max_diff = max(length_means.values()) - min(length_means.values())
            if max_diff > 2.0:
                report["warnings"].append(
                    f"{lang}: Length mismatch (max diff: {max_diff:.2f} tokens)."
                )
                report["passed"] = False

            report["statistics"][lang] = {
                "length_means": length_means,
                "freq_means": {
                    c: float(np.mean(lang_stats[f"{c}_freq"])) if lang_stats[f"{c}_freq"] else 0.0
                    for c in ["sacred", "secular", "inanimate"]
                },
            }
        return report
