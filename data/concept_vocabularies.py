"""
Concept vocabularies for SACRED experiments.

CONCEPT_VOCABULARIES: Token-level word lists per concept domain per language.
  Used by load_independent_sacred_tokens / measure_concept_deletion to map
  concepts to token IDs.

CONCEPT_DOMAINS: Domain-level contrastive pair templates.
  Used by ContrastivePairGenerator to build (positive, negative) sentence pairs.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Token-level vocabularies  {domain: {lang_code: [word_strings]}}
# ---------------------------------------------------------------------------

CONCEPT_VOCABULARIES: Dict[str, Dict[str, List[str]]] = {
    "sacred": {
        "eng_Latn": [
            "God", "Lord", "Creator", "Divine", "Almighty", "Allah",
            "Holy", "Sacred", "Heaven", "Deity", "Providence", "Eternal",
            "Christ", "Jesus", "Spirit", "Faith", "Blessed", "Righteous",
        ],
        "spa_Latn": [
            "Dios", "Señor", "Creador", "Divino", "Todopoderoso", "Alá",
            "Santo", "Sagrado", "Cielo", "Deidad", "Providencia", "Eterno",
            "Cristo", "Jesús", "Espíritu", "Fe", "Bendito", "Justo",
        ],
        "arb_Arab": [
            "الله", "الرب", "الخالق", "المقدس", "القدير", "السماء",
            "الإله", "المبارك", "الروح", "الإيمان", "العدل", "الأبدي",
        ],
        "zho_Hant": [
            "神", "上帝", "主", "天父", "創造者", "全能者",
            "神聖", "天堂", "聖靈", "信仰", "永恆", "正義",
        ],
        "qul_Latn": [
            "Dios", "Yaya", "Pachakamaq", "Wiraqocha", "Tayta",
            "Santo", "Sagrado", "Hanaq pacha", "Espíritu",
        ],
        "yor_Latn": [
            "Ọlọrun", "Olodumare", "Eledumare", "Oba", "Mimọ",
            "Orun", "Ẹmi", "Igbagbọ", "Olugbala",
        ],
        "hau_Latn": [
            "Allah", "Ubangiji", "Mahalicci", "Madaukaki", "Tsarki",
            "Sama", "Ruhu", "Imani", "Mai ceto",
        ],
    },
    "kinship": {
        "eng_Latn": [
            "mother", "father", "family", "child", "son", "daughter",
            "parent", "sibling", "brother", "sister", "grandmother",
            "grandfather", "ancestor", "relative", "kin",
        ],
        "spa_Latn": [
            "madre", "padre", "familia", "niño", "hijo", "hija",
            "progenitor", "hermano", "hermana", "abuela", "abuelo",
            "antepasado", "pariente",
        ],
        "arb_Arab": [
            "أم", "أب", "عائلة", "طفل", "ابن", "ابنة",
            "والد", "أخ", "أخت", "جدة", "جد", "قريب",
        ],
        "zho_Hant": [
            "母親", "父親", "家庭", "孩子", "兒子", "女兒",
            "父母", "兄弟", "姐妹", "祖母", "祖父", "親戚",
        ],
        "qul_Latn": [
            "mama", "tayta", "ayllu", "wawa", "churi", "ñust'a",
        ],
    },
}


# ---------------------------------------------------------------------------
# Domain definitions  {domain: {concepts, controls, templates}}
# Used by ContrastivePairGenerator to build (positive, negative) pairs.
# ---------------------------------------------------------------------------

CONCEPT_DOMAINS: Dict[str, Dict] = {
    "sacred": {
        "concepts": [
            "God", "Allah", "the Divine", "the Creator", "the Almighty",
            "the Lord", "Providence", "the Supreme Being", "Yahweh",
            "the Holy One", "the Eternal", "the Heavenly Father",
        ],
        "controls": [
            "the king", "the president", "the judge", "the general",
            "the leader", "the minister", "the senator", "the chief",
            "the commander", "the governor",
        ],
        "templates": [
            ("{concept} watches over humanity.", "{control} watches over humanity."),
            ("{concept} guides the faithful.", "{control} guides the citizens."),
            ("People pray to {concept}.", "People appeal to {control}."),
            ("The word of {concept} is eternal.", "The decree of {control} is binding."),
            ("{concept} created the universe.", "{control} built the nation."),
            ("Believers trust in {concept}.", "Citizens trust in {control}."),
            ("The glory of {concept} is infinite.", "The power of {control} is vast."),
            ("{concept} forgives all sins.", "{control} pardons all crimes."),
            ("Prophets speak for {concept}.", "Ministers speak for {control}."),
            ("Temples are dedicated to {concept}.", "Monuments are dedicated to {control}."),
        ],
    },
    "kinship": {
        "concepts": ["mother", "father", "family", "child", "grandmother", "brother"],
        "controls": ["teacher", "leader", "group", "student", "colleague", "neighbor"],
        "templates": [
            ("My {concept} taught me everything.", "My {control} taught me everything."),
            ("The {concept} is important to me.", "The {control} is important to me."),
            ("I love my {concept} deeply.", "I respect my {control} deeply."),
            ("My {concept} raised me with care.", "My {control} mentored me with care."),
            ("We gathered around my {concept}.", "We gathered around my {control}."),
            ("The {concept} gave me advice.", "The {control} gave me advice."),
            ("I miss my {concept} every day.", "I think of my {control} every day."),
            ("My {concept} sacrificed a lot for me.", "My {control} worked hard for me."),
            ("The {concept} holds the family together.", "The {control} holds the team together."),
            ("I owe everything to my {concept}.", "I am grateful to my {control}."),
            ("My {concept} was always there for me.", "My {control} was always available."),
            ("The {concept} passed down traditions.", "The {control} passed down knowledge."),
            ("We celebrated with my {concept}.", "We celebrated with my {control}."),
            ("I spoke with my {concept} last night.", "I spoke with my {control} last night."),
            ("My {concept} knows me better than anyone.", "My {control} understands me well."),
        ],
    },
}
