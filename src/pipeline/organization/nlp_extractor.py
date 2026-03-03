from collections import Counter
from typing import List, Tuple, Set

import spacy
from spacy.matcher import PhraseMatcher

from .const import (
    ACTOR_DEPS,
    PROCESS_ACTION_VERBS,
    GLINER_LABELS,
    DEFAULT_SPACY_MODEL,
    DEFAULT_GLINER_MODEL,
    DEFAULT_GLINER_THRESHOLD,
    EU_INSTITUTIONS,
)
from .utils import normalize_name, is_actor, infer_type


class NLPActorCandidateExtractor:
    """
    Four-layer NLP pipeline for actor candidate extraction.
    Returns List[Tuple[str, str]] to allow the same name as both UNIT and ROLE.

    Layer 1  – PhraseMatcher (NAL bodies, dynamic): UNIT
    Layer 2  – spaCy NER: ORG / GPE / NORP: UNIT
    Layer 3  – Dep. parsing + action-verb heuristic: UNIT / ROLE
    Layer 3b – Frequency-weighted NP subjects (min_freq=2): UNIT / ROLE
    Layer 4  – GLiNER zero-shot (optional): overwrite
    Post     – Plural normalization + specificity filter
    """

    def __init__(
        self,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        use_gliner: bool = False,
        gliner_model: str = DEFAULT_GLINER_MODEL,
        gliner_threshold: float = DEFAULT_GLINER_THRESHOLD,
        freq_min_count: int = 2,
    ):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.nlp = spacy.load("en_core_web_lg")

        self.freq_min_count = freq_min_count
        self._build_unit_phrase_matcher(EU_INSTITUTIONS)

        self.use_gliner = use_gliner
        self.gliner_threshold = gliner_threshold
        if use_gliner:
            from gliner import GLiNER
            self.gliner = GLiNER.from_pretrained(gliner_model)

    def _build_unit_phrase_matcher(self, bodies: list[str]) -> None:
        self.unit_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.unit_matcher.add(
            "EU_NAL",
            [self.nlp.make_doc(t.lower()) for t in bodies if t.strip()],
        )

    def extract_candidates(self, text: str) -> List[Tuple[str, str]]:
        """
        Returns [(name, "UNIT"|"ROLE"), ...].
        Same name can appear twice with different types (dual actors).
        Plural forms are collapsed to singular before returning.
        """
        doc = self.nlp(text)
        units: Set[str] = set()
        roles: Set[str] = set()

        self._apply_spacy_ner(doc, units)
        self._apply_dep_parsing(doc, units, roles)
        self._apply_unit_phrase_matcher(doc, units)
        self._apply_frequent_subject_nps(doc, units, roles)

        if self.use_gliner:
            self._apply_gliner(text, units, roles)

        # Post-processing: remove noise, normalize plurals
        units = self._apply_specificity_filter(units)
        roles = self._apply_specificity_filter(roles)
        units = self._normalize_plurals(units)
        roles = self._normalize_plurals(roles)

        result: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for name in sorted(units):
            if (name, "UNIT") not in seen:
                result.append((name, "UNIT"))
                seen.add((name, "UNIT"))
        for name in sorted(roles):
            if (name, "ROLE") not in seen:
                result.append((name, "ROLE"))
                seen.add((name, "ROLE"))
        return result

    def _apply_specificity_filter(self, s: Set[str]) -> Set[str]:
        """
        Remove single-word lowercase common nouns — they are too generic to be actors
        (e.g. standalone "entities", "guidance", "incident").
        Single-word proper nouns (title case) and acronyms (all-caps) are kept.
        Multi-word phrases always pass.
        """
        result = set()
        for name in s:
            tokens = name.split()
            if len(tokens) == 1:
                # Keep if proper noun (starts uppercase) or acronym (all caps)
                if name[0].isupper() or name.isupper():
                    result.add(name)
                # else: single lowercase generic word
            else:
                result.add(name)
        return result

    def _to_singular_via_lemma(self, name: str) -> str:
        """
        Singularize a (possibly multi-word) actor name by lemmatizing
        the final token via spaCy. Falls back to a one-liner rule only
        for all-caps acronyms which the lemmatizer leaves unchanged.
        """
        tokens = name.split()
        if not tokens:
            return name

        last = tokens[-1]
        last_doc = self.nlp(last)
        lemma = last_doc[0].lemma_ if last_doc else last
        if lemma == last and last.isupper() and last.endswith("S") and len(last) > 2:
            lemma = last[:-1]

        return " ".join(tokens[:-1] + [lemma])

    def _normalize_plurals(self, s: Set[str]) -> Set[str]:
        """
        Collapse plural/singular pairs to the singular form using lemmatization.
        If both "CSIRT" and "CSIRTs" exist, keeps "CSIRT".
        If both "Member State" and "Member States" exist, keeps "Member State".
        """
        names_lower: Set[str] = {n.lower() for n in s}
        to_remove: Set[str] = set()

        for name in s:
            singular = self._to_singular_via_lemma(name)
            if singular.lower() != name.lower() and singular.lower() in names_lower:
                to_remove.add(name)

        return s - to_remove

    # Layers

    def _apply_spacy_ner(self, doc, units: Set[str]) -> None:
        for ent in doc.ents:
            if ent.label_ in {"ORG", "GPE", "NORP"}:
                name = normalize_name(ent.text)
                if name and is_actor(name):
                    units.add(name)

    def _apply_dep_parsing(self, doc, units: Set[str], roles: Set[str]) -> None:
        for token in doc:
            if token.dep_ not in ACTOR_DEPS or token.pos_ not in {"NOUN", "PROPN"}:
                continue
            chunk_text = next(
                (ch.text for ch in doc.noun_chunks if ch.root == token),
                token.text,
            )
            name = normalize_name(chunk_text)
            if not name or not is_actor(name):
                continue

            head_lemma = token.head.lemma_.lower()
            is_process_actor = any(
                head_lemma.startswith(v) for v in PROCESS_ACTION_VERBS
            )
            if is_process_actor:
                roles.add(name)
            else:
                t = infer_type(name)
                (roles if t == "ROLE" else units).add(name)

    def _apply_unit_phrase_matcher(self, doc, units: Set[str]) -> None:
        for _, start, end in self.unit_matcher(doc):
            name = normalize_name(doc[start:end].text)
            if name:
                units.add(name)

    def _apply_frequent_subject_nps(
        self,
        doc,
        units: Set[str],
        roles: Set[str],
        min_freq: int = None,
    ) -> None:
        if min_freq is None:
            min_freq = self.freq_min_count

        np_total: Counter = Counter()
        np_as_subject: Counter = Counter()
        np_original: dict = {}

        for chunk in doc.noun_chunks:
            name = normalize_name(chunk.text)
            if name and is_actor(name) and len(name.split()) <= 6:
                key = name.lower()
                np_total[key] += 1
                if key not in np_original or name[0].isupper():
                    np_original[key] = name

        for token in doc:
            if token.dep_ in ACTOR_DEPS and token.pos_ in {"NOUN", "PROPN"}:
                chunk_text = next(
                    (ch.text for ch in doc.noun_chunks if ch.root == token),
                    token.text,
                )
                name = normalize_name(chunk_text)
                if name and is_actor(name):
                    np_as_subject[name.lower()] += 1

        for name_lower, subj_count in np_as_subject.items():
            if np_total.get(name_lower, 0) >= min_freq:
                original = np_original.get(name_lower, name_lower.title())
                t = infer_type(original)
                (roles if t == "ROLE" else units).add(original)

    def _apply_gliner(self, text: str, units: Set[str], roles: Set[str]) -> None:
        for ent in self.gliner.predict_entities(
            text, GLINER_LABELS, threshold=self.gliner_threshold
        ):
            name = normalize_name(ent["text"])
            if name and is_actor(name):
                (roles if "role" in ent["label"] else units).add(name)
