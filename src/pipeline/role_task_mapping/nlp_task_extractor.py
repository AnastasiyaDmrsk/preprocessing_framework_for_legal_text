import re
from typing import Dict, List, Set, Tuple

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token

from src.pipeline.organigram.const import DEFAULT_SPACY_MODEL
from src.pipeline.organigram.utils import normalize_name, is_actor
from .const import (
    DEONTIC_PATTERNS,
    DEONTIC_MODAL_LEMMAS,
    DELEGATED_OBLIGATION_VERBS,
    REQUEST_VERBS,
    CONDITION_MARKERS, _LABEL_NOISE_RE, _SKIP_RE, _VP_EXCLUDE_DEPS, _PRONOUN_TAGS,
)
from .models import TaskCandidate


class NLPTaskCandidateExtractor:

    def __init__(self, spacy_model: str = DEFAULT_SPACY_MODEL):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.nlp = spacy.load("en_core_web_lg")
        self._build_deontic_matcher()

    def extract_candidates(self, text: str) -> List[TaskCandidate]:
        doc = self.nlp(text)
        article, para_index = self._build_source_index(text)
        candidates: List[TaskCandidate] = []

        for sent in doc.sents:
            if _SKIP_RE.match(sent.text.strip()):
                continue
            paragraph = self._get_paragraph_for_offset(sent.start_char, para_index)
            for deontic in self._detect_deontics(doc, sent):
                candidates.extend(
                    self._process_deontic(sent, deontic, article, paragraph)
                )

        return candidates

    def _build_source_index(
            self, text: str
    ) -> Tuple[str, List[Tuple[int, str]]]:
        art_match = re.search(r'\bArticle\s+(\d+[a-z]?)\b', text, re.IGNORECASE)
        article = art_match.group(1) if art_match else "UNKNOWN"

        para_re = re.compile(r'(?m)^[ \t]*(\d{1,2}(?:\([a-z]\))?)\s*[.\s]')
        index = [(m.start(), m.group(1)) for m in para_re.finditer(text)]
        return article, sorted(index, key=lambda x: x[0])

    def _get_paragraph_for_offset(
            self,
            offset: int,
            para_index: List[Tuple[int, str]],
    ) -> str:
        current = "UNKNOWN"
        for pos, pid in para_index:
            if pos <= offset:
                current = pid
            else:
                break
        return current

    def _build_deontic_matcher(self) -> None:
        self._matcher = Matcher(self.nlp.vocab)
        self._matcher.add("REQUIRED_TO", [
            [{"LEMMA": "be"}, {"LOWER": "required"}, {"LOWER": "to"}],
        ])

    def _detect_deontics(self, doc: Doc, sent: Span) -> List[Dict]:
        results: List[Dict] = []
        seen_heads: Set[int] = set()

        for token in sent:
            if token.dep_ not in {"aux", "auxpass"}:
                continue
            if token.lemma_ not in DEONTIC_MODAL_LEMMAS:
                continue
            head = token.head
            if head.i in seen_heads:
                continue
            seen_heads.add(head.i)
            negated = self._is_negated(head)
            deontic_type, deontic_modality = self._resolve_deontic_type(
                token.lemma_, negated
            )
            results.append({
                "head_verb": head,
                "deontic_type": deontic_type,
                "deontic_modality": deontic_modality,
            })

        for _, start, end in self._matcher(doc):
            if doc[start].sent.start != sent.start:
                continue
            if end < len(doc) and doc[end].pos_ == "VERB":
                head = doc[end]
                if head.i not in seen_heads:
                    seen_heads.add(head.i)
                    results.append({
                        "head_verb": head,
                        "deontic_type": "obligation",
                        "deontic_modality": "is required to",
                    })

        return results

    def _is_negated(self, verb: Token) -> bool:
        return any(
            c.dep_ == "neg" and c.lemma_ in {"not", "n't"}
            for c in verb.children
        )

    def _resolve_deontic_type(
            self, lemma: str, negated: bool
    ) -> Tuple[str, str]:
        key = f"{lemma} not" if negated else lemma
        for pattern, deontic_type in DEONTIC_PATTERNS:
            if pattern == key:
                return deontic_type, pattern
        return "obligation", lemma

    def _process_deontic(
            self,
            sent: Span,
            deontic: Dict,
            article: str,
            paragraph: str,
    ) -> List[TaskCandidate]:
        head_verb = deontic["head_verb"]
        subject_names = self._extract_subjects(head_verb)
        conditions = self._extract_conditions(head_verb)
        head_lemma = head_verb.lemma_.lower()
        sentence = sent.text.strip()

        if head_lemma in DELEGATED_OBLIGATION_VERBS:
            return self._handle_delegated_obligation(
                head_verb, subject_names, deontic,
                article, paragraph, sentence, conditions,
            )

        if head_lemma in REQUEST_VERBS:
            return self._handle_request_pattern(
                head_verb, subject_names, deontic,
                article, paragraph, sentence, conditions,
            )

        return [
            self._make_candidate(
                label, deontic["deontic_type"], deontic["deontic_modality"],
                conditions, article, paragraph, sentence,
            )
            for label in self._split_compound_verbs(head_verb)
        ]

    def _split_compound_verbs(self, head: Token) -> List[str]:
        labels = [self._extract_verb_phrase(head)]
        for child in head.children:
            if child.dep_ == "conj" and child.pos_ == "VERB":
                label = self._extract_verb_phrase(child)
                if label:
                    labels.append(label)
        return [l for l in labels if l]

    def _handle_delegated_obligation(
            self,
            head: Token,
            subjects: List[str],
            deontic: Dict,
            article: str,
            paragraph: str,
            sentence: str,
            conditions: List[str],
    ) -> List[TaskCandidate]:
        candidates = [
            self._make_candidate(
                self._extract_verb_phrase(head),
                deontic["deontic_type"], deontic["deontic_modality"],
                conditions, article, paragraph, sentence,
            )
        ]
        for child in head.children:
            if child.dep_ in {"ccomp", "xcomp"} and child.pos_ == "VERB":
                candidates.append(self._make_candidate(
                    self._extract_verb_phrase(child),
                    "obligation", "shall",
                    conditions, article, paragraph, sentence,
                ))
        return candidates

    def _handle_request_pattern(
            self,
            head: Token,
            subjects: List[str],
            deontic: Dict,
            article: str,
            paragraph: str,
            sentence: str,
            conditions: List[str],
    ) -> List[TaskCandidate]:
        candidates = [
            self._make_candidate(
                self._extract_verb_phrase(head),
                deontic["deontic_type"], deontic["deontic_modality"],
                conditions, article, paragraph, sentence,
            )
        ]
        for child in head.children:
            if child.dep_ in {"xcomp", "ccomp"} and child.pos_ == "VERB":
                candidates.append(self._make_candidate(
                    self._extract_verb_phrase(child),
                    "obligation", "shall",
                    conditions + ["if requested"], article, paragraph, sentence,
                ))
        return candidates

    def _extract_verb_phrase(self, verb: Token) -> str:
        def collect(token: Token) -> List[Token]:
            out = [token]
            for child in token.children:
                if child.dep_ not in _VP_EXCLUDE_DEPS:
                    out.extend(collect(child))
            return out

        tokens = sorted(collect(verb), key=lambda t: t.i)
        raw = " ".join(t.text for t in tokens).strip()
        return self._clean_label(raw)

    def _clean_label(self, label: str) -> str:
        label = _LABEL_NOISE_RE.sub("", label).strip()
        return re.sub(r"\s+", " ", label)

    def _extract_conditions(self, verb: Token) -> List[str]:
        conditions: List[str] = []
        for child in verb.children:
            if child.dep_ != "advcl":
                continue
            marker = next(
                (c.lemma_.lower() for c in child.children if c.dep_ == "mark"),
                None,
            )
            if marker and marker in CONDITION_MARKERS:
                clause = " ".join(
                    t.text for t in sorted(child.subtree, key=lambda t: t.i)
                ).strip()
                if clause:
                    conditions.append(clause)
        return conditions

    def _extract_subjects(self, verb: Token) -> List[str]:
        subjects: List[str] = []

        for child in verb.children:
            if child.dep_ == "agent":
                for gc in child.children:
                    if gc.dep_ == "pobj" and gc.tag_ not in _PRONOUN_TAGS:
                        chunk = next(
                            (ch.text for ch in verb.doc.noun_chunks if ch.root == gc),
                            gc.text,
                        )
                        name = normalize_name(chunk)
                        if name and is_actor(name):
                            subjects.append(name)
        if subjects:
            return subjects

        for child in verb.children:
            if child.dep_ not in {"nsubj", "nsubjpass"}:
                continue
            if child.tag_ in _PRONOUN_TAGS:
                continue
            chunk = next(
                (ch.text for ch in verb.doc.noun_chunks if ch.root == child),
                child.text,
            )
            name = normalize_name(chunk)
            if name and is_actor(name):
                subjects.append(name)

        if not subjects and verb.dep_ in {"ccomp", "xcomp", "conj"}:
            parent = verb.head
            if parent != verb:
                subjects = self._extract_subjects(parent)

        return subjects

    def _make_candidate(
            self,
            label: str,
            deontic_type: str,
            deontic_modality: str,
            conditions: List[str],
            article: str,
            paragraph: str,
            sentence: str,
    ) -> TaskCandidate:
        return TaskCandidate(
            label=label,
            deontic_type=deontic_type,
            deontic_modality=deontic_modality,
            conditions=conditions,
            source_article=article,
            source_paragraph=paragraph,
            sentence=sentence,
        )
