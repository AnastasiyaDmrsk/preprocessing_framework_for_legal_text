import re
from typing import Tuple, List, Optional, Set, Callable

import spacy
from spacy.matcher import Matcher, PhraseMatcher

from .text_utils import build_eu_ref_matcher
from .const import (
    ARTICLE_STANDARD_RE, ARTICLE_ALT_RE, SECTION_NUMBERING_RE,
    ARTICLE_MULTI_REF_RE, ARTICLE_SINGLE_REF_RE, INTERNAL_REF_RE,
    DEONTIC_MODALS, OBLIGATION_RE,
    SUBORDINATE_LABELS, CONDITIONAL_SUBORDINATORS,
    REFERENCE_MARKERS, LEGAL_ENTITY_KEYWORDS,
    PARALLEL_GATEWAY_START, PARALLEL_GATEWAY_END, GATEWAY_MARKER_PREFIX,
    _BENEPAR_MAX_TOKENS,
)
from .nlp_utils import (
    plan_filler_removal,
    extract_explicit_actor, has_actor_and_activity,
    plan_pronoun_resolution, plan_passive_resolution,
    highlight_or_between_verbs, highlight_and_between_modal_verbs, remove_external_reference_phrases,
)
from .text_utils import (
    normalize_whitespace, apply_iaw, apply_static_fillers, normalize_if,
    TokenTransformPlan,
)


class RegulatoryTextPreprocessor:
    """
    Preprocessing pipeline for EU regulatory text targeting BPMN extraction.
    Install
    -------
        pip install spacy benepar
        python -m spacy download en_core_web_md
        python -c "import benepar; benepar.download('benepar_en3')"
    """

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError("No spaCy model found.")

        self._benepar_available = False
        if "benepar" not in self.nlp.pipe_names:
            try:
                import benepar
                self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})
                self._benepar_available = True
            except Exception:
                pass
        else:
            self._benepar_available = True

        self._eu_ref_matcher       = None
        self._internal_ref_matcher = None
        self._ref_marker_matcher   = None
        self._init_matchers()

    def _init_matchers(self) -> None:
        self._eu_ref_matcher = build_eu_ref_matcher(self.nlp)
        internal = Matcher(self.nlp.vocab)
        for label in ["Article", "Paragraph", "Section", "Art."]:
            internal.add(f"INT_REF_{label.upper().rstrip('.')}", [[
                {"LOWER": label.lower().rstrip(".")},
                {"LIKE_NUM": True},
            ]])
        self._internal_ref_matcher = internal
        phrase = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        phrase.add("REF_MARKER", [self.nlp.make_doc(m) for m in REFERENCE_MARKERS])
        self._ref_marker_matcher = phrase


    def _parse_plain(self, text: str):
        """Parse with spaCy dep/POS only — BenePar disabled."""
        if self._benepar_available:
            with self.nlp.select_pipes(disable=["benepar"]):
                return self.nlp(text)
        return self.nlp(text)


    def _extract_references_from_doc(
        self, doc, location: str
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        refs: List[str] = []
        ext_spans: List[Tuple[int, int]] = []
        if not location:
            return refs, ext_spans
        for _match_id, start, end in self._eu_ref_matcher(doc):
            refs.append(f"{location},{doc[start:end].text.strip()}")
            ext_spans.append((start, end))
        marker_end_positions: Set[int] = {
            m_end for _, _m_start, m_end in self._ref_marker_matcher(doc)
        }
        for _, a_start, a_end in self._internal_ref_matcher(doc):
            if a_start in marker_end_positions or (a_start - 1) in marker_end_positions:
                refs.append(f"{location},{doc[a_start:a_end].text.strip()}")
        text = doc.text
        for m in ARTICLE_MULTI_REF_RE.finditer(text):
            for part in re.split(r",|and", m.group(1)):
                num = part.strip()
                if num:
                    refs.append(f"{location},Article {num}")
        for m in ARTICLE_SINGLE_REF_RE.finditer(text):
            refs.append(f"{location},Article {m.group(1)}")
        return refs, ext_spans

    def _plan_subordinate_removal(self, doc, plan: TokenTransformPlan) -> None:
        """
        Write subordinate-span removals to the plan using BenePar constituency trees.
        Silent no-op if BenePar is unavailable or sentence exceeds token limit;
        the regex fallback runs pre-parse in _process_sentence.
        """
        if not self._benepar_available or len(doc) > _BENEPAR_MAX_TOKENS:
            return
        try:
            for sent_span in doc.sents:
                self._collect_subordinate_spans(sent_span, plan)
        except Exception:
            pass

    def _collect_subordinate_spans(self, sent_span, plan: TokenTransformPlan) -> None:
        """
        Recursive BenePar tree walk. Marks SBAR/WHNP/WHADVP/WHPP for removal
        unless a guard fires:

        G1  Deontic modal present         → obligation or condition carrier — KEEP
        G2  Conditional first token       → gateway condition — KEEP
        G3  Internal reference present    → paragraph/article X — KEEP
        G5  Process actor + activity      → legal entity as nsubj — KEEP

        G4 (external citation) is NOT a guard: citations are already extracted
        into the reference table by _extract_references_from_doc before this
        runs, so the clause can be removed without losing the reference.
        """
        def _walk(constituent) -> None:
            if not any(lbl in SUBORDINATE_LABELS for lbl in constituent._.labels):
                for child in constituent._.children:
                    _walk(child)
                return
            if any(t.lemma_ in DEONTIC_MODALS for t in constituent):               return  # G1
            if constituent and constituent[0].text.lower() in CONDITIONAL_SUBORDINATORS:
                                                                                    return  # G2
            if INTERNAL_REF_RE.search(constituent.text):                            return  # G3
            if has_actor_and_activity(constituent):                                 return  # G5
            plan.remove_span(constituent.start, constituent.end)

        _walk(sent_span)

    def _remove_subordinate_clauses_regex(self, sentence: str) -> str:
        """
        Regex fallback (BenePar unavailable). Same guards as _collect_subordinate_spans.
        G4 is not a guard here either — external refs are extracted before this runs.
        """
        def _check(match: re.Match) -> str:
            clause = match.group(0)
            if OBLIGATION_RE.search(clause):                                    return clause  # G1
            first = clause.strip().split()[0].lower() if clause.strip() else ""
            if first in CONDITIONAL_SUBORDINATORS:                              return clause  # G2
            if INTERNAL_REF_RE.search(clause):                                  return clause  # G3
            clause_doc = self._parse_plain(clause)
            for token in clause_doc:
                if token.dep_ in {"nsubj", "nsubjpass"}:
                    if any(kw in token.text.lower() for kw in LEGAL_ENTITY_KEYWORDS):
                        return clause  # G5
                    for chunk in clause_doc.noun_chunks:
                        if chunk.root.i == token.i and any(
                                kw in chunk.text.lower() for kw in LEGAL_ENTITY_KEYWORDS):
                            return clause  # G5
            return ""

        return normalize_whitespace(re.sub(
            r"\s+(?:that|which|who)\s+[^,.;]+", _check, sentence, flags=re.IGNORECASE
        ))

    def preprocess(self, input_text: str) -> Tuple[str, List[str]]:
        input_text = "\n".join(line.strip() for line in input_text.split("\n"))
        dispatch: dict[str, Callable[[str], Tuple[str, List[str]]]] = {
            "standard_article":  self._preprocess_standard_articles,
            "alt_article":       self._preprocess_alt_articles,
            "section_numbering": self._preprocess_section_numbering,
        }
        structure = self._detect_document_structure(input_text)
        handler: Callable[[str], Tuple[str, List[str]]] = dispatch.get(
            structure, self._preprocess_raw_text
        )
        return handler(input_text)


    def _detect_document_structure(self, text: str) -> str:
        if ARTICLE_STANDARD_RE.search(text):  return "standard_article"
        if ARTICLE_ALT_RE.search(text):       return "alt_article"
        if SECTION_NUMBERING_RE.search(text): return "section_numbering"
        return "raw_text"


    def _collect_existing_articles(self, text: str, pattern) -> Set[str]:
        return {m.group(2) for m in pattern.finditer(text) if len(m.groups()) >= 2}

    @staticmethod
    def _split_by_pattern(text: str, pattern, fallback_id: str):
        matches = list(pattern.finditer(text))
        if not matches:
            return [(fallback_id, "", text)]
        return [
            (m.group(1), m.group(3).strip() if len(m.groups()) >= 3 else "",
             text[m.end():(matches[i+1].start() if i+1 < len(matches) else len(text))].strip())
            for i, m in enumerate(matches)
        ]

    def _split_standard_articles(self, text: str):
        return self._split_by_pattern(text, ARTICLE_STANDARD_RE, "Article 0")

    def _split_alt_articles(self, text: str):
        return self._split_by_pattern(text, ARTICLE_ALT_RE, "Art. 0")

    def _split_section_numbering(self, text: str):
        matches = list(SECTION_NUMBERING_RE.finditer(text))
        if not matches:
            return [("0", "", text)]
        return [
            (m.group(1), m.group(2).strip(),
             text[m.end():(matches[i+1].start() if i+1 < len(matches) else len(text))].strip())
            for i, m in enumerate(matches)
        ]


    def _preprocess_standard_articles(self, text: str) -> Tuple[str, List[str]]:
        existing = self._collect_existing_articles(text, ARTICLE_STANDARD_RE)
        return self._process_articles(self._split_standard_articles(text), existing)

    def _preprocess_alt_articles(self, text: str) -> Tuple[str, List[str]]:
        existing = self._collect_existing_articles(text, ARTICLE_ALT_RE)
        return self._process_articles(self._split_alt_articles(text), existing)

    def _preprocess_section_numbering(self, text: str) -> Tuple[str, List[str]]:
        sections = self._split_section_numbering(text)
        clauses, refs, last_actor = [], set(), None
        for sec_id, heading, sec_text in sections:
            if heading:
                clauses.append(f"{sec_id} {heading}".strip())
            for sent in self._split_sentences(sec_text):
                result = self._process_sentence(sent.strip(), sec_id, set(), last_actor, refs)
                if result:
                    proc, actor = result
                    clauses.append(f"{sec_id} {proc}")
                    if actor:
                        last_actor = actor
        return "\n".join(clauses), sorted(refs)

    def _preprocess_raw_text(self, text: str) -> Tuple[str, List[str]]:
        clauses, refs, last_actor = [], set(), None
        for sent in self._split_sentences(text):
            result = self._process_sentence(sent.strip(), "", set(), last_actor, refs)
            if result:
                proc, actor = result
                clauses.append(proc)
                if actor:
                    last_actor = actor
        return "\n".join(clauses), sorted(refs)

    def _process_articles(self, articles, existing_articles: Set[str]) -> Tuple[str, List[str]]:
        clauses, refs, last_actor = [], set(), None
        for art_idx, (art_id, heading, art_text) in enumerate(articles):
            clauses.append(f"{art_id}: {heading}")
            paragraphs = self._parse_paragraphs_with_lists(art_text, art_id)
            current_para = None
            for para_loc, para_content, _is_list_item, _is_last, list_letter in paragraphs:
                if para_content.startswith(GATEWAY_MARKER_PREFIX):
                    clauses.append(para_content)
                    continue
                para_num = para_loc.split(".")[-1].split("(")[0].strip()
                result = self._process_sentence(
                    para_content, para_loc, existing_articles, last_actor, refs)
                if result:
                    sent, actor = result
                    if list_letter:
                        clauses.append(f"{list_letter} {sent}")
                    elif para_num != current_para:
                        clauses.append(f"{para_num} {sent}")
                        current_para = para_num
                    else:
                        clauses.append(sent)
                    if actor:
                        last_actor = actor
            if art_idx < len(articles) - 1:
                clauses.append("---")
        return "\n".join(clauses), sorted(refs)

    def _parse_paragraphs_with_lists(self, art_text: str, art_id: str) -> List:
        result = []
        para_parts = re.split(r"(?m)^(\d+)\.\s*", art_text)
        for i in range(1, len(para_parts), 2):
            if i + 1 >= len(para_parts):
                continue
            para_num  = para_parts[i]
            para_text = para_parts[i + 1].strip()
            para_loc  = f"{art_id}.{para_num}"
            list_paren    = list(re.finditer(
                r"\(([a-z])\)\s*\n(.+?)(?=\n\s*\([a-z]\)|\Z)", para_text, re.DOTALL))
            list_no_paren = list(re.finditer(
                r"^([a-z])\)\s*\n?(.+?)(?=^[a-z]\)|\Z)", para_text,
                re.MULTILINE | re.DOTALL))
            list_items = list_paren if list_paren else list_no_paren
            if list_items and len(list_items) > 1:
                leading = para_text[:list_items[0].start()].strip()
                leading = re.sub(r":\s*$", "", leading)
                leading = re.sub(
                    r"\s+(?:the\s+following|as\s+follows)\s*$", "", leading,
                    flags=re.IGNORECASE).strip()
                result.append((para_loc, PARALLEL_GATEWAY_START, False, False, None))
                for idx, match in enumerate(list_items):
                    letter    = (f"{para_num}({match.group(1)})" if list_paren
                                 else f"{para_num}{match.group(1)})")
                    item_text = match.group(2).strip().rstrip(";").rstrip(".").strip()
                    result.append((
                        f"{para_loc} ({letter})",
                        f"{leading} {item_text}",
                        True, idx == len(list_items) - 1, letter,
                    ))
                result.append((para_loc, PARALLEL_GATEWAY_END, False, False, None))
            else:
                for sent in self._split_sentences(para_text):
                    if sent.strip():
                        result.append((para_loc, sent.strip(), False, False, None))
        return result


    def _process_sentence(
        self,
        sentence:          str,
        location:          str,
        existing_articles: Set[str],
        last_actor:        Optional[str],
        references_set:    Set[str],
    ) -> Optional[Tuple[str, Optional[str]]]:

        sentence, _ = apply_static_fillers(sentence)
        sentence    = normalize_whitespace(sentence)

        if self._benepar_available:
            try:
                doc = self.nlp(sentence)
            except Exception:
                sentence = self._remove_subordinate_clauses_regex(sentence)
                sentence = normalize_whitespace(sentence)
                doc = self._parse_plain(sentence)
        else:
            sentence = self._remove_subordinate_clauses_regex(sentence)
            sentence = normalize_whitespace(sentence)
            doc = self._parse_plain(sentence)
        plan = TokenTransformPlan(doc)

        extracted, ext_spans = self._extract_references_from_doc(doc, location)
        references_set.update(extracted)
        for start, end in ext_spans:
            plan.remove_span(start, end)

        plan_filler_removal(doc, plan)
        self._plan_subordinate_removal(doc, plan)

        explicit_actor = extract_explicit_actor(doc)
        plan_pronoun_resolution(doc, plan, explicit_actor, last_actor)
        plan_passive_resolution(doc, plan, explicit_actor or last_actor)

        sentence = plan.apply()
        sentence = remove_external_reference_phrases(sentence, existing_articles)
        if not OBLIGATION_RE.search(sentence):
            return None

        # BenePar OFF
        doc      = self._parse_plain(sentence)
        sentence = highlight_or_between_verbs(sentence, doc)
        sentence = highlight_and_between_modal_verbs(sentence, doc)
        sentence = normalize_if(sentence)
        sentence = apply_iaw(sentence)

        return sentence, explicit_actor

    def _split_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        return [s.text.strip() for s in self._parse_plain(text).sents if s.text.strip()]
