import re
from pathlib import Path
from typing import Tuple, List, Optional, Set
import spacy

# --- Regex patterns ---
ARTICLE_STANDARD_RE = re.compile(r'(?m)^(Article\s+(\d+))\s*\n+([^\n]+)', flags=re.IGNORECASE)
ARTICLE_ALT_RE = re.compile(r'(?m)^(Art\.\s+(\d+)(?:\s+[A-Z]+)?)\.?\s+([^\n]+)', flags=re.IGNORECASE)
SECTION_NUMBERING_RE = re.compile(r'(?m)^(\d+(?:\.\d+)*)\.\s+([^\n]+)', flags=re.IGNORECASE)

# External references pattern - improved to capture Directive/Regulation with year
CROSS_REF_ACT_RE = re.compile(
    r'\b(Directive|Regulation|Implementing Regulation)\s*\((?:EU|EC)\)\s*\d{4}/\d+',
    flags=re.IGNORECASE
)

ARTICLE_MULTI_REF_RE = re.compile(
    r'\bArticles?\s+((?:\d+(?:\([0-9]+\))?)(?:\s*,\s*\d+(?:\([0-9]+\))?)*(?:\s*and\s*\d+(?:\([0-9]+\))?)?)',
    flags=re.IGNORECASE
)
ARTICLE_SINGLE_REF_RE = re.compile(r'\bArticle\s+([0-9]+(?:\([0-9]+\))?)', flags=re.IGNORECASE)


class RegulatoryTextPreprocessor:
    """
    Preprocess regulatory text for BPMN/control-flow extraction with NLP.

    Optional: For better legal entity recognition, consider installing:
    - LexNLP: pip install lexnlp
    - Legal NER model: python -m spacy download en_legal_ner_trf
    """

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "SpaCy model 'en_core_web_sm' not found. Install: python -m spacy download en_core_web_sm")

        # Try to load legal NER model if available
        self.legal_nlp = None
        try:
            self.legal_nlp = spacy.load("en_legal_ner_trf")
        except:
            pass  # Fall back to standard model

    def preprocess(self, input_text: str) -> Tuple[str, List[str]]:
        doc_structure = self._detect_document_structure(input_text)
        if doc_structure == "standard_article":
            return self._preprocess_standard_articles(input_text)
        elif doc_structure == "alt_article":
            return self._preprocess_alt_articles(input_text)
        elif doc_structure == "section_numbering":
            return self._preprocess_section_numbering(input_text)
        else:
            return self._preprocess_raw_text(input_text)

    def _detect_document_structure(self, text: str) -> str:
        if ARTICLE_STANDARD_RE.search(text):
            return "standard_article"
        if ARTICLE_ALT_RE.search(text):
            return "alt_article"
        if SECTION_NUMBERING_RE.search(text):
            return "section_numbering"
        return "raw_text"

    def _collect_existing_articles(self, text: str, pattern) -> Set[str]:
        existing = set()
        for m in pattern.finditer(text):
            if len(m.groups()) >= 2:
                existing.add(m.group(2))
        return existing

    def _preprocess_standard_articles(self, input_text: str) -> Tuple[str, List[str]]:
        existing_articles = self._collect_existing_articles(input_text, ARTICLE_STANDARD_RE)
        articles = self._split_standard_articles(input_text)
        return self._process_articles(articles, existing_articles)

    def _preprocess_alt_articles(self, input_text: str) -> Tuple[str, List[str]]:
        existing_articles = self._collect_existing_articles(input_text, ARTICLE_ALT_RE)
        articles = self._split_alt_articles(input_text)
        return self._process_articles(articles, existing_articles)

    def _split_standard_articles(self, text: str):
        matches = list(ARTICLE_STANDARD_RE.finditer(text))
        if not matches:
            return [("Article 0", "", text)]
        res = []
        for i, m in enumerate(matches):
            art_full = m.group(1)
            heading = m.group(3).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            res.append((art_full, heading, text[start:end].strip()))
        return res

    def _split_alt_articles(self, text: str):
        matches = list(ARTICLE_ALT_RE.finditer(text))
        if not matches:
            return [("Art. 0", "", text)]
        res = []
        for i, m in enumerate(matches):
            art_full = m.group(1)
            heading = m.group(3).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            res.append((art_full, heading, text[start:end].strip()))
        return res

    def _process_articles(self, articles, existing_articles: Set[str]) -> Tuple[str, List[str]]:
        """Process articles with new format: Article once, then paragraph numbers."""
        clauses, references_set = [], set()
        last_actor = None

        for art_idx, (art_id, heading, art_text) in enumerate(articles):
            # Add article header with title
            clauses.append(f"{art_id}: {heading}")

            paragraphs = self._parse_paragraphs_with_lists(art_text, art_id)
            current_para_num = None

            for para_loc, para_content, is_list_item, is_last_item, list_letter in paragraphs:
                # Extract paragraph number from location (e.g., "Article 23.1" -> "1")
                para_num = para_loc.split('.')[-1].split('(')[0].strip()

                processed = self._process_sentence(para_content, para_loc, existing_articles, last_actor,
                                                   references_set)
                if processed:
                    sent, actor = processed
                    if is_list_item:
                        sent = sent.rstrip('.') + (" AND" if not is_last_item else "")

                    # Format with list letter if applicable
                    if list_letter:
                        clauses.append(f"{list_letter} {sent}")
                    elif para_num != current_para_num:
                        clauses.append(f"{para_num} {sent}")
                        current_para_num = para_num
                    else:
                        clauses.append(sent)

                    if actor:
                        last_actor = actor

            # Add separator after each article (except the last one)
            if art_idx < len(articles) - 1:
                clauses.append("---")

        return "\n".join(clauses), sorted(list(references_set))

    def _parse_paragraphs_with_lists(self, art_text: str, art_id: str):
        """Parse paragraphs and detect lists with patterns: (a)...; or a)...;"""
        result = []
        para_splits = re.split(r'(?m)^(\d+)\.\s*', art_text)

        for i in range(1, len(para_splits), 2):
            if i + 1 >= len(para_splits):
                continue

            para_num = para_splits[i]
            para_text = para_splits[i + 1].strip()
            para_loc = f"{art_id}.{para_num}"

            # Detect list patterns: (a) or a) followed by content
            list_items_paren = list(re.finditer(r'\(([a-z])\)\s*\n(.+?)(?=\n\s*\([a-z]\)|\Z)', para_text, re.DOTALL))
            list_items_no_paren = list(
                re.finditer(r'^([a-z])\)\s*\n?(.+?)(?=^[a-z]\)|\Z)', para_text, re.MULTILINE | re.DOTALL))

            list_items = list_items_paren if list_items_paren else list_items_no_paren

            if list_items and len(list_items) > 1:
                # Extract leading text (everything before first list item)
                first_match = list_items[0]
                leading_text = para_text[:first_match.start()].strip()
                # Remove trailing colon and "the following" patterns
                leading_text = re.sub(r':\s*$', '', leading_text)
                leading_text = re.sub(r'\s+(?:the\s+following|as\s+follows)\s*$', '', leading_text, flags=re.IGNORECASE)
                leading_text = leading_text.strip()

                # Process each list item with letter preserved
                for idx, match in enumerate(list_items):
                    letter = para_num + "(" + match.group(1) + ")" if list_items_paren else para_num + match.group(1) + ")"
                    item_text = match.group(2).strip().rstrip(';').rstrip('.').strip()
                    combined = f"{leading_text} {item_text}"
                    item_loc = f"{para_loc} ({letter})"
                    is_last = (idx == len(list_items) - 1)
                    result.append((item_loc, combined, True, is_last, letter))
            else:
                # No list detected, process as regular sentences
                sentences = self._split_sentences(para_text)
                for sent in sentences:
                    if sent.strip():
                        result.append((para_loc, sent.strip(), False, False, None))

        return result

    def _preprocess_section_numbering(self, input_text: str) -> Tuple[str, List[str]]:
        sections = self._split_section_numbering(input_text)
        clauses, references_set = [], set()
        last_actor = None

        for sec_id, heading, sec_text in sections:
            if heading:
                clauses.append(f"{sec_id} {heading}".strip())
            sentences = self._split_sentences(sec_text)
            for sent in sentences:
                if sent.strip():
                    processed = self._process_sentence(sent.strip(), sec_id, set(), last_actor, references_set)
                    if processed:
                        sent_proc, actor = processed
                        clauses.append(f"{sec_id} {sent_proc}")
                        if actor:
                            last_actor = actor

        return "\n".join(clauses), sorted(list(references_set))

    def _split_section_numbering(self, text: str):
        matches = list(SECTION_NUMBERING_RE.finditer(text))
        if not matches:
            return [("0", "", text)]
        res = []
        for i, m in enumerate(matches):
            sec_num = m.group(1)
            heading = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            res.append((sec_num, heading, text[start:end].strip()))
        return res

    def _preprocess_raw_text(self, input_text: str) -> Tuple[str, List[str]]:
        clauses, references_set = [], set()
        last_actor = None
        sentences = self._split_sentences(input_text)
        for sent in sentences:
            if sent.strip():
                processed = self._process_sentence(sent.strip(), "", set(), last_actor, references_set)
                if processed:
                    sent_proc, actor = processed
                    clauses.append(sent_proc)
                    if actor:
                        last_actor = actor
        return "\n".join(clauses), sorted(list(references_set))

    def _process_sentence(self, sentence: str, location: str, existing_articles: Set[str],
                          last_actor: Optional[str], references_set: Set[str]) -> Optional[Tuple[str, Optional[str]]]:
        """NLP pipeline for sentence processing."""
        # Extract references FIRST (before any text modifications)
        sent, refs = self._extract_references_dynamic(sentence, location)
        references_set.update(refs)

        # Remove external references from text
        sent = self._remove_external_references(sent)

        # Now proceed with rest of processing
        sent = self._remove_filler_phrases_dynamic(sent)
        sent = self._remove_subordinate_clauses_improved(sent)

        if existing_articles:
            sent = self._remove_nonexisting_references(sent, existing_articles)

        if not self._is_obligation_clause(sent):
            return None

        doc = self.nlp(sent)
        explicit_actor = self._extract_explicit_actor_dep(doc)
        sent = self._resolve_pronoun_it_dep(doc, sent, explicit_actor, last_actor)

        doc = self.nlp(sent)
        explicit_actor = self._extract_explicit_actor_dep(doc)
        sent = self._resolve_passive_missing_agent_dep(doc, sent, explicit_actor or last_actor)
        sent = self._highlight_or_between_verbs_only(sent)
        sent = self._highlight_and_between_modal_verbs_only(sent)
        sent = self._normalize_if(sent)

        return (sent, explicit_actor)

    def _remove_external_references(self, sentence: str) -> str:
        """Remove external Directive/Regulation references from text with proper spacing."""
        ranges_to_remove = []
        for m in CROSS_REF_ACT_RE.finditer(sentence):
            ranges_to_remove.append((m.start(), m.end()))

        if not ranges_to_remove:
            return sentence

        result = []
        last_end = 0
        for start, end in ranges_to_remove:
            result.append(sentence[last_end:start])
            last_end = end
        result.append(sentence[last_end:])

        final = ''.join(result)
        final = re.sub(r'\s+', ' ', final)
        final = re.sub(r'\bunder\s+([,.])', r'\1', final)
        final = re.sub(r'\bunder\s*$', '', final)
        return final.strip()

    def _remove_filler_phrases_dynamic(self, sentence: str) -> str:
        """Use dependency parsing to remove filler phrases, plus static patterns."""
        doc = self.nlp(sentence)
        indices_to_remove = []

        # Dynamic removal with dependency parsing
        for token in doc:
            if token.lower_ == "where" and token.dep_ in {"advmod", "mark"}:
                start_idx, end_idx = token.i, token.i + 1
                for i in range(token.i + 1, len(doc)):
                    if doc[i].pos_ in {"ADJ", "ADV", "NOUN"}:
                        end_idx = i + 1
                    else:
                        break
                if end_idx < len(doc) and doc[end_idx].text == ',':
                    end_idx += 1
                if start_idx > 0 and doc[start_idx - 1].text == ',':
                    start_idx -= 1
                indices_to_remove.append((start_idx, end_idx))

            elif token.lower_ == "including" and token.dep_ in {"prep", "mark"}:
                start_idx = token.i
                if start_idx > 0 and doc[start_idx - 1].text == ',':
                    start_idx -= 1
                end_idx = token.i + 1
                for i in range(token.i + 1, len(doc)):
                    if doc[i].text in {',', '.', ';'}:
                        end_idx = i + 1 if doc[i].text == ',' else i
                        break
                    end_idx = i + 1
                indices_to_remove.append((start_idx, end_idx))

        # Static filler removal patterns (including "where necessary")
        static_fillers = [
            r',?\s*without undue delay,?\s*',
            r',?\s*inter alia,?\s*',
            r',?\s*for example:?,?\s*',
            r',?\s*where necessary,?\s*',
            r',?\s*where appropriate,?\s*',
            r',?\s*as appropriate,?\s*'
        ]
        result = sentence
        for filler in static_fillers:
            result = re.sub(filler, ' ', result, flags=re.IGNORECASE)

        # Apply dynamic removals
        if indices_to_remove:
            # Re-parse after static removal
            doc = self.nlp(result)
            kept_tokens = []
            skip_until = -1
            for i, token in enumerate(doc):
                if i < skip_until:
                    continue
                should_skip = False
                for start, end in indices_to_remove:
                    if start <= i < end:
                        should_skip = True
                        skip_until = end
                        break
                if not should_skip:
                    kept_tokens.append(token.text_with_ws)
            result = ''.join(kept_tokens)

        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\s+([,.;:])', r'\1', result)
        return result

    def _remove_subordinate_clauses_improved(self, sentence: str) -> str:
        """
        Remove subordinate clauses but preserve legal entity phrases.
        Improved to detect entities like "essential and important entities".
        """
        doc = self.nlp(sentence) if not self.legal_nlp else self.legal_nlp(sentence)

        # Detect legal entity phrases (noun phrases with entities/organizations)
        entity_spans = set()
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if any(keyword in chunk_text for keyword in [
                'entities', 'entity', 'authority', 'authorities', 'member state',
                'commission', 'council', 'operator', 'provider', 'recipient',
                'controller', 'processor', 'organisation', 'organization'
            ]):
                entity_spans.add((chunk.start, chunk.end))

        def check_clause(match):
            clause = match.group(0)

            # Check if contains obligation verb
            if re.search(r'\b(shall|must|may|should)\b', clause, flags=re.IGNORECASE):
                return clause

            # Check if the clause contains a legal entity phrase
            clause_doc = self.nlp(clause)
            for chunk in clause_doc.noun_chunks:
                chunk_text = chunk.text.lower()
                if any(keyword in chunk_text for keyword in [
                    'entities', 'entity', 'authority', 'authorities', 'member state',
                    'essential', 'important', 'critical'
                ]):
                    return clause

            return ''

        result = re.sub(r'\s+(?:that|which|who)\s+[^,.;]+', check_clause, sentence, flags=re.IGNORECASE)
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\s+([,.;:])', r'\1', result)
        return result

    def _extract_references_dynamic(self, sentence: str, location: str) -> Tuple[str, List[str]]:
        refs = []
        if not location:
            return sentence, refs

        for m in CROSS_REF_ACT_RE.finditer(sentence):
            refs.append(f"{location},{m.group(0).strip()}")

        doc = self.nlp(sentence)
        reference_markers = {"in accordance with", "pursuant to", "referred to in", "set out in", "as referred to in",
                             "under"}
        for token in doc:
            phrase = " ".join([t.text.lower() for t in doc[max(0, token.i - 3):token.i + 1]])
            for marker in reference_markers:
                if marker in phrase:
                    for i in range(token.i, min(token.i + 10, len(doc))):
                        if doc[i].text.lower() == "article":
                            for j in range(i + 1, min(i + 5, len(doc))):
                                if doc[j].like_num:
                                    refs.append(f"{location},Article {doc[j].text}")

        for m in ARTICLE_MULTI_REF_RE.finditer(sentence):
            parts = re.split(r",|and", m.group(1))
            for p in parts:
                num = p.strip()
                if num:
                    refs.append(f"{location},Article {num}")

        for m in ARTICLE_SINGLE_REF_RE.finditer(sentence):
            refs.append(f"{location},Article {m.group(1)}")

        return sentence, refs

    def _remove_nonexisting_references(self, sentence: str, existing_articles: Set[str]) -> str:
        doc = self.nlp(sentence)
        indices_to_remove = []

        reference_markers = {"in accordance with", "pursuant to", "referred to in", "set out in"}
        for token in doc:
            phrase = " ".join([t.text.lower() for t in doc[max(0, token.i - 3):token.i + 1]])
            for marker in reference_markers:
                if marker in phrase:
                    found_existing = False
                    for i in range(token.i, min(token.i + 10, len(doc))):
                        if doc[i].text.lower() == "article":
                            for j in range(i + 1, min(i + 5, len(doc))):
                                if doc[j].like_num and doc[j].text in existing_articles:
                                    found_existing = True
                                    break

                    if not found_existing:
                        start_idx = max(0, token.i - 3)
                        end_idx = min(token.i + 10, len(doc))
                        indices_to_remove.append((start_idx, end_idx))

        if indices_to_remove:
            kept_tokens = []
            skip_until = -1
            for i, token in enumerate(doc):
                if i < skip_until:
                    continue
                should_skip = False
                for start, end in indices_to_remove:
                    if start <= i < end:
                        should_skip = True
                        skip_until = end
                        break
                if not should_skip:
                    kept_tokens.append(token.text_with_ws)
            sentence = ''.join(kept_tokens)

        sentence = re.sub(r'\s+', ' ', sentence).strip()
        sentence = re.sub(r'\s+([,.;:])', r'\1', sentence)
        sentence = re.sub(r'\s*,\s*\.', '.', sentence)
        return sentence

    def _split_sentences(self, text: str) -> List[str]:
        return re.split(r'(?<=[.;])\s+(?=[A-Z])', text.strip())

    def _is_obligation_clause(self, sentence: str) -> bool:
        return bool(re.search(r"\b(shall|must|may|should)\b", sentence, flags=re.IGNORECASE))

    def _extract_explicit_actor_dep(self, doc) -> Optional[str]:
        ignore = {"Article", "Paragraph", "Section", "Union", "programme", "work", "rolling", "scheme",
                  "acts", "incident", "entities", "recipients", "services", "measures", "remedies",
                  "notification", "information", "status"}

        modal_verbs = [t for t in doc if t.lemma_ in {"shall", "must", "may", "should"}]
        if not modal_verbs:
            return None

        first_modal = modal_verbs[0]
        for token in doc:
            if token.dep_ in {"nsubj", "nsubjpass"} and token.i < first_modal.i:
                actor_tokens = []
                for child in token.lefts:
                    if child.dep_ in {"compound", "amod"} and child.pos_ != "DET":
                        actor_tokens.append(child)
                actor_tokens.append(token)
                for child in token.rights:
                    if child.dep_ == "compound":
                        actor_tokens.append(child)

                actor_text = " ".join([t.text for t in sorted(actor_tokens, key=lambda x: x.i)])
                if actor_text and actor_text[0].isupper():
                    if not any(ign.lower() in actor_text.lower() for ign in ignore):
                        return actor_text
        return None

    def _resolve_pronoun_it_dep(self, doc, sentence: str, explicit_actor: Optional[str],
                                last_actor: Optional[str]) -> str:
        actor = explicit_actor or last_actor
        if not actor:
            return sentence
        for token in doc:
            if token.lower_ == "it" and token.dep_ in {"nsubj", "nsubjpass"}:
                start, end = token.idx, token.idx + len(token.text)
                sentence = sentence[:start] + actor + sentence[end:]
                break
        return sentence

    def _resolve_passive_missing_agent_dep(self, doc, sentence: str, actor_candidate: Optional[str]) -> str:
        ignore = {"Article", "Paragraph", "Section", "programme", "work", "scheme", "acts",
                  "incident", "entities", "notification", "information", "status"}

        for token in doc:
            if token.dep_ == "auxpass" and token.lemma_ == "be":
                has_agent = any(child.dep_ == "agent" for child in token.head.children)
                if not has_agent:
                    participle = token.head
                    if actor_candidate and not any(w.lower() in actor_candidate.lower() for w in ignore):
                        end_idx = participle.idx + len(participle.text)
                        sentence = sentence[:end_idx] + f" by {actor_candidate}" + sentence[end_idx:]
                    else:
                        end_idx = participle.idx + len(participle.text)
                        sentence = sentence[:end_idx] + " by corresponding authority" + sentence[end_idx:]
                    break
        return sentence

    def _highlight_or_between_verbs_only(self, sentence: str) -> str:
        doc = self.nlp(sentence)
        verbs = [t for t in doc if t.pos_ == "VERB"]
        if len(verbs) < 2:
            return sentence

        for token in doc:
            if token.lower_ == "or":
                left = doc[token.i - 1] if token.i > 0 else None
                right = doc[token.i + 1] if token.i < len(doc) - 1 else None

                if left and right:
                    has_verb_left = any(doc[j].pos_ == "VERB" for j in range(max(0, token.i - 2), token.i))
                    has_verb_right = any(doc[j].pos_ == "VERB" for j in range(token.i + 1, min(len(doc), token.i + 3)))

                    if has_verb_left and has_verb_right:
                        start, end = token.idx, token.idx + len(token.text)
                        sentence = sentence[:start] + "OR" + sentence[end:]
                        break

        return sentence

    def _highlight_and_between_modal_verbs_only(self, sentence: str) -> str:
        doc = self.nlp(sentence)
        modals = [t for t in doc if t.lemma_ in {"shall", "must", "may", "should"}]
        if len(modals) < 2:
            return sentence
        for i, m1 in enumerate(modals[:-1]):
            m2 = modals[i + 1]
            for token in doc:
                if token.lower_ == "and" and token.pos_ == "CCONJ" and m1.i < token.i < m2.i:
                    has_verb = any(t.pos_ == "VERB" and token.i < t.i <= token.i + 3 for t in doc)
                    if has_verb:
                        start, end = token.idx, token.idx + len(token.text)
                        sentence = sentence[:start] + "AND" + sentence[end:]
                        break
        return sentence

    def _normalize_if(self, sentence: str) -> str:
        s = re.sub(r"^\s*Where\b", "IF", sentence, flags=re.IGNORECASE)
        s = re.sub(r"^\s*If\b", "IF", s, flags=re.IGNORECASE)
        return s


def preprocess_legal_text(input_text: str, path: Path) -> Tuple[str, List[str]]:
    """
    Preprocess regulatory text for BPMN/control-flow extraction.

    New output format:
    - Article listed once with heading
    - Paragraph numbers only (not repeated Article.X)
    - List items with letters (a, b, c...)
    - Separator "---" between articles

    Supports: Standard articles, Alt format (Art. X), Section numbering, Raw text
    Features:
    - Dynamic NLP-based filler removal (including "where necessary", "where appropriate")
    - Improved legal entity recognition (preserves "essential and important entities")
    - List detection with leading text attachment
    - Actor extraction and pronoun resolution
    - Gateway highlighting (OR/AND between verbs/modals)
    - Reference validation
    - Proper spacing preservation

    Requirements: python -m spacy download en_core_web_sm

    Optional for better legal entity recognition:
    - Legal NER model: python -m spacy download en_legal_ner_trf
    - LexNLP: pip install lexnlp

    Args:
        input_text: Raw regulatory text
        path: Output directory

    Returns:
        Tuple of (preprocessed_text, references_list)
    """
    preprocessor = RegulatoryTextPreprocessor()
    preprocessed_text, references = preprocessor.preprocess(input_text)

    path.mkdir(parents=True, exist_ok=True)
    with open(path / "preprocess.txt", "w", encoding="utf-8") as f:
        f.write(preprocessed_text)
    with open(path / "references.csv", "w", encoding="utf-8") as f:
        f.write("Location,Reference\n")
        for ref in references:
            f.write(ref + "\n")

    return preprocessed_text, references
