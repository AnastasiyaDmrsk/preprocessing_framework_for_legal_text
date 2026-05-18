import re
from typing import Optional, Set

from .const import (
    DEONTIC_MODALS, ACTOR_IGNORE, ACTOR_NER_LABELS,
    LEGAL_ENTITY_KEYWORDS, _DEMONSTRATIVE_STARTS, _RE_DISC_ART, _RE_BARE_ART, _RE_ACT_CITE,
    _RE_ORPH_NUM, _RE_ORPH_DISC, DEONTIC_MODAL_PHRASES, OBLIGATION_RE, CONDITIONAL_SUBORDINATORS, _STRUCTURAL,
)
from .text_utils import TokenTransformPlan, normalize_whitespace


def plan_filler_removal(doc, plan: TokenTransformPlan) -> None:
    """Dependency-based dynamic filler detection for 'where' and 'including'."""
    for token in doc:
        if token.lower_ != "including":
            continue
        if token.dep_ not in {"prep", "mark"}:
            continue
        s_idx = (token.i - 1
                 if token.i > 0 and doc[token.i - 1].text == ","
                 else token.i)
        e_idx = token.i + 1
        for j in range(token.i + 1, len(doc)):
            if doc[j].text in {",", ".", ";"}:
                e_idx = j + 1 if doc[j].text == "," else j
                break
            e_idx = j + 1
        if _including_span_has_meaningful_content(doc, token.i, e_idx):
            continue
        plan.remove_span(s_idx, e_idx)


def remove_external_reference_phrases(
        sentence: str, existing_articles: Set[str]
) -> str:
    if not existing_articles:
        return sentence

    def _is_ext(art_num: str) -> bool:
        return art_num not in existing_articles

    def _sub_disc_art(m: re.Match) -> str:
        a = re.search(r"Article\s+(\d+)", m.group(0), re.IGNORECASE)
        return "" if a and _is_ext(a.group(1)) else m.group(0)

    def _sub_bare_art(m: re.Match) -> str:
        return "" if _is_ext(m.group(1)) else m.group(0)

    sentence = _RE_DISC_ART.sub(_sub_disc_art, sentence)
    sentence = _RE_BARE_ART.sub(_sub_bare_art, sentence)
    sentence = _RE_ACT_CITE.sub("", sentence)
    sentence = _RE_ORPH_NUM.sub("", sentence)
    sentence = _RE_ORPH_DISC.sub("", sentence)

    sentence = re.sub(r"\s+(?:and|or)\s+(?=[,;.])", " ", sentence)
    sentence = re.sub(r"\s+(?:and|or)\s*\.", ".", sentence)
    sentence = re.sub(r"(?<=\w)\s+(?:and|or)\s*$", "", sentence)

    sentence = re.sub(r",\s*,", ",", sentence)
    sentence = re.sub(r",\s*(?=\.)", "", sentence)
    sentence = re.sub(r"\s+,", ",", sentence)

    return normalize_whitespace(sentence)


# Actor extraction
def extract_explicit_actor(doc) -> Optional[str]:
    """
    Extract the BPMN actor(s): all coordinated nsubj tokens that appear
    before the first deontic modal, returned as a single joined string.
    """
    first_modal_idx = _find_first_modal_idx(doc)
    if first_modal_idx is None:
        return None

    for token in doc:
        if token.dep_ not in {"nsubj", "nsubjpass"}:
            continue
        if token.i >= first_modal_idx:
            continue

        if token.pos_ == "PRON":
            return None

        head_parts = sorted(
            [c for c in token.lefts if c.dep_ in {"compound", "amod"}
             and c.pos_ != "DET"]
            + [token]
            + [c for c in token.rights if c.dep_ == "compound"],
            key=lambda t: t.i,
        )
        head_actor = " ".join(t.text for t in head_parts)

        if not head_actor:
            continue
        if head_actor.split()[0].lower() in _DEMONSTRATIVE_STARTS:
            return None
        if not head_actor[0].isupper():
            return None
        if any(ign.lower() in head_actor.lower() for ign in ACTOR_IGNORE):
            return None

        actor_parts: list[str] = [head_actor]

        for conj in sorted(
                (c for c in token.rights if c.dep_ == "conj"),
                key=lambda t: t.i,
        ):
            cc = next((c for c in conj.lefts if c.dep_ == "cc"), None)

            conj_parts = sorted(
                [c for c in conj.lefts if c.dep_ in {"compound", "amod"}
                 and c.pos_ != "DET"]
                + [conj]
                + [c for c in conj.rights if c.dep_ == "compound"],
                key=lambda t: t.i,
            )
            conj_text = " ".join(t.text for t in conj_parts)

            if not conj_text:
                continue
            if not conj_text[0].isupper():
                continue
            if any(ign.lower() in conj_text.lower() for ign in ACTOR_IGNORE):
                continue

            connector = cc.text if cc else "and"
            actor_parts.append(connector)
            actor_parts.append(conj_text)

        return " ".join(actor_parts)

    return None


def has_actor_and_activity(constituent) -> bool:
    """
    True if the constituent contains BOTH:
      (a) a VERB  aka the 'activity', and
      (b) an nsubj / nsubjpass that resolves to a legal entity or role noun.
    """
    if not any(t.pos_ == "VERB" for t in constituent):
        return False

    for token in constituent:
        if token.dep_ not in {"nsubj", "nsubjpass"}:
            continue
        if token.pos_ == "PRON":
            continue
        if any(kw in token.text.lower() for kw in LEGAL_ENTITY_KEYWORDS):
            return True
        for chunk in token.doc.noun_chunks:
            if chunk.root.i == token.i:
                if any(kw in chunk.text.lower() for kw in LEGAL_ENTITY_KEYWORDS):
                    return True
                break
        for ent in token.doc.ents:
            if ent.label_ in ACTOR_NER_LABELS and ent.start <= token.i < ent.end:
                return True
        if token.pos_ in {"NOUN", "PROPN"}:
            if not any(ign.lower() in token.text.lower() for ign in ACTOR_IGNORE):
                return True

    return False


def find_intrasentence_antecedent(pronoun_token) -> Optional[str]:
    """
    Return the main-clause nsubj antecedent for a pronoun in a subordinate clause.
    Only applies when the pronoun appears AFTER the first modal.
    """
    try:
        sent = pronoun_token.sent
    except Exception:
        return None
    first_modal_idx = _find_first_modal_idx(sent)
    if not first_modal_idx:
        return None
    if pronoun_token.i <= first_modal_idx:
        return None
    for token in sent:
        if token.dep_ not in {"nsubj", "nsubjpass"}:
            continue
        if token.i >= first_modal_idx or token.i == pronoun_token.i:
            continue
        parts = sorted(
            [c for c in token.lefts if c.dep_ in {"compound", "amod", "det"}]
            + [token]
            + [c for c in token.rights if c.dep_ == "compound"],
            key=lambda x: x.i,
        )
        antecedent = " ".join(t.text for t in parts).strip()
        return re.sub(r"^[Aa]n?\s+|^The\s+", "the ", antecedent) if antecedent else None
    return None


def plan_pronoun_resolution(
        doc,
        plan: TokenTransformPlan,
        explicit_actor: Optional[str],
        last_actor: Optional[str],
) -> None:
    """Resolve 'it' and 'they' in nsubj/nsubjpass: intrasentence > explicit > last."""
    for token in doc:
        if token.lower_ not in {"it", "they"} or token.dep_ not in {"nsubj", "nsubjpass"}:
            continue
        antecedent = find_intrasentence_antecedent(token)
        if antecedent:
            plan.replace_token(token.i, antecedent)
            continue
        if explicit_actor:
            plan.replace_token(token.i, explicit_actor)
            continue
        if last_actor:
            plan.replace_token(token.i, last_actor)


def plan_passive_resolution(
        doc, plan: TokenTransformPlan, actor_candidate: Optional[str]
) -> None:
    """Append 'by <actor>' after agentless passive participles."""
    for token in doc:
        if token.dep_ == "auxpass" and token.lemma_ == "be":
            if not any(c.dep_ == "agent" for c in token.head.children):
                agent = (
                    actor_candidate
                    if actor_candidate and not any(
                        w.lower() in actor_candidate.lower() for w in ACTOR_IGNORE)
                    else "corresponding authority"
                )
                plan.insert_after(token.head.i, f" by {agent}")


def highlight_gateway_coordinators(sentence: str, doc) -> str:
    """
    Capitalize OR / AND that mark BPMN gateways.
    """
    first_modal_idx = _find_first_modal_idx(doc)
    if first_modal_idx is None:
        return sentence

    modal_head = doc[first_modal_idx].head

    replacements: list[tuple[int, int, str]] = []

    for token in doc:
        if token.lower_ not in {"or", "and"}:
            continue
        if token.dep_ != "cc":
            continue

        head = token.head

        if token.i > first_modal_idx and head.pos_ == "VERB":
            is_modal_head = head.i == modal_head.i
            is_conj_of_root = (head.dep_ == "conj" and head.head.i == modal_head.i)
            if is_modal_head or is_conj_of_root:
                replacements.append((token.idx, token.idx + len(token.text),
                                     token.text.upper()))
            continue

        if token.i < first_modal_idx:
            if head.dep_ in {"nsubj", "nsubjpass"}:
                replacements.append((token.idx, token.idx + len(token.text),
                                     token.text.upper()))
            elif head.dep_ == "conj" and head.head.dep_ in {"nsubj", "nsubjpass"}:
                replacements.append((token.idx, token.idx + len(token.text),
                                     token.text.upper()))

    for char_s, char_e, repl in reversed(sorted(replacements, key=lambda x: x[0])):
        sentence = sentence[:char_s] + repl + sentence[char_e:]

    return sentence


def _has_deontic_in_span(span) -> bool:
    if any(t.lemma_ in DEONTIC_MODALS for t in span):
        return True
    text_lower = span.text.lower()
    return any(phrase in text_lower for phrase in DEONTIC_MODAL_PHRASES)


def _find_first_modal_idx(doc) -> Optional[int]:
    indices: list[int] = []

    for t in doc:
        if t.lemma_ in DEONTIC_MODALS:
            indices.append(t.i)

    lowers = [t.lower_ for t in doc]
    for phrase in DEONTIC_MODAL_PHRASES:
        words = phrase.split()
        n = len(words)
        for i in range(len(lowers) - n + 1):
            if lowers[i: i + n] == words:
                indices.append(i)

    return min(indices) if indices else None


def _has_deontic_in_span_text(text: str) -> bool:
    return bool(OBLIGATION_RE.search(text)) or any(
        ph in text.lower() for ph in DEONTIC_MODAL_PHRASES
    )


def _is_conditional_opener(constituent) -> bool:
    """
    True if the constituent opens with a conditional, temporal, concessive,
    or causal subordinating conjunction.

    Two-tier check:
      1. Explicit CONDITIONAL_SUBORDINATORS membership  (fast path, certain)
      2. POS == SCONJ, excluding pure complementizers "that" / "whether"
         spaCy correctly tags "before", "after", "until", "while", "since",
         "although" etc. as SCONJ in adverbial-clause position.
    """
    if not constituent:
        return False

    first = constituent[0]
    first_lower = first.text.lower()

    if first_lower in CONDITIONAL_SUBORDINATORS:
        return True
    if first.pos_ == "SCONJ" and first_lower not in {"that", "whether"}:
        return True
    return False


def _is_complement_clause(constituent) -> bool:
    """
    True if this constituent is the direct clausal complement (ccomp /
    xcomp) of the main clause verb.
    """
    for token in constituent:
        if token.head.i < constituent.start or token.head.i >= constituent.end:
            if token.dep_ in {"ccomp", "xcomp"}:
                return True
    return False


def _including_span_has_meaningful_content(
        doc, incl_token_i: int, e_idx: int
) -> bool:
    """
    Return True if the content of an 'including ...' phrase contains relevant information
    """

    for i in range(incl_token_i + 1, min(e_idx, len(doc))):
        t = doc[i]
        if t.is_punct or t.is_space:
            continue
        if t.lower_ in _STRUCTURAL:
            continue

        for ent in t.doc.ents:
            if ent.label_ in ACTOR_NER_LABELS and ent.start <= t.i < ent.end:
                return True

        if any(kw in t.text.lower() for kw in LEGAL_ENTITY_KEYWORDS):
            return True

        if t.pos_ in {"NOUN", "PROPN"}:
            return True
    return False
