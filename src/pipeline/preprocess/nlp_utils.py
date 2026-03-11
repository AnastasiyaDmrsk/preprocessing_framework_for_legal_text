import re
from typing import Optional, Set

from .const import (
    DEONTIC_MODALS, ACTOR_IGNORE, ACTOR_NER_LABELS,
    LEGAL_ENTITY_KEYWORDS, _DEMONSTRATIVE_STARTS, _RE_DISC_ART, _RE_BARE_ART, _RE_ACT_CITE,
    _RE_ORPH_NUM, _RE_ORPH_DISC,
)
from .text_utils import TokenTransformPlan, normalize_whitespace


# Filler removal

def plan_filler_removal(doc, plan: TokenTransformPlan) -> None:
    """Dependency-based dynamic filler detection for 'where' and 'including'."""
    for token in doc:
        if token.lower_ == "where" and token.dep_ in {"advmod", "mark"}:
            s_idx, e_idx = token.i, token.i + 1
            for j in range(token.i + 1, len(doc)):
                if doc[j].pos_ in {"ADJ", "ADV", "NOUN"}:
                    e_idx = j + 1
                else:
                    break
            if e_idx < len(doc) and doc[e_idx].text == ",":
                e_idx += 1
            if s_idx > 0 and doc[s_idx - 1].text == ",":
                s_idx -= 1
            plan.remove_span(s_idx, e_idx)

        elif token.lower_ == "including" and token.dep_ in {"prep", "mark"}:
            s_idx = token.i - (1 if token.i > 0 and doc[token.i - 1].text == "," else 0)
            e_idx = token.i + 1
            for j in range(token.i + 1, len(doc)):
                if doc[j].text in {",", ".", ";"}:
                    e_idx = j + 1 if doc[j].text == "," else j
                    break
                e_idx = j + 1
            plan.remove_span(s_idx, e_idx)


# External reference removal
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

    sentence = _RE_DISC_ART.sub(_sub_disc_art, sentence)  # discourse + Article X
    sentence = _RE_BARE_ART.sub(_sub_bare_art, sentence)  # bare Article X
    sentence = _RE_ACT_CITE.sub("", sentence)  # Directive/Regulation
    sentence = _RE_ORPH_NUM.sub("", sentence)  # orphaned (2), (4)
    sentence = _RE_ORPH_DISC.sub("", sentence)  # orphaned discourse at end

    # Clean up orphaned conjunctions before punctuation or at sentence end
    sentence = re.sub(r"\s+(?:and|or)\s+(?=[,;.])", " ", sentence)
    sentence = re.sub(r"\s+(?:and|or)\s*\.", ".", sentence)
    sentence = re.sub(r"(?<=\w)\s+(?:and|or)\s*$", "", sentence)

    sentence = re.sub(r",\s*,", ",", sentence)
    sentence = re.sub(r",\s*(?=\.)", "", sentence)
    sentence = re.sub(r"\s+,", ",", sentence)

    return normalize_whitespace(sentence)


# Actor extraction
def extract_explicit_actor(doc) -> Optional[str]:
    """Extract the BPMN actor: nsubj before the first deontic modal."""
    modals = [t for t in doc if t.lemma_ in DEONTIC_MODALS]
    if not modals:
        return None
    first_modal = modals[0]
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"} and token.i < first_modal.i:
            parts = (
                    [c for c in token.lefts if c.dep_ in {"compound", "amod"} and c.pos_ != "DET"]
                    + [token]
                    + [c for c in token.rights if c.dep_ == "compound"]
            )
            actor = " ".join(t.text for t in sorted(parts, key=lambda x: x.i))
            if not actor or actor.split()[0].lower() in _DEMONSTRATIVE_STARTS:
                return None
            if actor[0].isupper():
                if not any(ign.lower() in actor.lower() for ign in ACTOR_IGNORE):
                    return actor
    return None


def has_actor_and_activity(constituent) -> bool:
    """
    G5: return True if the constituent has an nsubj/nsubjpass resolving to a
    legal entity keyword, noun chunk, or NER actor label.
    """
    for token in constituent:
        if token.dep_ not in {"nsubj", "nsubjpass"}:
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
    return False


# Pronoun and passive resolution

def find_intrasentence_antecedent(pronoun_token) -> Optional[str]:
    """
    Return the main-clause nsubj antecedent for a pronoun in a subordinate clause.
    Only applies when the pronoun appears AFTER the first modal (subordinate position).
    """
    try:
        sent = pronoun_token.sent
    except Exception:
        return None
    modals = [t for t in sent if t.lemma_ in DEONTIC_MODALS]
    if not modals:
        return None
    first_modal = modals[0]
    if pronoun_token.i <= first_modal.i:
        return None
    for token in sent:
        if token.dep_ not in {"nsubj", "nsubjpass"}:
            continue
        if token.i >= first_modal.i or token.i == pronoun_token.i:
            continue
        parts = sorted(
            [c for c in token.lefts  if c.dep_ in {"compound", "amod", "det"}]
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
    last_actor:     Optional[str],
) -> None:
    """Resolve 'it' and 'they' in nsubj/nsubjpass: intrasentence > explicit > last."""
    for token in doc:
        if token.lower_ not in {"it", "they"} or token.dep_ not in {"nsubj", "nsubjpass"}:
            continue
        antecedent = find_intrasentence_antecedent(token)
        if antecedent:
            plan.replace_token(token.i, antecedent)
            return
        if explicit_actor:
            plan.replace_token(token.i, explicit_actor)
            return
        if last_actor:
            plan.replace_token(token.i, last_actor)
            return


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
                plan.insert_after(token.head.i, f"by {agent}")
                return


# Gateway highlighting

def highlight_or_between_verbs(sentence: str, doc) -> str:
    """Capitalize 'or' between two verbs to mark an exclusive gateway."""
    if sum(1 for t in doc if t.pos_ == "VERB") < 2:
        return sentence
    for token in doc:
        if token.lower_ == "or":
            left  = any(doc[j].pos_ == "VERB" for j in range(max(0, token.i - 2), token.i))
            right = any(doc[j].pos_ == "VERB" for j in range(token.i + 1, min(len(doc), token.i + 3)))
            if left and right:
                s, e = token.idx, token.idx + len(token.text)
                return sentence[:s] + "OR" + sentence[e:]
    return sentence


def highlight_and_between_modal_verbs(sentence: str, doc) -> str:
    """Capitalize 'and' between two modal-verb clauses to mark a parallel gateway."""
    modals = [t for t in doc if t.lemma_ in DEONTIC_MODALS]
    if len(modals) < 2:
        return sentence
    for i, m1 in enumerate(modals[:-1]):
        m2 = modals[i + 1]
        for token in doc:
            if token.lower_ == "and" and token.pos_ == "CCONJ" and m1.i < token.i < m2.i:
                if any(t.pos_ == "VERB" and token.i < t.i <= token.i + 3 for t in doc):
                    s, e = token.idx, token.idx + len(token.text)
                    return sentence[:s] + "AND" + sentence[e:]
    return sentence
