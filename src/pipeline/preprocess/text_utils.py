import re
from typing import Dict, Set
from spacy.tokens import Doc
from src.pipeline.preprocess.const import (
    STATIC_FILLER_PATTERNS, IAW_RE, ACT_TYPES, _DANGLING,
)
from spacy.matcher import Matcher


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"(?m)^[\s,;:.]+(?=[A-Za-z])", "", text)
    return text.strip()



def apply_iaw(text: str) -> str:
    return IAW_RE.sub("IAW", text)


def apply_static_fillers(text: str) -> str:
    for pattern in STATIC_FILLER_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def normalize_if(sentence: str) -> str:
    """Normalise sentence-initial Where/If to the canonical token IF."""
    s = re.sub(r"^\s*Where\b", "IF", sentence, flags=re.IGNORECASE)
    return re.sub(r"^\s*If\b",  "IF", s, flags=re.IGNORECASE)


class TokenTransformPlan:

    def __init__(self, doc:Doc):
        self._doc:          Doc = doc
        self._remove:       Set[int]        = set()
        self._replace:      Dict[int, str]  = {}
        self._insert_after: Dict[int, str]  = {}

    def remove_span(self, start: int, end: int) -> None:
        for i in range(start, end):
            self._remove.add(i)

    def replace_token(self, token_i: int, text: str) -> None:
        self._replace[token_i] = text  # conflict resolved at apply() time

    def insert_after(self, token_i: int, text: str) -> None:
        self._insert_after[token_i] = text

    def is_empty(self) -> bool:
        return not (self._remove or self._replace or self._insert_after)

    def apply(self) -> str:
        parts: list[str] = []
        for tok in self._doc:
            if tok.i in self._remove:
                continue
            if tok.i in self._replace:
                parts.append(self._replace[tok.i] + tok.whitespace_)
            elif tok.i in self._insert_after:
                parts.append(tok.text_with_ws + self._insert_after[tok.i] + " ")
            else:
                parts.append(tok.text_with_ws)

        result = "".join(parts)

        prev = None
        while prev != result:
            prev = result
            result = _DANGLING.sub(".", result)
            result = re.sub(r",\s*\.", ".", result)
            result = re.sub(r"\s+\.", ".", result)

        return normalize_whitespace(result)


def build_eu_ref_matcher(nlp):
    matcher   = Matcher(nlp.vocab)

    for act in ACT_TYPES:
        words = act.split()
        base  = [{"LOWER": w.lower()} for w in words]
        label = f"EU_ACT_{act.upper().replace(' ', '_')}"

        # Pattern A: Directive (EU) 2016/1148
        matcher.add(label, [
            base + [
                {"TEXT": "("},
                {"TEXT": {"IN": ["EU", "EC", "Euratom"]}},
                {"TEXT": ")"},
                {"LIKE_NUM": True},
                {"TEXT": "/"},
                {"LIKE_NUM": True},
            ],
            # Pattern B: Regulation (EU) No 2016/679
            base + [
                {"TEXT": "("},
                {"TEXT": {"IN": ["EU", "EC", "Euratom"]}},
                {"TEXT": ")"},
                {"TEXT": {"REGEX": r"^[Nn]o\.?$"}},
                {"LIKE_NUM": True},
                {"TEXT": "/"},
                {"LIKE_NUM": True},
            ],
            # Pattern C: Directive (EC) 2009/136/EC  (trailing jurisdiction suffix)
            base + [
                {"TEXT": "("},
                {"TEXT": {"IN": ["EU", "EC", "Euratom"]}},
                {"TEXT": ")"},
                {"LIKE_NUM": True},
                {"TEXT": "/"},
                {"LIKE_NUM": True},
                {"TEXT": "/"},
                {"TEXT": {"IN": ["EU", "EC"]}},
            ],
        ])

    return matcher
