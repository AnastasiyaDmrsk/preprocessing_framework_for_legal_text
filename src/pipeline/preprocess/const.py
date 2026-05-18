import re
from typing import FrozenSet

_BENEPAR_MAX_TOKENS = 100
_SECTION_MIN_MATCHES = 3
_SECTION_HEADING_MAX_WORDS = 10
_REF_CONTEXT_WINDOW = 60

# Document structure
ARTICLE_STANDARD_RE = re.compile(r"(?m)^(Article\s+(\d+))\s*\n+([^\n]+)", re.IGNORECASE)
ARTICLE_ALT_RE = re.compile(r"(?m)^(Art\.\s+(\d+)(?:\s+[A-Z]+)?)\.?\s+([^\n]+)", re.IGNORECASE)
SECTION_NUMBERING_RE = re.compile(r"(?m)^(\d+(?:\.\d+)*)\.\s+([^\n]+)", re.IGNORECASE)
_PARAGRAPH_STARTERS = frozenset({
    "the", "a", "an", "this", "these", "that", "those",
    "it", "they", "each", "any", "all", "no", "in", "for",
    "where", "if", "unless", "member",
})

_STRUCTURAL = frozenset({"and", "or", "nor", "the", "a", "an"})

PARA_BODY_RE = re.compile(
    r"(?m)^\d{1,2}\.\s+(?=[A-Z])",
)

# Cross-references
ACT_TYPES = [
    "Directive",
    "Regulation",
    "Decision",
    "Implementing Regulation",
    "Delegated Regulation",
    "Implementing Decision",
    "Delegated Decision",
]
ARTICLE_MULTI_REF_RE = re.compile(
    r"\bArticles?\s+((?:\d+(?:\([0-9]+\))?)(?:\s*,\s*\d+(?:\([0-9]+\))?)*(?:\s*and\s*\d+(?:\([0-9]+\))?)?)",
    re.IGNORECASE)
ARTICLE_SINGLE_REF_RE = re.compile(r"\bArticle\s+([0-9]+(?:\([0-9]+\))?)", re.IGNORECASE)
INTERNAL_REF_RE = re.compile(
    r"\b(?:paragraph|subparagraph|article|section|annex|indent|point"
    r"|sub-paragraph|recital|schedule|appendix)"
    r"\s+"
    r"(?:\d+|[IVXLCDM]{1,10})"
    r"(?:\s*\(\s*\d+\s*\))?"
    r"(?:\s*,\s*point\s+\d+)?",
    re.IGNORECASE,
)

# Deontic modality
DEONTIC_MODALS: FrozenSet[str] = frozenset({"shall", "must", "may", "should"})
DEONTIC_MODAL_PHRASES: FrozenSet[str] = frozenset({"have to", "required to", "obligated to", })
OBLIGATION_RE = re.compile(
    r"\b(shall|must|may|should|have to|required to|obligated to)\b",
    re.IGNORECASE)

# Text normalisation
IAW_RE = re.compile(r"\bin accordance with\b", re.IGNORECASE)
STATIC_FILLER_PATTERNS = [
    re.compile(r",?\s*without undue delay,?\s*", re.IGNORECASE),
    re.compile(r",?\s*inter alia,?\s*", re.IGNORECASE),
    re.compile(r",?\s*for example:?,?\s*", re.IGNORECASE),
    re.compile(r",?\s*as appropriate,?\s*", re.IGNORECASE),
    re.compile(r",?\s*by means of implementing acts,?\s*", re.IGNORECASE),
    re.compile(r",?\s*in particular,?\s*", re.IGNORECASE),
]

# NLP
# BenePar labels identifying subordinate clauses
SUBORDINATE_LABELS: FrozenSet[str] = frozenset({"SBAR", "WHNP", "WHADVP", "WHPP"})

# First-token subordinators that introduce gateway conditions
# const.py
CONDITIONAL_SUBORDINATORS: FrozenSet[str] = frozenset({
    "if", "unless", "provided", "subject", "except",
    "in the event", "where", "notwithstanding",
    "when", "before", "after", "until", "once", "while",
    "whenever", "as soon as", "as long as",
    "although", "though", "even", "because", "since",
})

# Markers used to locate article cross-references; IAW replacement happens AFTER these lookups
REFERENCE_MARKERS: FrozenSet[str] = frozenset({
    "in accordance with", "pursuant to", "referred to in", "as defined in", "listed in", "specified in", "referred to under",
    "set out in", "as referred to in", "under", "IAW", "laid down in", "provided for in", "provided for by", "as provided in",
    "in line with", "according to", "by virtue of",
})

# Actor / entity vocabularies
# Keywords that identify legal actors in subordinate clause guards (G5)
LEGAL_ENTITY_KEYWORDS: FrozenSet[str] = frozenset({
    "entities", "entity", "authority", "authorities", "member state",
    "commission", "council", "operator", "provider", "recipient",
    "controller", "processor", "organisation", "organization",
    "essential", "important", "critical",
})

# spaCy NER labels accepted as valid process actors
ACTOR_NER_LABELS: FrozenSet[str] = frozenset({"PERSON", "ORG", "GPE", "NORP"})

_DEMONSTRATIVE_STARTS = frozenset({"those", "these", "this", "that", "such", "it", "they", "he", "she", "we", "i", "you",})

# Syntactic subjects excluded from top-level BPMN actor extraction
ACTOR_IGNORE: FrozenSet[str] = frozenset({
    "Article", "Paragraph", "Section", "Union", "programme", "work", "rolling",
    "scheme", "acts", "incident", "entities", "recipients", "services", "measures",
    "remedies", "notification", "information", "status",
    "data", "record", "document", "report", "request", "response",
    "measure", "provision", "requirement", "obligation", "right",
    "period", "date", "time", "year", "month", "day",
})

# BPMN gateway markers
PARALLEL_GATEWAY_START = "--PARALLEL TASKS--"
PARALLEL_GATEWAY_END = "--END OF PARALLEL TASKS--"
GATEWAY_MARKER_PREFIX = "--"

# References
_DISC = (
    r"(?:referred\s+to\s+in"
    r"|as\s+referred\s+to\s+in"
    r"|in\s+accordance\s+with"
    r"|pursuant\s+to"
    r"|under"
    r"|according\s+to"
    r"|by\s+virtue\s+of"
    r"|in\s+line\s+with"
    r")"
)
_ART_TAIL = r"(?:\s*\(\s*\d+\s*\))?(?:\s*,\s*point\s+\(\s*[a-z]\s*\))?"
_ART_REF = r"Article\s+\d+" + _ART_TAIL
_ACT_CITE = (
    r"(?:Directive|Regulation|Decision|Implementing\s+Regulation|Delegated\s+Regulation)"
    r"\s*(?:\(\s*[A-Z]+\s*\)\s*)?\d+/\d+[\w/]*"
)

_RE_DISC_ART = re.compile(rf"\b{_DISC}\s+{_ART_REF}", re.IGNORECASE)
_RE_BARE_ART = re.compile(rf"\bArticle\s+(\d+){_ART_TAIL}", re.IGNORECASE)
_RE_ACT_CITE = re.compile(rf"\b(?:{_DISC}\s+)?{_ACT_CITE}", re.IGNORECASE)
_RE_ORPH_NUM = re.compile(r"(?<!\d)\s*\(\s*\d+\s*\)(?:\s*,\s*point\s+\(\s*[a-z]\s*\))?")
_RE_ORPH_DISC = re.compile(rf"\b{_DISC}\s*[,.]?\s*$", re.IGNORECASE | re.MULTILINE)
_REF_CONTEXT_MARKERS: FrozenSet[str] = frozenset({
    m.lower() for m in REFERENCE_MARKERS
} | {
    "iaw",
})

_DANGLING = re.compile(
    r"\s+\b(?:and|or|with|to|of|for|by|in|on|at|from|pursuant|under|as|via|per)\b\s*\.",
    re.IGNORECASE,
)

_ROMAN: dict = {
            "IS_UPPER": True,
            "TEXT": {"REGEX": r"^[IVXLCDM]{1,10}$"},
        }

_REF_TYPES: list[tuple[str, str]] = [
            ("article", "ARTICLE"),
            ("paragraph", "PARAGRAPH"),
            ("section", "SECTION"),
            ("annex", "ANNEX"),
            ("point", "POINT"),
            ("recital", "RECITAL"),
            ("schedule", "SCHEDULE"),
            ("appendix", "APPENDIX"),
        ]

# Patterns
_P1 = re.compile(
    r"(?:,?\s*)\b(?:that|which|who)\b\s+"
    r"(?:[^,;.]+(?:,(?!\s+[A-Z])[^,;.]*)*)",
    re.IGNORECASE,
)

_P2 = re.compile(
    r",?\s*\bas\s+(?:defined|referred\s+to|established|set\s+out)"
    r"\s+(?:in|by|under)\s+[^,;.]+",
    re.IGNORECASE,
)

_CONDITIONAL_FIRST = CONDITIONAL_SUBORDINATORS | {
    "unless", "except", "notwithstanding", "until", "before",
    "after", "once", "provided", "although", "though",
}
