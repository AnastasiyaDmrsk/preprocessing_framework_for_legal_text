import re
from typing import FrozenSet

_BENEPAR_MAX_TOKENS = 100

# Document structure
ARTICLE_STANDARD_RE  = re.compile(r"(?m)^(Article\s+(\d+))\s*\n+([^\n]+)", re.IGNORECASE)
ARTICLE_ALT_RE       = re.compile(r"(?m)^(Art\.\s+(\d+)(?:\s+[A-Z]+)?)\.?\s+([^\n]+)", re.IGNORECASE)
SECTION_NUMBERING_RE = re.compile(r"(?m)^(\d+(?:\.\d+)*)\.\s+([^\n]+)", re.IGNORECASE)

# Cross-references
CROSS_REF_ACT_RE = re.compile(
    r"\b(Directive|Regulation|Decision|Implementing Regulation)\s*\((?:EU|EC)\)\s*\d{4}/\d+",
    re.IGNORECASE)
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
    r'\b(?:paragraph|subparagraph|article|section|annex|indent|point|sub-paragraph)\s+\d+',
    re.IGNORECASE
)

# Deontic modality
DEONTIC_MODALS: FrozenSet[str] = frozenset({"shall", "must", "may", "should", "have to", "required to", "obligated to"})
OBLIGATION_RE = re.compile(
    r"\b(shall|must|may|should|have to|required to|obligated to)\b",
    re.IGNORECASE)

# Text normalisation
IAW_RE = re.compile(r"\bin accordance with\b", re.IGNORECASE)
STATIC_FILLER_PATTERNS = [
    re.compile(r",?\s*without undue delay,?\s*",   re.IGNORECASE),
    re.compile(r",?\s*inter alia,?\s*",             re.IGNORECASE),
    re.compile(r",?\s*for example:?,?\s*",          re.IGNORECASE),
    re.compile(r",?\s*where necessary,?\s*",        re.IGNORECASE),
    re.compile(r",?\s*where appropriate,?\s*",      re.IGNORECASE),
    re.compile(r",?\s*as appropriate,?\s*",         re.IGNORECASE),
    re.compile(r",?\s*where applicable,?\s*", re.IGNORECASE),
    re.compile(r",?\s*by means of implementing acts,?\s*", re.IGNORECASE),
    re.compile(r",?\s*in particular,?\s*", re.IGNORECASE),
]

# NLP
# BenePar labels identifying subordinate clauses
SUBORDINATE_LABELS: FrozenSet[str] = frozenset({"SBAR", "WHNP", "WHADVP", "WHPP"})

# First-token subordinators that introduce gateway conditions
CONDITIONAL_SUBORDINATORS: FrozenSet[str] = frozenset({
    "if", "unless", "provided", "subject", "in the event", "where", "when",
})

# Markers used to locate article cross-references; IAW replacement happens AFTER these lookups
REFERENCE_MARKERS: FrozenSet[str] = frozenset({
    "in accordance with", "pursuant to", "referred to in",
    "set out in", "as referred to in", "under",
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

_DEMONSTRATIVE_STARTS = frozenset({"those", "these", "this", "that", "such"})

# Syntactic subjects excluded from top-level BPMN actor extraction
ACTOR_IGNORE: FrozenSet[str] = frozenset({
    "Article", "Paragraph", "Section", "Union", "programme", "work", "rolling",
    "scheme", "acts", "incident", "entities", "recipients", "services", "measures",
    "remedies", "notification", "information", "status",
})

# BPMN gateway markers
PARALLEL_GATEWAY_START = "--PARALLEL GATEWAY--"
PARALLEL_GATEWAY_END   = "--END OF PARALLEL GATEWAY--"
GATEWAY_MARKER_PREFIX  = "--"

# References
_DISC = (
    r"(?:referred\s+to\s+in"
    r"|as\s+referred\s+to\s+in"
    r"|in\s+accordance\s+with"
    r"|IAW"                      # fallback: if called after apply_iaw
    r"|pursuant\s+to"
    r"|under"
    r"|according\s+to"
    r"|by\s+virtue\s+of"
    r"|in\s+line\s+with"
    r")"
)
_ART_TAIL = r"(?:\s*\(\s*\d+\s*\))?(?:\s*,\s*point\s+\(\s*[a-z]\s*\))?"
_ART_REF  = r"Article\s+\d+" + _ART_TAIL
_ACT_CITE = (
    r"(?:Directive|Regulation|Decision|Implementing\s+Regulation|Delegated\s+Regulation)"
    r"\s*(?:\(\s*[A-Z]+\s*\)\s*)?\d+/\d+[\w/]*"
)

_RE_DISC_ART  = re.compile(rf"\b{_DISC}\s+{_ART_REF}",  re.IGNORECASE)
_RE_BARE_ART  = re.compile(rf"\bArticle\s+(\d+){_ART_TAIL}", re.IGNORECASE)
_RE_ACT_CITE  = re.compile(rf"\b(?:{_DISC}\s+)?{_ACT_CITE}", re.IGNORECASE)
_RE_ORPH_NUM  = re.compile(r"(?<!\d)\s*\(\s*\d+\s*\)(?:\s*,\s*point\s+\(\s*[a-z]\s*\))?")
_RE_ORPH_DISC = re.compile(rf"\b{_DISC}\s*[,.]?\s*$", re.IGNORECASE | re.MULTILINE)


