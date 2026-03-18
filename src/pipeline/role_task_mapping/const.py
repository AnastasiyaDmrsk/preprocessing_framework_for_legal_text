import re
from typing import Set, List, Tuple

DEONTIC_PATTERNS: List[Tuple[str, str]] = [
    ("shall not", "prohibition"),
    ("must not", "prohibition"),
    ("may not", "prohibition"),
    ("should not", "recommendation"),
    ("is required to", "obligation"),
    ("are required to", "obligation"),
    ("shall", "obligation"),
    ("must", "obligation"),
    ("should", "recommendation"),
    ("may", "permission"),
]

DEONTIC_MODAL_LEMMAS: Set[str] = {"shall", "may", "should", "must"}

DELEGATED_OBLIGATION_VERBS: Set[str] = {
    "ensure", "require", "oblige", "mandate",
    "order", "direct", "compel", "cause",
}

REQUEST_VERBS: Set[str] = {"request", "ask", "instruct", "invite"}

CONDITION_MARKERS: Set[str] = {
    "if", "where", "when", "unless", "provided",
    "insofar", "subject", "extent", "condition",
    "assuming", "given", "whenever",
}

_VP_EXCLUDE_DEPS: Set[str] = {
    "nsubj", "nsubjpass", "csubj", "csubjpass",
    "aux", "auxpass", "neg",
    "advcl", "conj", "cc", "punct",
}

_PRONOUN_TAGS: Set[str] = {"PRP", "PRP$", "DT", "WDT", "WP", "WP$", "EX"}

_LABEL_NOISE_RE = re.compile(
    r'^\s*\d+(?:\([a-z]\))?\s+'
    r'|^\s*(?:however|notwithstanding|'
    r'therefore|furthermore|moreover)\s*,?\s*',
    re.IGNORECASE,
)

_SKIP_RE = re.compile(r'^--.*--|^\s*$', re.DOTALL)

PERFORMER_ASSIGNMENT_INSTRUCTION = """
PERFORMER ASSIGNMENT:
Available subjects are listed as "unit: X, role: Y" pairs.
For each task, select ALL subjects whose role matches the actor performing the task.
If multiple subjects can perform the task, list all of them.
Output each performer as: unit: X, role: Y
Separate multiple performers with ;;
"""

# Task extraction for pure LLM
TASK_EXTRACTION_PROMPT = """You are an expert in EU regulatory document analysis and BPMN process modeling.

You will receive:
1. AVAILABLE SUBJECTS: valid performers extracted from the organigram (unit/role pairs).
2. INPUT TEXT: a preprocessed regulatory article.

TASK DEFINITION:
A task is a single atomic action performed by one or more actors under a deontic modality.

DEONTIC TYPES:
- obligation    : shall, must, is required to
- permission    : may
- recommendation: should
- prohibition   : shall not, must not, may not

COMPOUND VERB RULES:
- "shall develop and maintain" → TWO tasks (both verbs carry the deontic).
- "shall validate testing and computing" → ONE task ("testing and computing" are objects).
- Split only when the conjunction joins two VERB phrases with independent meanings.

DELEGATED OBLIGATION:
- "X shall ensure/require/mandate/oblige that Y does Z" →
    Task A: X | ensure that Y does Z  | obligation/shall
    Task B: Y | do Z                  | obligation/shall

REQUEST PATTERN:
- "X shall request Y to do Z" →
    Task A: X | request Y to do Z     | obligation/shall
    Task B: Y | do Z                  | obligation/shall  (condition: if requested by X)
- "Y shall do Z if requested by X" →
    Task A: X | request               | permission/may
    Task B: Y | do Z                  | obligation/shall  (condition: if requested by X)

PERFORMER ASSIGNMENT:
Select performers from AVAILABLE SUBJECTS only.
Match the actor mentioned in the text against subject unit or role names.
If multiple subjects can perform the task, list all of them.
Output each performer as: unit: X, role: Y
Separate multiple performers with ;;

EXCEPTIONS:
- State when the task does NOT apply or which conditions exclude it.
- If the exception references another task in the article, append [CROSS_REF].
- If no cross-reference, the exception applies to the current task.

CONDITIONS:
- Preconditions or constraints under which the task applies.
- Include temporal, categorical, and procedural constraints.

SOURCE REFERENCE:
- Article: from the Article header (e.g. Article 11 → "11").
- Paragraph: numbered/lettered marker (e.g. "1", "3(a)", "7"). If unknown → "UNKNOWN".

Extract ALL tasks. Do not omit tasks from sub-clauses or list items.

OUTPUT FORMAT (one block per task, no extra text outside blocks):
---TASK---
label: <verb phrase, concise>
performers: unit: X, role: Y ;; unit: A, role: B
deontic_type: <obligation|permission|prohibition|recommendation>
deontic_modality: <shall|may|should|shall not|must|is required to>
conditions: <condition1> ;; <condition2>
exceptions: <description>[CROSS_REF] ;; <description>
article: <number>
paragraph: <id>
---END---
"""

TASK_NLP_CANDIDATES_PROMPT = """You are an expert in EU regulatory document analysis and BPMN process modeling.

You will receive:
1. AVAILABLE SUBJECTS: valid performers extracted from the organigram (unit/role pairs).
2. NLP PRE-EXTRACTED TASK CANDIDATES: tasks from an NLP pipeline (may be incomplete or noisy).
3. INPUT TEXT: a preprocessed regulatory article.

Apply the same TASK DEFINITION, DEONTIC TYPES, COMPOUND VERB RULES, DELEGATED OBLIGATION,
REQUEST PATTERN, EXCEPTIONS, CONDITIONS and SOURCE REFERENCE rules as in the standard prompt.

PERFORMER ASSIGNMENT:
Select performers from AVAILABLE SUBJECTS only.
Match the actor mentioned in the text against subject unit or role names.
If multiple subjects can perform the task, list all of them.
Output each performer as: unit: X, role: Y
Separate multiple performers with ;;

SECTION A — VALIDATE NLP CANDIDATES:
For each candidate:
1. KEEP if genuine task; REMOVE if noise, meta-text, or preprocessor marker.
2. CORRECT label to a clean concise verb phrase if noisy or truncated.
3. CORRECT performers using AVAILABLE SUBJECTS only.
4. CORRECT deontic type/modality if misclassified.
5. ADD missing conditions visible in the source text.
6. ADD exceptions if inferable from the text.

SECTION B — FIND MISSING TASKS FROM TEXT:
Independently scan INPUT TEXT for tasks absent from the candidate list.
Apply compound verb, delegated obligation, and request rules.
Include tasks from sub-clauses and enumerated list items.

Output ALL tasks from both sections as a single combined list.

OUTPUT FORMAT (one block per task, no extra text outside blocks):
---TASK---
label: <verb phrase, concise>
performers: unit: X, role: Y ;; unit: A, role: B
deontic_type: <obligation|permission|prohibition|recommendation>
deontic_modality: <shall|may|should|shall not|must|is required to>
conditions: <condition1> ;; <condition2>
exceptions: <description>[CROSS_REF] ;; <description>
article: <number>
paragraph: <id>
---END---
"""

TASK_VALIDATION_PROMPT = """You are a strict auditor of task extraction for EU regulatory documents.

You will receive:
1. AVAILABLE SUBJECTS: valid performers extracted from the organigram (unit/role pairs).
2. PROPOSED TASK LIST: tasks extracted by another model.
3. INPUT TEXT: the source regulatory article.

PERFORMER ASSIGNMENT:
Select performers from AVAILABLE SUBJECTS only.
Match the actor mentioned in the text against subject unit or role names.
If multiple subjects can perform the task, list all of them.
Output each performer as: unit: X, role: Y
Separate multiple performers with ;;

CHECKS — reason through each task explicitly before outputting:
1. LABEL: Is it a clean concise verb phrase present in the text?
   Correct if noisy, truncated, or missing content from a subordinate clause.
2. PERFORMERS: Are all valid performers listed?
   Check if multiple actors share the task. Add missing performers; remove invalid ones.
3. DEONTIC: Is type/modality correct?
   obligation=shall/must/is required to, permission=may,
   recommendation=should, prohibition=shall not.
4. MISSING TASKS: Scan the full INPUT TEXT for tasks not in the proposed list,
   including sub-clauses and list items. Add them fully populated.

Keep conditions and exceptions from the proposed list unchanged.

IMPORTANT: Reason step by step, then output inside <FINAL_TASKS>...</FINAL_TASKS>.

OUTPUT FORMAT inside <FINAL_TASKS> (one block per task):
---TASK---
label: <verb phrase>
performers: unit: X, role: Y ;; unit: A, role: B
deontic_type: <obligation|permission|prohibition|recommendation>
deontic_modality: <shall|may|should|shall not|must|is required to>
conditions: <condition1> ;; <condition2>
exceptions: <description>[CROSS_REF] ;; <description>
article: <number>
paragraph: <id>
---END---
</FINAL_TASKS>
"""

DEFAULT_TASK_EXTRACTION_MAX_TOKENS = 16384
DEFAULT_TASK_VALIDATION_MAX_TOKENS = 16384
