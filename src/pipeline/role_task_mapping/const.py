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

TASK DEFINITION:
A task is a single atomic action performed by one or more actors under a deontic modality. Example: "Member State shall report to Commission" is a task with label "Report to Commission", performer unit: Member State, role: Member State, deontic type: obligation, deontic modality: shall.

DEONTIC TYPES:
- obligation    : shall, must, is required to
- permission    : may
- recommendation: should
- prohibition   : shall not, must not, may not

COMPOUND VERB RULES:
- "shall develop and maintain": TWO tasks (both verbs carry the deontic).
- "shall validate testing and computing": ONE task ("testing and computing" are objects).
- Split only when the conjunction joins two VERB phrases with independent meanings.

DELEGATED OBLIGATION:
- "X shall ensure/require/mandate/oblige that Y does Z":
    Task A: X | ensure that Y does Z  | obligation/shall
    Task B: Y | do Z                  | obligation/shall

REQUEST PATTERN:
- "X shall request Y to do Z":
    Task A: X | request Y to do Z     | obligation/shall
    Task B: Y | do Z                  | obligation/shall  (condition: if requested by X)
- "Y shall do Z by request of X" →
    Task A: X | request               | permission/may
    Task B: Y | do Z                  | obligation/shall  (condition: if requested by X)

EXCEPTIONS:
- State when the task does NOT apply or which conditions exclude it.
- If the exception references another task in the article, append [CROSS_REF].
- If no cross-reference, the exception applies to the current task.

CONDITIONS:
- Preconditions or constraints under which the task applies.

SOURCE REFERENCE:
- Article: from the Article header (e.g. Article 11: "11").
- Paragraph: numbered/lettered marker (e.g. "1", "3(a)", "7"). If unknown, set "UNKNOWN".

PERFORMER ASSIGNMENT:
- Select performers from AVAILABLE SUBJECTS only. If a performer is not explicitly in the AVAILABLE SUBJECTS, you must deduce the most logical actor from the list based on context. Only use 'UNKNOWN' if absolutely no logical inference can be made.
- Match the actor mentioned in the text against subject unit or role names.
- If multiple subjects can perform the task, list all of them. For example "Member State and Commission shall comply with X" results into performers: unit: Member State, role: Member State ;; unit: Commission, role: Commission
- Output each performer as: unit: X, role: Y
- Separate multiple performers with ;;

SECTION A — VALIDATE NLP CANDIDATES:
For each candidate:
1. KEEP if genuine task; REMOVE if noise, meta-text, or preprocessor marker.
2. CORRECT label to a clean concise verb phrase if noisy or truncated. The label should be as short as possible but contain all necessary information.
3. CORRECT performers using AVAILABLE SUBJECTS only.
4. CORRECT deontic type/modality if misclassified.
5. ADD missing conditions visible in the source text. In case no conditions are explicitly mentioned, leave empty.
6. ADD exceptions if inferable from the text.

SECTION B — FIND MISSING TASKS FROM TEXT:
Independently scan INPUT TEXT for explicit and implicit tasks absent from the candidate list: 
- Each line has most probably at least one task. 
- Explicit tasks are directly stated with a deontic verb, e.g., "Commission shall improve the acts" has a label "Improve the acts" and performers unit = Commission, role = Commission. 
- Implicit tasks may arise from compound verb structures, delegated obligations, or requests, e.g., "Member State shall ensure that the authority does X" implies two tasks: "Ensure that the authority does X" (performed by Member State) and "Do X" (performed by the authority).
Another implicit example is "If requested by Member State, the authority shall do X" which implies "Request the authority to do X" (performed by Member State) and "Do X" (performed by the authority, condition: if requested by Member State).
- Exclude receiver tasks since they have no influence on the process and would not be an activity in the BPMN model, for example "The report shall be received by Commission", where "Receive report" should NOT be classified as a task (in BPMN would be a message event).
- Apply compound verb, delegated obligation, and request rules. Check for hidden tasks in passive voice or in from sub-clauses and enumerated list items.

The label of the task should be as short as possible and only contain the most important information for the BPMN activity. 

If a task has no conditions (should be executed independently of any specific circumstances), leave the conditions field empty (conditions:). If no exceptions are mentioned, leave the exceptions field empty (exceptions:).

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

TASK_VALIDATION_PROMPT = """You are a strict auditor of task extraction for EU regulatory documents, focusing on all tasks required in the BPMN modeling.

You will receive:
1. AVAILABLE SUBJECTS: valid performers extracted from the organigram (unit/role pairs, where role directly executes the tasks).
2. PROPOSED TASK LIST: tasks extracted by another model.
3. INPUT TEXT: the source regulatory article.

TASK DEFINITION:
A task is a single atomic action performed by one or more actors under a deontic modality. Example: "Member State shall report to Commission" is a task with label "Report to Commission", performer unit: Member State, role: Member State, deontic type: obligation, deontic modality: shall.

DEONTIC TYPES:
- obligation    : shall, must, is required to
- permission    : may
- recommendation: should
- prohibition   : shall not, must not, may not

COMPOUND VERB RULES:
- "shall develop and maintain": TWO tasks (both verbs carry the deontic).
- "shall validate testing and computing": ONE task ("testing and computing" are objects).
- Split only when the conjunction joins two VERB phrases with independent meanings.

DELEGATED OBLIGATION:
- "X shall ensure/require/mandate/oblige that Y does Z":
    Task A: X | ensure that Y does Z  | obligation/shall
    Task B: Y | do Z                  | obligation/shall

REQUEST PATTERN:
- "X shall request Y to do Z":
    Task A: X | request Y to do Z     | obligation/shall
    Task B: Y | do Z                  | obligation/shall  (condition: if requested by X)
- "Y shall do Z by request of X" →
    Task A: X | request               | permission/may
    Task B: Y | do Z                  | obligation/shall  (condition: if requested by X)

PERFORMER ASSIGNMENT:
- Select performers from AVAILABLE SUBJECTS only. If a performer is not explicitly in the AVAILABLE SUBJECTS, you must deduce the most logical actor from the list based on context. Only use 'UNKNOWN' if absolutely no logical inference can be made.
- Match the actor mentioned in the text against subject unit or role names.
- If multiple subjects can perform the task, list all of them. For example "Member State and Commission shall comply with X" results into performers: unit: Member State, role: Member State ;; unit: Commission, role: Commission
- Output each performer as: unit: X, role: Y
- Separate multiple performers with ;; For example "Member State and Commission shall comply with X" results into performers: unit: Member State, role: Member State ;; unit: Commission, role: Commission

CHECKS: reason through each task explicitly before outputting:
1. LABEL: Is it a clean concise verb phrase present in the text?
   Correct if noisy, truncated, or missing content from a subordinate clause, but keep as short as possible with only necessary information (as a label for the BPMN activity). For example from "Member State shall ensure that the authority does X" the label should be "Ensure that the authority does X" and not just "Ensure", together with an additional task "Do X" performed by the authority.
2. PERFORMERS: Is the task performed by performers? Are all valid performers listed?
   Check if multiple actors share the task. Add missing performers; remove invalid ones. Each task MUST have at least one performer from the AVAILABLE SUBJECTS. If the text mentions an actor that is not in the AVAILABLE SUBJECTS, deduce the most logical performer from the list based on context. Only assign "UNKNOWN" if absolutely no logical inference can be made.
3. DEONTIC: Is type/modality correct?
   obligation=shall/must/is required to, permission=may,
   recommendation=should, prohibition=shall not.
4. MISSING TASKS: Scan the full INPUT TEXT for tasks not in the proposed list,
   including sub-clauses, passive voice, requests, implicit tasks and list items. Add them fully populated. Make sure that each subject is assigned at least one task. 
   For example "Commission shall coordinate by request" should result in two tasks: "Coordinate" (performed by Commission, obligation/shall, condition: if requested) and "Request coordination" or "Request Commission to coordinate" (performed by the requester based on the text, permission/may, condition: if needed).
   Include only the tasks executed by the actor and not data objects. For example "The information shall include the address" should NOT be classified as a task, since information is not an actor but an object.
5. CHECK TASK PRESENCE: For each task, check if it is explicitly or implicitly supported by the text. 
If you find a task that cannot be justified by the text DIRECTLY, remove it. 
Always check if the task goes both ways, e.g., "After consultation with the authority, Member State shall..." has "Consult Member State" and "Consult the authority" since consultation goes both ways. On the other hand, "By request of Member State" has a one way task "Request".
6. CONDITIONS AND EXCEPTIONS CHECK: check if all conditions and exceptions were identified. If a task has no conditions (should be executed independently of any specific circumstances), 
leave the conditions field empty (conditions:). If no exceptions are mentioned, leave the exceptions field empty (exceptions:). The conditions and exceptions should also be as short as possible but contain all necessary information.
7. NO DUPLICATES: make sure there are no duplicate tasks with the same label and performers, except when the text mentions explicitly that the task repeats in the process multiple times. Sometimes, regulatory text may be non-sequent and mention the same task in different contexts, but if the task is essentially the same and performed by the same actors, it should not be duplicated in the output.
8. SINGLE TASKS ONLY: always separate multiple tasks (indicator: same deontic verb for multiple verbs), for example "Member State shall analyse, interpret and apply the criteria" should be split into tasks: analyse the criteria, interpret the criteria, apply the criteria.

IMPORTANT: Reason step by step, then output inside <FINAL_TASKS>...</FINAL_TASKS>. All the tasks should make sense as in the process model.

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
