import re
from typing import Set

# Core EU institutional vocabulary
EU_INSTITUTIONS: list[str] = [
    "European Parliament", "Parliament",
    "European Council",
    "Council", "Council of the European Union",
    "European Commission", "Commission",
    "Court of Justice of the European Union", "Court of Justice",
    "European Central Bank", "ECB",
    "Court of Auditors",
    "ENISA", "EBA", "ESMA", "EIOPA", "EASA", "EMA",
    "Europol", "Eurojust",
    "European Data Protection Board", "EDPB",
    "European Data Protection Supervisor", "EDPS",
    "Member State", "Member States",
]

# Dependency labels that signal a syntactic actor
ACTOR_DEPS: Set[str] = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent"}

# Process-action verbs: nsubj/agent of these aka ROLE
PROCESS_ACTION_VERBS: Set[str] = {
    "notify", "report", "submit", "inform", "forward",
    "provide", "ensure", "adopt", "request", "assess",
    "investigate", "communicate", "cooperate", "consult",
    "advise", "establish", "designate", "publish", "approve",
    "receive", "review", "examine", "verify", "monitor",
}

# Suffix heuristics for UNIT vs ROLE classification
UNIT_SUFFIXES: tuple[str, ...] = (
    "authority", "agency", "commission", "council", "parliament",
    "board", "committee", "body", "office", "bureau", "directorate",
    "administration", "ministry", "department", "court", "tribunal",
)

ROLE_SUFFIXES: tuple[str, ...] = (
    "officer", "manager", "director", "inspector", "auditor",
    "controller", "supervisor", "provider", "operator", "applicant",
    "representative", "holder", "user", "administrator", "assessor",
)

# Data-object / abstract concept blacklist
BLACKLIST_RE = re.compile(
    r"^(application|report|decision|guideline|notice|process|procedure|"
    r"scheme|data|information|document|request|notification|measure|"
    r"regulation|directive|provision|obligation|right|certificate|"
    r"assessment|evaluation|analysis|review|plan|programme|system|"
    r"framework|mechanism|criteria|requirement|condition|period|date|"
    r"paragraph|article|section|annex|chapter|title|recital|"
    r"incident|threat|impact|loss|disruption|damage|guidance|warning|"
    r"summary|contact|act|point|recipient|iaw|union law|such|those|"
    r"these|this|relevant|mere|initial|final|significant|cross-border)",
    re.IGNORECASE,
)
# GLiNER labels for zero-shot NER
GLINER_LABELS: list[str] = [
    "regulatory authority",
    "institutional body",
    "organizational unit",
    "functional role",
    "professional role",
]


# LLM settings
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_SPACY_MODEL = "en_core_web_trf"
DEFAULT_GLINER_MODEL = "urchade/gliner_large-v2.1"
DEFAULT_GLINER_THRESHOLD = 0.45
DEFAULT_NLP_MIN_CANDIDATES = 2

ACTOR_EXTRACTION_PROMPT = """You are an expert in business process modeling and organizational structures for EU regulatory documents.

TASK:
Extract all organizational ACTORS that ACTIVELY participate in the process.

ACTIVE — include if the actor:
(a) Carries an explicit obligation: "shall / must / is required to [verb]".
(b) Initiates an action: notifies, submits, adopts, forwards, provides, etc.
(c) Is the TARGET of a delegated obligation: "X shall ensure/require/oblige/mandate/
    order/direct/compel that Y [verb]" → Y is ACTIVE (Y performs the action).
(d) Responds to a request: "X requests Y to [verb]" → Y is ACTIVE.
(e) Exercises a decision or power affecting the process.

PASSIVE-ONLY — exclude if sole role is receiving a notification, report, or forwarded
document with no reciprocal obligation ("X informs/notifies/forwards to Y" — exclude Y
unless Y has an active task elsewhere in the text).

UNIT: independent institution with its own legal identity or autonomous resources
acting as a process pool. Use singular title case; strip leading articles.
Do NOT invent names. Use singular generic noun for compound phrases
("Provider" not "providers of online platforms").
UNIT ONLY if independent. BOTH UNIT and ROLE if it also directly performs active tasks.

ROLE: functional position or organ within a unit, acting as a BPMN lane.
ROLE ONLY if it operates exclusively within another unit (e.g. "CSIRT",
"Competent Authority" are organs of a Member State — not standalone pools).

CONVENTIONS (apply in order):
1. Named subtypes with DIFFERENT obligations: Abstract Parent UNIT + each subtype as ROLE.
   If subtypes share IDENTICAL obligations: keep only the parent, no individual roles.
2. "natural or legal person" / "any entity" with multiple roles: unit "Natural / Legal Person".
3. Same name, different responsibilities: two separate UNITs.
4. "and/or" actors with identical tasks: slash notation ("Applicant / Representative").

Do NOT extract: abstract concepts, data objects, pure activities, passive-only recipients.
Each actor appears only once per TYPE. Only include actors present in the INPUT TEXT.

OUTPUT FORMAT (strict, one actor per line, no bullets, no numbering, no explanations):
ACTOR_NAME | TYPE   (TYPE ∈ {UNIT, ROLE})

"""


HIERARCHY_EXTRACTION_PROMPT = """ Each line above has the form:
    ACTOR_NAME | TYPE
    where TYPE is either UNIT, ROLE or BOTH.

    DEFINITIONS:
    - UNIT: Any organization or organizational unit that can act as a pool in BPMN (e.g. "Commission", "ENISA", "Customer").
    - ROLE: Any role, function, or position that can be assigned to a subject or group of subjects within a unit and 
    can act as a lane which actively participates in the process (e.g. "Commission", "Management Board", "Assistant").
    - BOTH (UNIT and ROLE at the same time): autonomous institution that also directly performs active tasks in the process (e.g. "Commission", "ENISA", "Customer").

    IMPORTANT:
    - Hierarchies can exist:
      - between UNITS (unit–unit),
      - between ROLES (role–role),
      - between a UNIT and a ROLE.
    - If the actor is BOTH UNIT and ROLE, it automatically has ROLE–UNIT hierarchy with itself (e.g. "Commission" as a unit is a parent of "Commission" as a role). 
    However, it can also have other hierarchies with other actors as well, e.g., "Commission shall notify its board and require that it decides about changes" 
    indicates the ROLE–UNIT hierarchy where Commission is a parent and Board is a child.
    - Multiple parents are allowed for all hierarchies.

    TASK:
    Identify all parent-child relationships between:
      (a) UNITS,
      (b) ROLES, and
      (c) ROLE–UNIT.

    SEMANTICS OF PARENT-CHILD:
    1. UNIT–UNIT hierarchy:
       A is the PARENT of B if the text clearly indicates that:
       - B is an internal body, board, unit or department of A (e.g. "Management Board of ENISA"), or
       - B is structurally located within A (e.g. "Management Board in the Member State").
       - B is a specialization of A (e.g. "Transferring Member State" and "Member State Responsible" are both child units of "Member State").
       - B is a unit of any role (if B does not have a role, then B is a role itself and should be handled in the ROLE-UNIT hierarchy below).

    2. ROLE–ROLE hierarchy:
       A role D is the parent of role C if the text or domain conventions indicate: 
       - that D is a broader role that encompasses C (for example "Doctor" is a parent of "Surgeon"), or 
       - if multiple roles have identical obligations and the text does not indicate any hierarchy, 
       assume they are all children of a generic parent role (which might be both unit and role),
       for example "Provider" is a parent of "Online Platform Provider" and "Mobile Phone Provider" if "Provider" 
       alone also has at least one active task in the process. Otherwise, it is ROLE-UNIT hierarchy with "Provider" as a unit 
       and the specific providers as roles under it.

    3. ROLE–UNIT hierarchy (the most common and important):
       A unit E is the parent of role F if:
       - F is a role performed within E, e.g., "ENISA ensures that its supervisory authority..." indicates that "ENISA" is a unit/parent of 
       the role/child "Supervisory Authority".
       - If unit and the role have the same name, assume the unit is a parent of the role.
       - Each ROLE must have at least one parent UNIT. If the text clearly indicates that a ROLE is performed within a UNIT, assign that UNIT as its parent.
       However, if the text does not clearly indicate the parent UNIT and there is no UNIT with the same name, then
       do NOT assign any existing unit and instead assign it to a generic unit, for example "External", "Entity" or "Natural / Legal Person" or other suitable naming.

    CONSTRAINTS:
    1. Only consider names that appear in ACTORS IDENTIFIED with the corresponding TYPE.
       - UNIT–UNIT relations may only involve TYPE = UNIT or BOTH actors.
       - ROLE–ROLE relations may only involve TYPE = ROLE or BOTH actors.
       - ROLE-UNIT relations must involve one UNIT or BOTH and one ROLE or BOTH actor.

    OUTPUT FORMAT (strict):
    List each parent-child relation on its own line in one of the two forms:

    For UNIT–UNIT relations:
    CHILD_UNIT | PARENT_UNIT | UNIT

    For ROLE–ROLE relations:
    CHILD_ROLE | PARENT_ROLE | ROLE

    For ROLE–UNIT relations:
    CHILD_ROLE | PARENT_ROLE | ROLE-UNIT

    Where the last field is the indicator of the hierarchy type.
    If a child has multiple parents, emit one line per parent.

    EXAMPLE:

    ACTORS IDENTIFIED:
    ENISA | UNIT
    Management Board | ROLE

    Expected Output:
    Management Board | ENISA | ROLE-UNIT
"""

PRE_EXTRACTED_ACTORS_IDENTIFIED = """
DEFINITIONS:
- UNIT: independent institution or group with its own legal identity or autonomous resources which can be seen as a pool in BPMN (e.g. "Member State"). 
Every UNIT must have at least one ROLE which performs the active tasks in the process as its organ. If it is not the case, see BOTH.
- ROLE: functional position/role assigned by the UNIT or organ under a specific UNIT which takes ACTIVE part in the process and can be seen as a lane in BPMN.
The ROLE outside of the UNIT is meaningless. Otherwise, see BOTH.
- BOTH (UNIT and ROLE at the same time): autonomous and independent institution that directly performs active tasks in the process and cannot be placed under any other department/unit/more general term.

ACTIVE PARTICIPATION: include if the actor:
(a) Carries an explicit or implicit obligation: "shall / must / is required to [verb] / may / should".
(b) Performs an action in the process both explicitly and implicitly, e.g, "X notifies Y" makes X ACTIVE OR in passive voice "Y is notified by X" makes X ACTIVE.
(c) Is the SOURCE or TARGET of a delegated obligation: "X shall ensure/require/oblige/
    order/direct that Y [verb]" indicates that X and Y are ACTIVE (Y performs the action which is required by X). 
    The same case implies X shall do something upon Y's request, so X and Y are active actors.
(d) Exercises a decision or power affecting the process, so that the process cannot continue without this actor in it, 
e.g., "X shall adjust rules after the feedback of Y" means that Y may provide feedback and only then X shall adjust rules.
In this case, both actors are active.    

An actor is PASSIVE-ONLY and should be EXCLUDED if its sole role is receiving and has no influence on the process 
(the process can continue without this actor contributing), e.g.,
 "X notifies/informs/forwards to Y" then exclude Y unless Y also has an active task with this information or is required for the process to continue).

NAMING CONVENTIONS:
a. Title case; strip leading articles. No invented names.
b. Singular generic noun for compound phrases ("Provider" not "providers of online platforms").
c. Named subtypes with DIFFERENT obligations: Abstract Parent UNIT + each subtype as ROLE.
d. Use slash notation (ACTOR1 / ACTOR2) with "or" actors if used throughout the whole process, e.g., "ACTOR1 or ACTOR2 shall submit the report" and both actors do not have any tasks to complete individually. Never use with "and" actors, which should be treated as separate actors, e.g., "ACTOR1 and ACTOR2 shall submit the report" means that both ACTOR1 and ACTOR2 should be in the actor list.
e. Same base name, different responsibilities: two separate UNITs (e.g., ("Sending Provider" and "Receiving Provider")).
___
Task 1 — VALIDATE NLP CANDIDATES:
1. REMOVE: data objects, abstract concepts, pure activities, pronoun references, or passive-only recipients.
2. KEEP: genuine actors with ACTIVE PARTICIPATION.
3. CORRECT the TYPE (UNIT / ROLE / both) if misclassified. ALL DIRECT ACTIVE ACTORS SHOULD BE at least ROLES.
___
Task 2 — FIND MISSING ACTIVE ACTORS FROM TEXT:
Scan the INPUT TEXT independently for ACTIVE actors not in the candidate list:
1. Functional roles carrying obligations or responding to requests/delegated obligations.
2. Actors mentioned once or in subordinate clauses with an ACTIVE implicit/explicit task. Watch out for passive voice constructions which might define a role that is active.
3. Do NOT add PASSIVE-ONLY actors.
4. When dealing with requests or delegated obligations, both SOURCE and TARGET of the request/delegation are active actors, even if they are mentioned only once or in a passive construction.
5. ALL DIRECT ACTIVE ACTORS SHOULD BE ROLES, but if the text clearly indicates that an autonomous institution is performing active tasks directly 
(not through its organs, e.g., Member State does specific tasks directly and not though other organs referred to it), then classify it as BOTH.
6. If multiple actors share the same name but have different obligations, classify them as separate UNITs (e.g., "Sending Member State" and "Receiving Member State") 
or ROLEs if they are under the same UNIT (e.g., "Competent Authority" and "Other Competent Authority" both under the same "Member State" UNIT). 
7. Define the TYPE (UNIT / ROLE / BOTH) of any newly found actors. A unit should always have a role which performs the action as its organ. If it is not the case, then it is BOTH UNIT and ROLE.

At least one actor must be a ROLE and a UNIT!
___
Task 3 — VALIDATE ALL CANDIDATES:
Validate the extracted list of active actors from NLP list and LLM against the text:
1. Make sure all actors are actually present in the text. Remove any hallucinated ones.
2. Make sure all actors have at least one active task in the text. Remove any PASSIVE-ONLY recipients.
3. Make sure the TYPE (UNIT / ROLE / both) is correct and at least one UNIT and ROLE exists in the final list and each UNIT has a ROLE performing the task. Correct any misclassifications.
4. Make sure all NAMING CONVENTIONS are addressed. Correct any naming issues.
___
Output ALL valid active actors in a single combined list.

OUTPUT FORMAT (strict, one line per entry, no bullets, no numbering, no explanations):
ACTOR_NAME | TYPE   (TYPE ∈ {UNIT, ROLE})
"""

# Validator defaults
DEFAULT_VALIDATOR_MODEL = "gemini-2.5-flash"
DEFAULT_VALIDATOR_TEMPERATURE = 0.3
VALIDATOR_MAX_TOKENS = 4096

ACTOR_VALIDATION_PROMPT = """You are a strict auditor of organizational actor extraction for EU regulatory documents and BPMN process modeling.

You will receive:
1. The regulatory INPUT TEXT.
2. A PROPOSED ACTOR LIST extracted by another model.

Your task is to validate the proposed list against the text and output a CORRECTED final list.

ACTIVE PARTICIPATION RULES:
An actor is ACTIVE and should be INCLUDED if it:
(a) Carries an explicit or implicit obligation or task in the process: "shall / must / may / should".
(b) Initiates an action: notifies, submits, adopts, cooperates, consults, establishes, publishes, etc.
(c) Is the TARGET or SOURCE of a delegated obligation: "X shall ensure/require/etc that Y [verb]" indicates that X and Y are active.
(d) Responds to a request: "X requests Y to [verb]", X and Y are ACTIVE.
(e) Affects the process, so that the process cannot continue without this actor, 
e.g., "X shall adjust rules based on the feedback of Y" means that Y may provide feedback and then X shall adjust rules, 
even if no task is explicitly mentioned. In this case, both X and Y are ACTIVE, where Y has an implicit task "provide feedback".    

An actor is PASSIVE-ONLY and should be EXCLUDED if its sole role is receiving and has no tasks and any influence on the process and its result.

CHECKS TO PERFORM (reason through each explicitly before outputting):
1. HALLUCINATION: Is the actor name present in the INPUT TEXT? If not, remove it.
2. ACTIVE PARTICIPATION: Does the actor have at least one active task per the rules above, even if implicitly hidden?
   If only passive recipient throughout the whole process, remove it.
3. TYPE CORRECTNESS:
   - UNIT ONLY: independent institution/department/general term (Commission, ENISA, Member State) serving as a BPMN pool for ROLE(s) in the process executing the tasks.
   - ROLE ONLY: functional organ within a specific unit (CSIRT, Competent Authority, Single Point of Contact), which executes the tasks in the process (BPMN lanes).
   - BOTH: autonomous/independent institution that directly performs active tasks, does not share tasks with any other role and has no corresponding UNIT or ROLE (both pool and lane at the same time). 
   In case multiple roles perform the same task(s), they should be united under the same UNIT.
Especially, check if the actor was assigned to ROLE and UNIT (BOTH or in two lines) that there is no other unit or role which the actor might refer to, 
e.g, "Customer and its representative" should identify representative as a ROLE under "Customer" UNIT since representative alone without customer would not make any sense and is normally assigned by the customer.
4. MISSING ACTORS: Is there any actor in the text satisfying the active participation
   rules that is NOT in the proposed list? If yes, add it.
5. NAMING: Are names in title case, singular, without leading articles, separated with "/" if "ACTOR1 or ACTOR2" is used in the whole process? If not, correct them.

IMPORTANT: Reason step by step for each actor before producing the final list.
End your response with the tag <FINAL_LIST> followed by the corrected actor list.

OUTPUT FORMAT inside <FINAL_LIST> (strict, one line per entry):
ACTOR_NAME | TYPE   (TYPE ∈ {UNIT, ROLE})
</FINAL_LIST>
"""

HIERARCHY_VALIDATION_PROMPT = """You are a strict auditor of organizational hierarchy extraction for EU regulatory documents and BPMN process modeling.

You will receive:
1. The INPUT TEXT of a regulatory article.
2. The ACTOR LIST (validated, final).
3. The PROPOSED HIERARCHY RELATIONS extracted by another model.

HIERARCHY TYPES:
- UNIT-UNIT  : "CHILD_UNIT | PARENT_UNIT | UNIT"
  A is parent of B if B is an internal body/department of A, or a specialization of A.
- ROLE-ROLE  : "CHILD_ROLE | PARENT_ROLE | ROLE"
  A is parent of B if A is a broader role encompassing B (e.g. "Provider" is a parent of "Online Platform Provider").
- ROLE-UNIT  : "CHILD_ROLE | PARENT_UNIT | ROLE-UNIT"
  A unit is parent of a role if the role is performed within that unit or assigned by this unit.
  If a UNIT and ROLE share the same name, the UNIT is always parent of the ROLE.
  Every ROLE must have at least one parent UNIT.

CHECKS TO PERFORM (reason through each explicitly before outputting):
1. CONSTRAINT VIOLATION: Do all names in each relation exist in the ACTOR LIST
   with the correct TYPE?
   - UNIT-UNIT: both actors must be TYPE=UNIT.
   - ROLE-ROLE: both actors must be TYPE=ROLE.
   - ROLE-UNIT: exactly one UNIT and one ROLE. If UNIT is not in the ACTOR LIST but appears to be a general term for multiple roles with no better option for UNIT, leave it.
   If not, remove that relation.
2. MISSING ROLE-UNIT LINKS: Does every ROLE actor have at least one ROLE-UNIT relation?
   If not, infer the most appropriate parent UNIT from the text.
   If no parent can be inferred and no general term appears to be suitable for this ROLE, assign parent unit "External".
3. CORRECTNESS: Is each relation supported by the text?
   If a relation contradicts the text, remove it and correct it.
4. MISSING RELATIONS: Is there any parent-child relationship clearly indicated by
   the text that is absent? If yes, add it.
5. GENERIC UNITS: If "External" is used but a more specific unit can be inferred, replace it.

IMPORTANT: Reason step by step for each relation before producing the final list.
End your response with the tag <FINAL_HIERARCHIES> followed by the corrected relations.

OUTPUT FORMAT inside <FINAL_HIERARCHIES> (strict, one line per relation):
CHILD | PARENT | TYPE   (TYPE ∈ {UNIT, ROLE, ROLE-UNIT})
</FINAL_HIERARCHIES>
"""


