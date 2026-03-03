import re
from pathlib import Path
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

# ── Dependency labels that signal a syntactic actor ──────────────────────────
ACTOR_DEPS: Set[str] = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent"}

# ── Process-action verbs: nsubj/agent of these → strong ROLE signal ──────────
PROCESS_ACTION_VERBS: Set[str] = {
    "notify", "report", "submit", "inform", "forward",
    "provide", "ensure", "adopt", "request", "assess",
    "investigate", "communicate", "cooperate", "consult",
    "advise", "establish", "designate", "publish", "approve",
    "receive", "review", "examine", "verify", "monitor",
}

# ── Suffix heuristics for UNIT vs ROLE classification ───────────────────────
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

# ── Data-object / abstract concept blacklist ─────────────────────────────────
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
# ── GLiNER labels for zero-shot NER ─────────────────────────────────────────
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
Extract all organizational ACTORS from the regulatory text below.
Each actor produces one or more output lines (UNIT and/or ROLE entries).

UNIT DEFINITION:
A UNIT is any institution, agency, authority, body, or category of participant that:
- Acts as a participant in the process (pool), OR
- Has responsibilities, powers, or obligations in the text.
Naming rules:
- Use title case, exactly as the body appears in the text.
- Do NOT invent unit names not present in the text.
- Use singular generic noun when text uses compound phrase: for example for "providers of online platforms" use unit: "Provider"

ROLE DEFINITION:
A ROLE is any function, position, or responsibility that can be assigned to a subject or group of subjects 
within a unit (lane). So everything that has tasks in the process is a ROLE. If the pool has no lanes, the pool and the role are the same.
Additionally, apply these conventions (1 to 9) in order:

CONVENTION 1: If the text explicitly lists named subtypes of the same concept: Abstract parent UNIT + each subtype as ROLE.
- Example 1: "qualified / non-qualified trust service providers" should be a unit: "trust service provider"; roles: "qualified trust service provider", "non-qualified trust service provider"

CONVENTION 2: If text explicitly uses "natural or legal person" as a unit AND multiple participant roles with no other unit can relate to it:
- Create a unit: "Natural / Legal Person"; assign those roles to it.
- Similarly with "any entity" or "any person" or other general units if multiple roles relate to it without a more specific unit.

CONVENTION 3: If the text distinguishes TWO actors with the same name, e.g., "Member State", in the SAME process by assigning them different action verbs or responsibilities:
- Create TWO separate units instead of one generic
- Example: "Transferring Member State" and "Member State Responsible" units instead of "Member State" if different Member States perform different actions in the same process.

CONVENTION 4: If the text consistently groups two actors with "and/or" as a single decision point (e.g. "applicant and/or his or her representative"):
- Treat as one combined unit/role using slash notation:
- Example: create unit: "Applicant / Representative", role: "Applicant / Representative"

RULES:
1. Extract the exact surface form without leading articles:
   - "Commission" not "The Commission"
2. Include non-human actors if treated as resources:
   - e.g. "System"
5. Do NOT extract:
   - Abstract concepts (e.g. "programme", "scheme", "notice", "process")
   - Data objects (e.g. "report", "application", "guidelines", "database")
   - Pure activities (verbs or verb phrases without an actor)
6. Each ACTOR must appear only once per TYPE.
7. Only include actors that actually appear in the INPUT TEXT.

OUTPUT FORMAT (strict with one actor per line, no bullets, no numbering, no explanations):
ACTOR_NAME | TYPE   (TYPE ∈ {UNIT, ROLE})

"""
HIERARCHY_EXTRACTION_PROMPT = """ Each line above has the form:
    ACTOR_NAME | TYPE
    where TYPE is either UNIT or ROLE.

    RESOURCE MODEL:
    - UNIT: Any organization or organizational unit that can act as a pool (e.g. "Commission", "ENISA").
    - ROLE: Any role, function, or position that can be assigned to a subject or group of subjects within a unit and can act as a lane
      (e.g. "Commission", "Management Board", "Assistant").

    IMPORTANT:
    - An ACTOR can appear both as UNIT and as ROLE (e.g. "Commission", "ENISA", "Customer").
    - Hierarchies can exist:
      - between UNITS (unit–unit),
      - between ROLES (role–role),
      - between a UNIT and a ROLE.

    TASK:
    Identify all parent-child relationships between:
      (a) UNITS,
      (b) ROLES, and
      (c) UNIT–ROLE.

    Multiple parents are allowed for all hierarchies.

    SEMANTICS OF PARENT-CHILD:
    1. UNIT–UNIT hierarchy:
       A is the PARENT of B if the text clearly indicates that:
       - B is an internal body, board, unit or department of A (e.g. "Management Board of ENISA"), or
       - B is structurally located within A (e.g. "Management Board in the Member State").
       - B is a specialization of A (e.g. "Transferring Member State" and "Member State Responsible" are both child units of "Member State").
       - B is a unit of any role (if B does not have a role, then B is a role itself and should be handled in the ROLE-UNIT hierarchy below).

    2. ROLE–ROLE hierarchy:
       A role D is the parent of role C if the text or domain conventions indicate 
       that D is a broader role that encompasses C, for example "Doctor" is a parent of "Surgeon"

    3. ROLE–UNIT hierarchy:
       A unit E is the parent of role F if:
       - F is a role performed within E, for example "ENISA takes the role of a supervisory authority" indicates that "ENISA" is a unit of 
       the role "supervisory authority".
       - If unit and the role have the same name, assume the unit is a parent of the role, for example "Commission" as a unit is a parent of "Commission" as a role.
       - If role is located in the unit but has active tasks in the process, assume the unit is a parent of the role, for example "ENISA" as a unit is a parent of "Management Board" as a role.
       However, if the text does not clearly indicate the parent unit for a role, 
       do NOT assign an existing unit and instead assign it to a generic unit, for example "External", "Entity" or "Natural / Legal Person".
       In the end, all ROLE actors must have at least one parent UNIT, even if it's a generic one, and all UNIT actors that have a ROLE with the same name should be parents of that ROLE.

    CONSTRAINTS:
    1. Only consider names that appear in ACTORS IDENTIFIED with the corresponding TYPE.
       - UNIT–UNIT relations may only involve TYPE = UNIT actors.
       - ROLE–ROLE relations may only involve TYPE = ROLE actors.
       - ROLE-UNIT relations must involve one UNIT and one ROLE actor.

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
- UNIT ONLY if the actor is an independent institution with its own legal identity or
autonomous resources (e.g. "Commission", "ENISA", "Member State").
- ROLE ONLY if the actor is a functional position or organ operating exclusively
within another unit (e.g. "CSIRT", "Competent Authority", "Single Point of Contact"
are organs of Member States: they are ROLE only, not standalone pools).
- BOTH UNIT and ROLE only if the actor is simultaneously an autonomous institution
AND directly performs process tasks (e.g. "Commission", "ENISA").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 1: VALIDATE NLP CANDIDATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each candidate listed above:
1. REMOVE if it is a data object, abstract concept, pure activity, or pronoun reference.
2. CORRECT the TYPE (UNIT / ROLE / both) if misclassified.
3. KEEP if it is a genuine actor with responsibilities or obligations in the process.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 2: FIND MISSING ACTORS FROM TEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Independently scan the INPUT TEXT for actors NOT already in the candidate list BUT who participate in the process. Focus on:
1. Named functional roles ACTIVELY performing tasks/actions (e.g. "Trust Service Provider").
2. Actors mentioned only once or in subordinate clauses that NLP may have missed.
3. Actors referenced as recipients of obligations or guidance (PASSIVE participation).
4. Actors mentioned only by pronoun or short reference earlier resolved in context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAMING CONVENTIONS (apply to ALL outputs):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
a. Title case; strip leading articles ("the", "a", "an"). No invented names.
b. Singular generic noun for compound phrases ("Provider" not "providers of online platforms").
c. Named subtypes of same concept (e.g., online provider, offline provider): Abstract Parent UNIT (e.g., Provider) + each subtype as ROLE (Online Provider, Offline Provider).
d. "and/or" combined actors if they have identical tasks/actions: slash notation ("Applicant / Representative").
e. Same base name with different responsibilities or coordination between each other: two separate UNITs.
f. Actors that both ARE an institution and PERFORM tasks: output once as UNIT and once as ROLE.

IMPORTANT: Output ALL valid actors from BOTH sections in a single combined list.
Do not stop early. Include every actor that has any obligation, responsibility, or task in the text.

OUTPUT FORMAT (strict, one line per entry, no bullets, no numbering, no explanations):
ACTOR_NAME | TYPE   (TYPE ∈ {UNIT, ROLE})
"""
