import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple
from xml.dom import minidom

import google.genai as genai


@dataclass
class OrganizationalEntity:
    name: str
    entity_type: str  # "unit" or "role"
    parents: List[str] = field(default_factory=list)  # replaces single parent


@dataclass
class Subject:
    name: str
    uid: str
    unit: str
    role: str


class LLMOrganizationalExtractor:
    """
    LLM-only organizational knowledge extraction using Gemini which
    produces an XML organigram compatible with CPEE.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def extract_and_save_organigram(
            self,
            preprocessed_text: str,
            output_path: Path,
    ) -> str:
        """
        End‑to‑end: extract units, roles, subjects and save organigram.xml.
        Returns:
            XML string.
        """
        entities, subjects = self._extract_organizational_knowledge(
            preprocessed_text
        )
        xml = self._generate_organigram_xml(entities, subjects, output_path)
        return xml

    def _extract_organizational_knowledge(
            self,
            preprocessed_text: str
    ) -> Tuple[List[OrganizationalEntity], List[Subject]]:
        actors = self._extract_actors(preprocessed_text)
        entities = self._classify_and_structure_entities(actors, preprocessed_text)
        subjects = self._generate_dummy_subjects(entities)
        return entities, subjects

    def _extract_actors(
            self,
            text: str
    ) -> List[Tuple[str, str]]:
        prompt = self._build_actor_extraction_prompt(text)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        response_text = "".join(
            [p.text for p in response.candidates[0].content.parts if p.text]
        )
        return self._parse_actor_response(response_text)

    def _build_actor_extraction_prompt(
            self,
            text: str,
    ) -> str:
        prompt = """You are an expert in business process modeling and organizational structures for EU regulatory documents.

TASK:
From the preprocessed regulatory text below, extract all organizational ACTORS that participate in the process.

RESOURCE MODEL:
- Organization: A high-level legal entity (e.g. "Commission", "Member State", "Customer", "Entity").
- Organizational Unit: A subdivision within an organization that has its own responsibilities
  (e.g. "Management Board", "Executive Board").
- Role: A position or functional role that can be held by one or more subjects within an organization or unit
  (e.g. "Manager", "Officer", "Network Operator", "Customer", "Provider").
- Subject: A specific individual or system account (e.g. "Max Mustermann").

OUTPUT SCHEMA:
- TYPE = UNIT: any organization or organizational unit that can act as a pool or lane.
- TYPE = ROLE: any role or function that actively participates in the process and can be assigned to a subject within a unit.

IMPORTANT:
- If an entity acts both as a unit AND as a role (e.g. "Commission", "Board"),
  output it TWICE: once as UNIT and once as ROLE.
- For compound phrases like "Digital Services Coordinator of the Member State":
  extract BOTH the role ("Digital Services Coordinator") AND the unit ("Member State") separately.

ACTOR DEFINITION:
An ACTOR is any UNIT or ROLE that:
- Performs, is responsible for, or is associated with an activity or task, OR
- Is referenced as a participant, recipient of a duty, or authority in the process

RULES:
1. Extract the exact surface form without leading articles:
   - "Commission" not "The Commission"
2. For compound phrases with "of", extract each constituent separately:
   - "Digital Services Coordinator of the Member State" results into "Digital Services Coordinator | ROLE" and "Member State | UNIT"
3. Include generic roles if they perform or receive actions:
   - e.g. "Stakeholder", "Entity"
4. Include non-human actors if treated as resources:
   - e.g. "System"
5. Do NOT extract:
   - Abstract concepts (e.g. "programme", "scheme", "notice", "process")
   - Data objects (e.g. "report", "application", "guidelines", "database")
   - Pure activities (verbs or verb phrases without an actor)
6. Each ACTOR must appear only once per TYPE.
7. Only include actors that actually appear in the INPUT TEXT.


OUTPUT FORMAT (strict with one actor per line, no bullets, no numbering, no explanations):
ACTOR_NAME | TYPE

Where TYPE is exactly one of:
- UNIT
- ROLE
"""
        prompt += "\nINPUT TEXT:\n"
        prompt += text.strip()
        prompt += "\n\nOUTPUT (just the list, no explanations):\n"

        return prompt

    def _parse_actor_response(self, response: str) -> List[Tuple[str, str]]:
        actors: List[Tuple[str, str]] = []
        for line in response.strip().splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            name_part, type_part = line.split("|", 1)
            name = name_part.strip()
            t = type_part.strip().upper()
            if name and t in {"UNIT", "ROLE"}:
                actors.append((name, t))

        seen = set()
        unique: List[Tuple[str, str]] = []
        for name, t in actors:
            key = (name.lower(), t)
            if key not in seen:
                seen.add(key)
                unique.append((name, t))
        print(unique)
        return unique

    def _classify_and_structure_entities(
            self,
            actors: List[Tuple[str, str]],
            text: str,
    ) -> List[OrganizationalEntity]:
        actor_lines = "\n".join([f"- {n} ({t})" for n, t in actors])

        prompt = f"""You are analyzing organizational hierarchies in EU regulatory documents.

    ACTORS IDENTIFIED:
    {actor_lines}

    Each line above has the form:
    ACTOR_NAME | TYPE
    where TYPE is either UNIT or ROLE.

    RESOURCE MODEL:
    - UNIT: Any organization or organizational unit that can act as a pool or lane (e.g. "Commission", "ENISA").
    - ROLE: Any role, function, or position that can be assigned to a subject or group of subjects within a unit
      (e.g. "Commission", "Management Board", "Assistant").

    IMPORTANT:
    - An ACTOR can appear both as UNIT and as ROLE (e.g. "Commission", "ENISA", "Customer").
    - Hierarchies can exist:
      - between UNITS (unit–unit),
      - between ROLES (role–role),
      - but NOT between a UNIT and a ROLE.

    TASK:
    Identify all parent-child relationships between:
      (a) UNITS, and
      (b) ROLES.

    SEMANTICS OF PARENT-CHILD:
    1. UNIT–UNIT hierarchy:
       A unit B is a CHILD of unit A (A is the PARENT) if the text clearly indicates that:
       - B is an internal body, board, unit or department of A
         (e.g. "Management Board of ENISA"), or
       - B is structurally located within A (e.g. "Management Board in the Member State").

    2. ROLE–ROLE hierarchy:
       A role C is a CHILD of role D (D is the parent) if the text or domain conventions indicate that:
       - C is a specialization or sub-type of D,
         (e.g. "Assistant" is a staff role within "Project" and "Regular" roles),
       - or C inherits permissions/characteristics from D.
       Multiple parents are allowed for a role and a unit.

    CONSTRAINTS:
    1. Only consider names that appear in ACTORS IDENTIFIED with the corresponding TYPE.
       - UNIT–UNIT relations may only involve TYPE = UNIT actors.
       - ROLE–ROLE relations may only involve TYPE = ROLE actors.
    2. Never create UNIT–ROLE or ROLE–UNIT relations.
    3. Do NOT invent names that are not in ACTORS IDENTIFIED.
    4. If the text or domain knowledge does not clearly support any parent-child relations,
       output "NO_HIERARCHIES".

    DOMAIN HINTS:
    - If both "ENISA" and "Management Board" appear as UNITS, and the text states that
      decisions for ENISA are taken by the Management Board, assume:
      Management Board is a child UNIT of ENISA.
    - For roles like "Assistant" and "Staff" that are described as working under broader
      roles such as "Project" and "Regular", assume:
      Assistant is a child ROLE of Project and Regular,
      Staff is a child ROLE of Project and Regular.

    OUTPUT FORMAT (strict):
    List each parent-child relation on its own line in one of the two forms:

    For UNIT–UNIT relations:
    CHILD_UNIT | PARENT_UNIT | UNIT

    For ROLE–ROLE relations:
    CHILD_ROLE | PARENT_ROLE | ROLE

    Where the last field is the literal string "UNIT" or "ROLE" and indicates the hierarchy type.
    If a child has multiple parents, emit one line per parent.

    EXAMPLE:

    ACTORS IDENTIFIED:
    ENISA | UNIT
    Management Board | UNIT
    ENISA | ROLE
    Management Board | ROLE
    Assistant | ROLE
    Project | ROLE
    Regular | ROLE

    Expected Output:
    Management Board | ENISA | UNIT
    Assistant | Project | ROLE
    Assistant | Regular | ROLE

    INPUT TEXT:
    {text}
    """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        response_text = "".join(
            [p.text for p in response.candidates[0].content.parts if p.text]
        )

        # child_key -> list of parent names
        # key is (name.lower(), rel_type) so unit and role hierarchies are separate
        unit_hierarchies: Dict[str, List[str]] = {}
        role_hierarchies: Dict[str, List[str]] = {}

        for line in response_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if "NO_HIERARCHIES" in line.upper():
                unit_hierarchies = {}
                role_hierarchies = {}
                break
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    child_name, parent_name, rel_type = parts
                    rel_type = rel_type.upper()
                    if child_name and parent_name:
                        if rel_type == "UNIT":
                            unit_hierarchies.setdefault(child_name, []).append(parent_name)
                        elif rel_type == "ROLE":
                            role_hierarchies.setdefault(child_name, []).append(parent_name)

        entities: List[OrganizationalEntity] = []
        for name, t in actors:
            if t == "UNIT":
                parents = unit_hierarchies.get(name, [])
                entities.append(
                    OrganizationalEntity(name=name, entity_type="unit", parents=parents)
                )
            else:
                parents = role_hierarchies.get(name, [])
                entities.append(
                    OrganizationalEntity(name=name, entity_type="role", parents=parents)
                )

        # Add "External" unit if there are roles with no corresponding unit
        unit_names = {e.name for e in entities if e.entity_type == "unit"}
        role_names = {e.name for e in entities if e.entity_type == "role"}
        external_roles = role_names - unit_names
        if external_roles and "External" not in unit_names:
            entities.append(
                OrganizationalEntity(name="External", entity_type="unit", parents=[])
            )

        return entities

    def _generate_dummy_subjects(
            self,
            entities: List[OrganizationalEntity],
    ) -> List[Subject]:
        subjects: List[Subject] = []
        unit_names = {e.name for e in entities if e.entity_type == "unit"}
        roles = [e for e in entities if e.entity_type == "role"]

        for role in roles:
            unit = role.name if role.name in unit_names else "External"

            base = role.name.lower().replace("-", " ").replace("/", " ")
            tokens = [t for t in base.split() if t]
            if not tokens:
                tokens = ["role"]

            first = f"{tokens[0]}_name"
            last = f"{tokens[-1]}_surname"
            uid_parts = [tok[:2] for tok in tokens[:2]]
            uid = "".join(uid_parts) + "s"

            subjects.append(
                Subject(
                    name=f"{first} {last}",
                    uid=uid,
                    unit=unit,
                    role=role.name,
                )
            )

        return subjects

    def _generate_organigram_xml(
            self,
            entities: List[OrganizationalEntity],
            subjects: List[Subject],
            output_path: Path,
    ) -> str:
        root = ET.Element("organisation")
        root.set("xmlns", "http://cpee.org/ns/organisation/1.0")

        # units
        units_el = ET.SubElement(root, "units")
        for e in entities:
            if e.entity_type != "unit":
                continue
            u_el = ET.SubElement(units_el, "unit", id=e.name)
            for parent in e.parents:  # one <parent> per entry
                p_el = ET.SubElement(u_el, "parent")
                p_el.text = parent
            ET.SubElement(u_el, "permissions")

        # roles
        roles_el = ET.SubElement(root, "roles")
        for e in entities:
            if e.entity_type != "role":
                continue
            r_el = ET.SubElement(roles_el, "role", id=e.name)
            for parent in e.parents:  # one <parent> per entry
                p_el = ET.SubElement(r_el, "parent")
                p_el.text = parent
            ET.SubElement(r_el, "permissions")

        # subjects
        subjects_el = ET.SubElement(root, "subjects")
        for s in subjects:
            s_el = ET.SubElement(subjects_el, "subject", id=s.name, uid=s.uid)
            ET.SubElement(s_el, "relation", unit=s.unit, role=s.role)

        xml_str = ET.tostring(root, encoding="unicode")
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        pretty_xml = "\n".join(line for line in pretty_xml.splitlines() if line.strip())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        return pretty_xml


def build_organigram_xml(
        preprocessed_text: str,
        api_key: str,
        output_dir: Path,
        model: str = "gemini-2.5-flash",
) -> str:
    """
    Build organigram.xml from preprocessed text using LLM.

    The file is saved as: output_dir / "organigram.xml".
    """
    extractor = LLMOrganizationalExtractor(api_key=api_key, model=model)
    output_path = output_dir / "organigram.xml"
    xml = extractor.extract_and_save_organigram(
        preprocessed_text=preprocessed_text,
        output_path=output_path,
    )
    return xml
