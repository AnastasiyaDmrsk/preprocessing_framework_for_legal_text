import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from xml.dom import minidom

from .const import (
    BLACKLIST_RE,
    ROLE_SUFFIXES,
    UNIT_SUFFIXES,
    ACTOR_EXTRACTION_PROMPT,
    HIERARCHY_EXTRACTION_PROMPT,
    PRE_EXTRACTED_ACTORS_IDENTIFIED,
)
from .models import OrganizationalEntity, Subject


def _generate_organigram_xml(
        entities: List[OrganizationalEntity],
        subjects: List[Subject],
        output_path: Path,
) -> str:
    root = ET.Element("organisation")
    root.set("xmlns", "http://cpee.org/ns/organisation/1.0")

    units_el = ET.SubElement(root, "units")
    for e in entities:
        if e.entity_type != "unit":
            continue
        u_el = ET.SubElement(units_el, "unit", id=e.name)
        for parent in e.parents:
            p_el = ET.SubElement(u_el, "parent")
            p_el.text = parent
        ET.SubElement(u_el, "permissions")

    roles_el = ET.SubElement(root, "roles")
    for e in entities:
        if e.entity_type != "role":
            continue
        r_el = ET.SubElement(roles_el, "role", id=e.name)
        for parent in e.parents:
            p_el = ET.SubElement(r_el, "parent")
            p_el.text = parent
        ET.SubElement(r_el, "permissions")

    subjects_el = ET.SubElement(root, "subjects")
    for s in subjects:
        s_el = ET.SubElement(subjects_el, "subject", id=s.name, uid=s.uid)
        ET.SubElement(s_el, "relation", unit=s.unit, role=s.role)

    return create_xml(root, output_path)


def create_xml(root: ET.Element, output_path: Path) -> str:
    xml_str = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(xml_str)
    pretty = dom.toprettyxml(indent="  ")
    pretty = "\n".join(
        line for line in pretty.splitlines() if line.strip()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)
    return pretty


# Prompt builders

def _build_actor_extraction_prompt(text: str) -> str:
    prompt = ACTOR_EXTRACTION_PROMPT
    prompt += "\nINPUT TEXT:\n"
    prompt += text.strip()
    prompt += "\n\nOUTPUT (just the list, no explanations):\n"
    return prompt


def _build_hierarchy_extraction_prompt(actor_lines: str, text: str) -> str:
    prompt = "You are analyzing organizational hierarchies in EU regulatory documents.\nACTORS IDENTIFIED:\n"
    prompt += actor_lines.strip()
    prompt += HIERARCHY_EXTRACTION_PROMPT
    prompt += "\nINPUT TEXT:\n"
    prompt += text.strip()
    return prompt


def _build_pre_extracted_actors_hierarchy_prompt(candidate_lines: str, text: str) -> str:
    prompt = "You are an expert in EU regulatory document analysis and BPMN/CPEE organizational modeling.\n"
    prompt += "NLP PRE-EXTRACTED ACTOR CANDIDATES:\n"
    prompt += candidate_lines.strip()
    prompt += PRE_EXTRACTED_ACTORS_IDENTIFIED
    prompt += "\n\nINPUT TEXT:\n"
    prompt += text.strip()
    prompt += "\n\nOUTPUT:\n"
    return prompt


# Text helpers

def normalize_name(text: str) -> str:
    text = re.compile(
        r"^\s*(the|a|an|any|each|every|all|such|those|these|this|that|their|its)\s+",
        re.IGNORECASE,
    ).sub("", text.strip())
    return " ".join(text.split())


def is_actor(text: str) -> bool:
    return not BLACKLIST_RE.match(text.strip())


def infer_type(text: str) -> str:
    lower = text.lower()
    if any(lower.endswith(s) for s in ROLE_SUFFIXES):
        return "ROLE"
    if any(lower.endswith(s) for s in UNIT_SUFFIXES):
        return "UNIT"
    return "UNIT"
