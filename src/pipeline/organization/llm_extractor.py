from pathlib import Path
from typing import List, Dict, Tuple

import google.genai as genai

from .const import DEFAULT_MODEL
from .models import OrganizationalEntity, Subject
from .utils import _build_hierarchy_extraction_prompt, _build_actor_extraction_prompt, _generate_organigram_xml


def _generate_dummy_subjects(
        entities: List[OrganizationalEntity],
) -> List[Subject]:
    subjects: List[Subject] = []
    unit_names = {e.name for e in entities if e.entity_type == "unit"}
    roles = [e for e in entities if e.entity_type == "role"]

    uid_counts: Dict[str, int] = {}

    for role in roles:
        base = role.name.lower().replace("-", " ").replace("/", " ")
        tokens = [t for t in base.split() if t]
        if not tokens:
            tokens = ["role"]

        first = f"{tokens[0]}_name"
        last = f"{tokens[-1]}_surname"
        uid_base = "".join(tok[:2] for tok in tokens[:2])

        binding_units = [u for u in role.unit_parents if u in unit_names]
        if not binding_units:
            binding_units = [role.name if role.name in unit_names else "External"]

        for unit in binding_units:
            uid_counts[uid_base] = uid_counts.get(uid_base, 0) + 1
            count = uid_counts[uid_base]
            uid = uid_base + ("s" if count == 1 else str(count))

            subjects.append(Subject(
                name=f"{first} {last}",
                uid=uid,
                unit=unit,
                role=role.name,
            ))

    return subjects


class LLMOrganizationalExtractor:
    """
    Pure-LLM organigram extraction.
    Produces a CPEE-compatible organigram.xml.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_content(self, prompt: str, max_output_tokens: int) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_output_tokens,
            ),
        )
        response_text = "".join(
            p.text for p in response.candidates[0].content.parts if p.text
        )
        return response_text

    def extract_and_save_organigram(
            self,
            preprocessed_text: str,
            output_path: Path,
    ) -> str:
        entities, subjects = self._extract_organizational_knowledge(preprocessed_text)
        return _generate_organigram_xml(entities, subjects, output_path)

    def _extract_organizational_knowledge(
            self,
            preprocessed_text: str,
    ) -> Tuple[List[OrganizationalEntity], List[Subject]]:
        actors = self._extract_actors(preprocessed_text)
        entities = self._classify_and_structure_entities(actors, preprocessed_text)
        subjects = _generate_dummy_subjects(entities)
        return entities, subjects

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

        seen: set = set()
        unique: List[Tuple[str, str]] = []
        for name, t in actors:
            key = (name.lower(), t)
            if key not in seen:
                seen.add(key)
                unique.append((name, t))
        return unique

    def _extract_actors(self, text: str) -> List[Tuple[str, str]]:
        prompt = _build_actor_extraction_prompt(text)
        response_text = self.generate_content(prompt, 1024)
        return self._parse_actor_response(response_text)

    def _classify_and_structure_entities(
            self,
            actors: List[Tuple[str, str]],
            text: str,
    ) -> List[OrganizationalEntity]:
        actor_lines = "\n".join([f"- {n} ({t})" for n, t in actors])

        prompt = _build_hierarchy_extraction_prompt(actor_lines, text)
        response_text = self.generate_content(prompt, 4092)

        unit_hierarchies: Dict[str, List[str]] = {}
        role_hierarchies: Dict[str, List[str]] = {}
        role_unit_map: Dict[str, List[str]] = {}

        for line in response_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if "NO_HIERARCHIES" in line.upper():
                unit_hierarchies = {}
                role_hierarchies = {}
                role_unit_map = {}
                break
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    child_name, parent_name, rel_type = parts
                    rel_type = rel_type.upper().replace(" ", "-")
                    if child_name and parent_name:
                        if rel_type == "UNIT":
                            unit_hierarchies.setdefault(child_name, []).append(parent_name)
                        elif rel_type == "ROLE":
                            role_hierarchies.setdefault(child_name, []).append(parent_name)
                        elif rel_type == "ROLE-UNIT":
                            role_unit_map.setdefault(child_name, []).append(parent_name)

        entities: List[OrganizationalEntity] = []
        for name, t in actors:
            if t == "UNIT":
                entities.append(OrganizationalEntity(
                    name=name,
                    entity_type="unit",
                    parents=unit_hierarchies.get(name, []),
                ))
            else:
                entities.append(OrganizationalEntity(
                    name=name,
                    entity_type="role",
                    parents=role_hierarchies.get(name, []),
                    unit_parents=role_unit_map.get(name, []),
                ))

        existing_unit_names = {e.name for e in entities if e.entity_type == "unit"}

        referenced_but_missing: set = set()
        for e in entities:
            if e.entity_type == "role":
                for u in e.unit_parents:
                    if u not in existing_unit_names:
                        referenced_but_missing.add(u)

        for u in referenced_but_missing:
            entities.append(OrganizationalEntity(name=u, entity_type="unit", parents=[]))

        existing_unit_names |= referenced_but_missing
        for e in entities:
            if e.entity_type == "role" and not e.unit_parents:
                if e.name not in existing_unit_names:
                    e.unit_parents = ["External"]

        needs_external = any(
            e.entity_type == "role" and e.unit_parents == ["External"]
            for e in entities
        )
        if needs_external and "External" not in existing_unit_names:
            entities.append(OrganizationalEntity(name="External", entity_type="unit", parents=[]))

        return entities
