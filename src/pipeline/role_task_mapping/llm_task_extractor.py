import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple

import google.genai as genai

from src.pipeline.role_task_mapping.const import (
    TASK_EXTRACTION_PROMPT,
    TASK_VALIDATION_PROMPT,
    DEFAULT_TASK_EXTRACTION_MAX_TOKENS,
    DEFAULT_TASK_VALIDATION_MAX_TOKENS,
)
from .models import Task, TaskException, TaskPerformer
from ..organigram.const import DEFAULT_MODEL, DEFAULT_VALIDATOR_MODEL, DEFAULT_VALIDATOR_TEMPERATURE
from ..organigram.utils import create_xml


class LLMTaskExtractor:

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        validator_model: str = DEFAULT_VALIDATOR_MODEL,
        use_validator: bool = True,
    ):
        self._performer_lookup: Dict[str, List[TaskPerformer]] = {}
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.validator_model = validator_model
        self.use_validator = use_validator

    def extract_and_save_tasks(
            self,
            text: str,
            organigram_xml: str,
            output_path: Path,
    ) -> str:
        self._performer_lookup = self._build_performer_lookup(organigram_xml)
        tasks = self._extract_tasks(text, organigram_xml)
        if self.use_validator:
            tasks = self._validate_tasks(tasks, text, organigram_xml)
        self._assign_task_ids(tasks)
        self._resolve_cross_refs(tasks)
        return self._generate_tasks_xml(tasks, output_path)

    def _build_performer_lookup(
            self, organigram_xml: str
    ) -> Dict[str, List[TaskPerformer]]:
        lookup: Dict[str, List[TaskPerformer]] = {}
        try:
            root = ET.fromstring(organigram_xml)
            for subj in root.findall(".//{*}subject"):
                rel = subj.find("{*}relation")
                if rel is None:
                    continue
                unit = rel.get("unit", "").strip()
                role = rel.get("role", "").strip()
                if not unit or not role:
                    continue
                performer = TaskPerformer(unit=unit, role=role)
                for key in {unit.lower(), role.lower()}:
                    lookup.setdefault(key, [])
                    if performer not in lookup[key]:
                        lookup[key].append(performer)
        except ET.ParseError as e:
            warnings.warn(f"Failed to parse organigram XML: {e}", RuntimeWarning)
        return lookup

    def _parse_performers(self, raw: str) -> List[TaskPerformer]:
        performers: List[TaskPerformer] = []
        seen: Set[Tuple[str, str]] = set()

        def add(p: TaskPerformer) -> None:
            key = (p.unit, p.role)
            if key not in seen:
                seen.add(key)
                performers.append(p)

        for entry in raw.split(";;"):
            entry = entry.strip()
            if not entry or entry == "(none)":
                continue

            unit_match = re.search(r'unit\s*:\s*([^,]+)', entry, re.IGNORECASE)
            role_match = re.search(r'role\s*:\s*([^,;]+)', entry, re.IGNORECASE)
            if unit_match and role_match:
                add(TaskPerformer(
                    unit=unit_match.group(1).strip(),
                    role=role_match.group(1).strip(),
                ))
                continue

            if "/" in entry:
                unit, _, role = entry.partition("/")
                unit, role = unit.strip(), role.strip()
                if unit and role:
                    add(TaskPerformer(unit=unit, role=role))
                continue

            lookup = getattr(self, "_performer_lookup", {})
            key = entry.lower().strip()
            resolved = False
            for map_key, ps in lookup.items():
                if key == map_key or key in map_key or map_key in key:
                    for p in ps:
                        add(p)
                    resolved = True
                    break
            if not resolved:
                add(TaskPerformer(unit=entry, role=entry))

        return performers

    def _call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.1,
        model: str = None,
    ) -> str:
        response = self.client.models.generate_content(
            model=model or self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return "".join(
            p.text for p in response.candidates[0].content.parts if p.text
        )

    def _extract_tasks(self, text: str, organigram_xml: str) -> List[Task]:
        subjects_str = self._serialize_organigram_subjects(organigram_xml)
        prompt = (
            TASK_EXTRACTION_PROMPT
            + "\n\nORGANIGRAM SUBJECTS:\n"
            + subjects_str
            + "\n\nINPUT TEXT:\n"
            + text.strip()
        )
        response = self._call(prompt, DEFAULT_TASK_EXTRACTION_MAX_TOKENS)
        return self._parse_task_blocks(response)

    def _validate_tasks(
        self,
        tasks: List[Task],
        text: str,
        organigram_xml: str,
    ) -> List[Task]:
        subjects_str = self._serialize_organigram_subjects(organigram_xml)
        proposed = self._serialize_tasks_for_prompt(tasks)
        prompt = (
            TASK_VALIDATION_PROMPT
            + "\n\nORGANIGRAM SUBJECTS:\n"
            + subjects_str
            + "\n\nPROPOSED TASK LIST:\n"
            + proposed
            + "\n\nINPUT TEXT:\n"
            + text.strip()
            + "\n\nBegin your step-by-step reasoning, then output <FINAL_TASKS>:\n"
        )
        response = self._call(
            prompt,
            DEFAULT_TASK_VALIDATION_MAX_TOKENS,
            temperature=DEFAULT_VALIDATOR_TEMPERATURE,
            model=self.validator_model,
        )
        match = re.search(
            r"<FINAL_TASKS>(.*?)</FINAL_TASKS>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return self._parse_task_blocks(match.group(1))
        warnings.warn(
            "Task validator missing <FINAL_TASKS> tags. Using full response.",
            RuntimeWarning,
        )
        return self._parse_task_blocks(response)
    def _assign_task_ids(self, tasks: List[Task]) -> None:
        for i, task in enumerate(tasks, 1):
            task.id = f"t{i}"

    def _resolve_cross_refs(self, tasks: List[Task]) -> None:
        """
        Assign exception.ref values.
        Exceptions without [CROSS_REF] → self-reference (same task id).
        Exceptions with [CROSS_REF] → find best matching task by word overlap.
        """
        STOP_WORDS = {
            "the", "a", "an", "of", "in", "to", "and", "or", "that",
            "is", "are", "be", "by", "for", "with", "not", "if", "it",
        }

        def tokenize(text: str) -> Set[str]:
            return {
                w.lower() for w in re.findall(r'\w+', text)
                if w.lower() not in STOP_WORDS
            }

        task_tokens = [
            (t.id, tokenize(t.label + " " + " ".join(t.conditions)))
            for t in tasks
        ]

        for task in tasks:
            for exc in task.exceptions:
                if not exc.needs_cross_ref:
                    exc.ref = task.id
                    continue
                exc_tokens = tokenize(exc.description)
                best_id, best_score = task.id, 0
                for tid, tokens in task_tokens:
                    if tid == task.id:
                        continue
                    score = len(exc_tokens & tokens)
                    if score > best_score:
                        best_score = score
                        best_id = tid
                exc.ref = best_id
                exc.needs_cross_ref = False

    def _parse_task_blocks(self, response: str) -> List[Task]:
        blocks = re.findall(r'---TASK---(.*?)---END---', response, re.DOTALL)
        tasks: List[Task] = []
        for block in blocks:
            fields: Dict[str, str] = {}
            for line in block.strip().splitlines():
                if ':' not in line:
                    continue
                key, _, val = line.partition(':')
                fields[key.strip().lower()] = val.strip()

            label = fields.get('label', '').strip()
            if not label:
                continue
            performers = self._parse_performers(fields.get('performers', ''))
            conditions = [
                c.strip()
                for c in fields.get('conditions', '').split(';;')
                if c.strip()
            ]
            exceptions = self._parse_exceptions(fields.get('exceptions', ''))

            tasks.append(Task(
                id='',
                label=label,
                performers=performers,
                deontic_type=fields.get('deontic_type', 'obligation').strip(),
                deontic_modality=fields.get('deontic_modality', 'shall').strip(),
                conditions=conditions,
                exceptions=exceptions,
                source_article=fields.get('article', 'UNKNOWN').strip(),
                source_paragraph=fields.get('paragraph', 'UNKNOWN').strip(),
            ))

        return tasks

    def _parse_exceptions(self, raw: str) -> List[TaskException]:
        exceptions: List[TaskException] = []
        for exc_str in raw.split(';;'):
            exc_str = exc_str.strip()
            if not exc_str:
                continue
            needs_cross_ref = '[CROSS_REF]' in exc_str
            description = exc_str.replace('[CROSS_REF]', '').strip()
            if description:
                exceptions.append(TaskException(
                    description=description,
                    needs_cross_ref=needs_cross_ref,
                ))
        return exceptions

    def _serialize_organigram_subjects(self, organigram_xml: str) -> str:
        lines: List[str] = []
        try:
            root = ET.fromstring(organigram_xml)
            ns = {"o": "http://cpee.org/ns/organisation/1.0"}
            for subj in root.findall(".//{*}subject"):
                rel = subj.find("{*}relation")
                if rel is None:
                    continue
                unit = rel.get("unit", "").strip()
                role = rel.get("role", "").strip()
                if unit and role:
                    lines.append(f"unit: {unit}, role: {role}")
        except ET.ParseError as e:
            warnings.warn(f"Failed to parse organigram XML: {e}", RuntimeWarning)
        return "\n".join(lines) if lines else ""

    def _serialize_tasks_for_prompt(self, tasks: List[Task]) -> str:
        blocks: List[str] = []
        for task in tasks:
            performers_str = " ;; ".join(
                f"{p.unit}/{p.role}" for p in task.performers
            ) or ""
            conditions_str = " ;; ".join(task.conditions) or ""
            exceptions_str = " ;; ".join(
                e.description + ("[CROSS_REF]" if e.needs_cross_ref else "")
                for e in task.exceptions
            ) or ""
            blocks.append(
                f"---TASK---\n"
                f"label: {task.label}\n"
                f"performers: {performers_str}\n"
                f"deontic_type: {task.deontic_type}\n"
                f"deontic_modality: {task.deontic_modality}\n"
                f"conditions: {conditions_str}\n"
                f"exceptions: {exceptions_str}\n"
                f"article: {task.source_article}\n"
                f"paragraph: {task.source_paragraph}\n"
                f"---END---"
            )
        return "\n\n".join(blocks)

    def _generate_tasks_xml(self, tasks: List[Task], output_path: Path) -> str:
        root = ET.Element("tasks")

        for task in tasks:
            task_el = ET.SubElement(root, "task", id=task.id)

            label_el = ET.SubElement(task_el, "label")
            label_el.text = task.label

            performers_el = ET.SubElement(task_el, "performers")
            for p in task.performers:
                perf_el = ET.SubElement(performers_el, "performer")
                unit_el = ET.SubElement(perf_el, "unit")
                unit_el.text = p.unit
                role_el = ET.SubElement(perf_el, "role")
                role_el.text = p.role

            ET.SubElement(
                task_el, "deontic",
                type=task.deontic_type,
                modality=task.deontic_modality,
            )

            conditions_el = ET.SubElement(task_el, "conditions")
            for cond in task.conditions:
                cond_el = ET.SubElement(conditions_el, "condition")
                cond_el.text = cond

            exceptions_el = ET.SubElement(task_el, "exceptions")
            for exc in task.exceptions:
                attrs = {"description": exc.description}
                if exc.ref:
                    attrs["ref"] = exc.ref
                ET.SubElement(exceptions_el, "exception", **attrs)

            ET.SubElement(
                task_el, "source-ref",
                article=task.source_article,
                paragraph=task.source_paragraph,
            )

        return create_xml(root, output_path)
