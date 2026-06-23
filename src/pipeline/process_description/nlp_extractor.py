import re
import warnings
import xml.etree.ElementTree as ET
from typing import List

from .models import ExtractedTask
from .utils import parse_paragraph_sort_key


def _parse_xml(xml_content: str) -> List[ExtractedTask]:
    tasks: List[ExtractedTask] = []
    try:
        root = ET.fromstring(xml_content.strip().lstrip("\ufeff"))

        for task_node in root.findall(".//{*}task"):
            task_id = task_node.get("id", "unknown")

            label_node = task_node.find(".//{*}label")
            label = label_node.text.strip() if label_node is not None and label_node.text else ""

            deontic_node = task_node.find(".//{*}deontic")
            modality = deontic_node.get("modality", "shall") if deontic_node is not None else "shall"

            actors: List[str] = []
            for p_node in task_node.findall(".//{*}performer"):
                role_node = p_node.find("{*}role")
                if role_node is not None and role_node.text:
                    actor = role_node.text.strip()
                    if actor not in actors:
                        actors.append(actor)

            conditions = [
                c.text.strip()
                for c in task_node.findall(".//{*}condition")
                if c.text
            ]

            exceptions = [
                e.get("description", "").strip()
                for e in task_node.findall(".//{*}exception")
                if e.get("description", "").strip()
            ]

            source_node = task_node.find(".//{*}source-ref")
            article = source_node.get("article", "UNKNOWN") if source_node is not None else "UNKNOWN"
            paragraph = source_node.get("paragraph", "UNKNOWN") if source_node is not None else "UNKNOWN"

            tasks.append(ExtractedTask(
                task_id=task_id,
                actors=actors,
                deontic_modality=modality,
                label=label,
                conditions=conditions,
                exceptions=exceptions,
                source_article=article,
                source_paragraph=paragraph,
            ))

    except ET.ParseError as e:
        warnings.warn(f"Failed to parse role_task_mapping XML: {e}", RuntimeWarning)

    return tasks


class ProcessDescriptionNLPExtractor:

    def extract_tasks(self, xml_content: str) -> List[ExtractedTask]:
        tasks = _parse_xml(xml_content)
        return self._sort_tasks(tasks)

    def _sort_tasks(self, tasks: List[ExtractedTask]) -> List[ExtractedTask]:
        return sorted(
            tasks,
            key=lambda t: (
                self._article_sort_key(t.source_article),
                parse_paragraph_sort_key(t.source_paragraph),
                int(t.task_id[1:]) if t.task_id[1:].isdigit() else 0,
            ),
        )

    def _article_sort_key(self, article: str) -> int:
        m = re.match(r'(\d+)', article)
        return int(m.group(1)) if m else 999
