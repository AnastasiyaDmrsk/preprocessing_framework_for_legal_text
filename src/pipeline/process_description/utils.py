import re
from typing import List, Tuple

from .models import ExtractedTask


def parse_paragraph_sort_key(paragraph: str) -> Tuple:
    if paragraph == "UNKNOWN":
        return 999, ""
    m = re.match(r'(\d+)(?:\(([a-z])\))?', paragraph)
    if m:
        return int(m.group(1)), m.group(2) or ""
    return 999, paragraph


def serialize_tasks(tasks: List[ExtractedTask]) -> str:
    return "\n\n".join(_serialize_task(t) for t in tasks)


def _serialize_task(task: ExtractedTask) -> str:
    actor_str = " or ".join(task.actors) if task.actors else "Unknown Actor"
    lines = [f"[{task.task_id}] {actor_str} {task.deontic_modality} {task.label}"]
    for cond in task.conditions:
        lines.append(f"  CONDITION: {cond}")
    for exc in task.exceptions:
        lines.append(f"  EXCEPTION: {exc}")
    lines.append(f"  SOURCE: Article {task.source_article}, §{task.source_paragraph}")
    return "\n".join(lines)
