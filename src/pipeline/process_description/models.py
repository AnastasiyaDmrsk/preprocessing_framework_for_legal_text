from dataclasses import dataclass, field
from typing import List


@dataclass
class ExtractedTask:
    task_id: str
    actors: List[str]
    deontic_modality: str
    label: str
    conditions: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    source_article: str = "UNKNOWN"
    source_paragraph: str = "UNKNOWN"
