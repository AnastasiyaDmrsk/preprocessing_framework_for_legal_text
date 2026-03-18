from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TaskPerformer:
    unit: str
    role: str

@dataclass
class TaskCandidate:
    label: str
    deontic_type: str
    deontic_modality: str
    conditions: List[str]
    source_article: str
    source_paragraph: str
    sentence: str

@dataclass
class TaskException:
    description: str
    ref: Optional[str] = None
    needs_cross_ref: bool = False

@dataclass
class Task:
    id: str
    label: str
    performers: List[TaskPerformer]
    deontic_type: str
    deontic_modality: str
    conditions: List[str]
    exceptions: List[TaskException]
    source_article: str
    source_paragraph: str
