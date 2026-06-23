from dataclasses import dataclass, field
from typing import List


@dataclass
class OrganizationalEntity:
    name: str
    entity_type: str
    parents: List[str] = field(default_factory=list)
    unit_parents: List[str] = field(default_factory=list)


@dataclass
class Subject:
    name: str
    uid: str
    unit: str
    role: str
