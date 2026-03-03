from dataclasses import dataclass, field
from typing import List


@dataclass
class OrganizationalEntity:
    name: str
    entity_type: str  # "unit" or "role"
    parents: List[str] = field(default_factory=list)        # unit-unit OR role-role
    unit_parents: List[str] = field(default_factory=list)   # ROLE-UNIT


@dataclass
class Subject:
    name: str
    uid: str
    unit: str
    role: str
