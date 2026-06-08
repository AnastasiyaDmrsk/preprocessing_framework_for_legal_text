from dataclasses import dataclass

@dataclass
class CallInfo:
    call_id: str
    label: str
    role: str
    unit: str

@dataclass
class TaskRole:
    task_id: str
    label: str
    role: str
    unit: str