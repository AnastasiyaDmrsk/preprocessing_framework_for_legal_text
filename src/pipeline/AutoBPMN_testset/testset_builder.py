from __future__ import annotations

import difflib
import hashlib
import re
import uuid
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union, Callable
from urllib.parse import urlsplit

import google.genai as genai

from src.pipeline.AutoBPMN_testset.const import TESTSET_DESCRIPTION_PROMPT, TESTSET_DESCRIPTION_VALIDATION_PROMPT, \
    _TESTSET_TMPL, DESC_NS, DEFAULT_VALIDATOR_MAX_TOKENS, DEFAULT_GENERATOR_TEMPERATURE, DEFAULT_GENERATOR_MAX_TOKENS, \
    DEFAULT_VALIDATOR_TEMPERATURE, ANNO_NS, DEFAULT_CPEE_BASE, PROP_NS
from src.pipeline.AutoBPMN_testset.models import TaskRole, CallInfo

ET.register_namespace("", DESC_NS)


def _d(tag: str) -> str:
    return f"{{{DESC_NS}}}{tag}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read(source: Union[str, Path]) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if source.lstrip().startswith("<"):
        return source
    p = Path(source)
    return p.read_text(encoding="utf-8") if p.exists() else source


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:xml)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def cpee_endpoint(base_url: str, path: str, method: str) -> str:
    parts = urlsplit(base_url)
    scheme = parts.scheme or "https"
    host = parts.netloc or parts.path
    if not path.startswith("/"):
        path = "/" + path
    return f"{scheme}-{method}://{host}{path}"


def plain_url(base_url: str, path: str) -> str:
    parts = urlsplit(base_url)
    scheme = parts.scheme or "https"
    host = parts.netloc or parts.path
    if not path.startswith("/"):
        path = "/" + path
    return f"{scheme}://{host}{path}"


def worklist_path(job_id: str) -> str:
    return f"/worklist/{job_id}/"


def organigram_path(job_id: str) -> str:
    return f"/organigram/{job_id}"


def parse_role_mapping(source: Union[str, Path]) -> list[TaskRole]:
    out: list[TaskRole] = []
    try:
        root = ET.fromstring(_read(source).strip().lstrip("\ufeff"))
    except ET.ParseError as e:
        warnings.warn(f"Failed to parse role_task_mapping XML: {e}", RuntimeWarning)
        return out

    for task_node in root.findall(".//{*}task"):
        task_id = task_node.get("id", "unknown")
        label_node = task_node.find(".//{*}label")
        label = label_node.text.strip() if label_node is not None and label_node.text else ""

        performer = task_node.find(".//{*}performer")
        role, unit = "", "*"
        if performer is not None:
            role_node = performer.find("{*}role")
            unit_node = performer.find("{*}unit")
            if role_node is not None and role_node.text:
                role = role_node.text.strip()
            if unit_node is not None and unit_node.text:
                unit = unit_node.text.strip() or "*"
        out.append(TaskRole(task_id=task_id, label=label, role=role, unit=unit))
    return out


def serialize_role_mapping(mapping: list[TaskRole]) -> str:
    return "\n".join(f"- label={t.label!r} | role={t.role!r} | unit={t.unit!r}" for t in mapping)


class RoleAssigner:
    def __init__(self, mapping: list[TaskRole], by_id: dict[str, str] | None = None,
                 by_label: dict[str, str] | None = None):
        self.mapping = mapping
        self.by_id = by_id or {}
        self.by_label = by_label or {}
        self._unit_for_role = {t.role: t.unit for t in mapping if t.role}
        self._unit_for_label = {t.label: t.unit for t in mapping}

    def resolve(self, call_id: str, label: str) -> tuple[str, str]:
        role = self.by_id.get(call_id) or self.by_label.get(label) or ""
        if not role:
            role = self._fuzzy_label(label)
        unit = (self._unit_for_role.get(role) or self._unit_for_label.get(label) or "*")
        return role, unit or "*"

    def _fuzzy_label(self, label: str) -> str:
        best, score = None, -1.0
        for t in self.mapping:
            s = difflib.SequenceMatcher(None, label.lower(), t.label.lower()).ratio()
            if s > score:
                best, score = t, s
        return best.role if best else ""


class LLMTestsetDescriptionGenerator:
    def __init__(self, api_key: str, model: str, validator_model: str = None, use_validator: bool = True):
        # self.client = genai.Client(api_key=api_key)
        self.model = model
        self.validator_model = validator_model or model
        self.use_validator = use_validator

    def generate(self, process_description: str, serialized_mapping: str) -> str:
        block = self._generate(process_description, serialized_mapping)
        if self.use_validator:
            block = self._validate(block, process_description, serialized_mapping)
        return _strip_fences(block)

    def _generate(self, process_description: str, serialized_mapping: str) -> str:
        prompt = (TESTSET_DESCRIPTION_PROMPT.replace("[PROCESS_DESCRIPTION]", process_description.strip()).replace(
            "[ROLE_TASK_MAPPING]", serialized_mapping.strip()))
        result = self._call(prompt, DEFAULT_GENERATOR_MAX_TOKENS, DEFAULT_GENERATOR_TEMPERATURE)
        return result

    def _validate(self, block: str, process_description: str, serialized_mapping: str) -> str:
        prompt = (
            TESTSET_DESCRIPTION_VALIDATION_PROMPT.replace("[PROCESS_DESCRIPTION]", process_description.strip()).replace(
                "[ROLE_TASK_MAPPING]", serialized_mapping.strip()).replace("[PROPOSED_DESCRIPTION]", block.strip()))
        response = self._call(prompt, DEFAULT_VALIDATOR_MAX_TOKENS, DEFAULT_VALIDATOR_TEMPERATURE,
                              model=self.validator_model)
        match = re.search(r"<FINAL_DESCRIPTION>(.*?)</FINAL_DESCRIPTION>", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        warnings.warn("Testset validator missing <FINAL_DESCRIPTION> tags. Using generator output.", RuntimeWarning, )
        return block

    def _call(self, prompt: str, max_tokens: int, temperature: float, model: str = None) -> str:
        response = self.client.models.generate_content(model=model or self.model, contents=prompt,
                                                       config=genai.types.GenerateContentConfig(temperature=temperature,
                                                                                                max_output_tokens=max_tokens, ), )
        return "".join(p.text for p in response.candidates[0].content.parts if p.text).strip()


def _inner_description(xml_content: str) -> ET.Element:
    root = ET.fromstring(_read(xml_content).strip().lstrip("\ufeff"))
    if root.tag == _d("description"):
        return root
    for el in root.iter(_d("description")):
        return el
    if _local(root.tag) == "description":  # block without the xmlns
        return root
    raise ValueError("No <description> element found in the control-flow block.")


def _iter_calls(elem: ET.Element):
    return (e for e in elem.iter() if _local(e.tag) == "call")


def extract_roles_from_block(block_xml: str) -> tuple[dict[str, str], dict[str, str]]:
    by_id: dict[str, str] = {}
    by_label: dict[str, str] = {}
    try:
        inner = _inner_description(block_xml)
    except (ET.ParseError, ValueError):
        return by_id, by_label
    for call in _iter_calls(inner):
        cid = call.get("id", "")
        label_node = call.find(".//{*}label")
        role_node = call.find(".//{*}role")
        label = label_node.text.strip() if label_node is not None and label_node.text else ""
        role = role_node.text.strip() if role_node is not None and role_node.text else ""
        if role:
            if cid:
                by_id[cid] = role
            if label:
                by_label[label] = role
    return by_id, by_label


def enrich_structure(structure_xml: str, *, role_for: Callable[[str, str], tuple[str, str]], orgmodel_url: str,
                     form_url: str, priority: int = 1, handling: str = "single", ) -> tuple[str, list[CallInfo]]:
    inner = _inner_description(structure_xml)

    for el in inner.iter():
        for k in list(el.attrib):
            if k.startswith(f"{{{ANNO_NS}}}"):
                del el.attrib[k]

    calls: list[CallInfo] = []
    for call in _iter_calls(inner):
        label_node = call.find(".//{*}label")
        label = label_node.text.strip() if label_node is not None and label_node.text else ""
        cid = call.get("id", "")
        role, unit = role_for(cid, label)

        call.set("endpoint", "worklist")
        for child in list(call):
            call.remove(child)

        params = ET.SubElement(call, _d("parameters"))
        ET.SubElement(params, _d("label")).text = label
        args = ET.SubElement(params, _d("arguments"))
        ET.SubElement(args, _d("orgmodel")).text = orgmodel_url
        ET.SubElement(args, _d("form")).text = form_url
        ET.SubElement(args, _d("role")).text = role
        ET.SubElement(args, _d("priority")).text = str(priority)
        ET.SubElement(args, _d("handling")).text = handling

        calls.append(CallInfo(cid, label, role, unit))

    ET.indent(inner, space="  ")
    return ET.tostring(inner, encoding="unicode"), calls


def assemble_testset(inner_description_xml: str, *, worklist_endpoint: str, orgmodel_url: str) -> str:
    indented = "\n".join("    " + line for line in inner_description_xml.splitlines())
    return _TESTSET_TMPL.format(prop=PROP_NS, worklist_ep=worklist_endpoint, orgmodel=orgmodel_url, inner=indented, )


def build_worklist_xml(calls: list[CallInfo], *, orgmodel_url: str, cpee_base: str = DEFAULT_CPEE_BASE,
                       process_name: str = "Worklist", instance: Optional[int] = None,
                       instance_uuid: Optional[str] = None, user_uid: str = "test", user_name: str = "Tester",
                       job_id: str = "", ) -> str:
    if instance is None:
        seed = job_id or uuid.uuid4().hex
        instance = int(hashlib.md5(seed.encode()).hexdigest()[:6], 16)
    if instance_uuid is None:
        instance_uuid = str(uuid.uuid4())

    root = ET.Element("tasks")
    seen: set[tuple[str, str, str]] = set()
    for c in calls:
        key = (c.call_id, c.label, c.role)
        if key in seen:
            continue
        seen.add(key)
        callback_id = uuid.uuid4().hex
        t = ET.SubElement(root, "task", {"callback_id": callback_id,
                                         "cpee_callback": f"{cpee_base}{instance}/callbacks/{callback_id}/",
                                         "cpee_instance": str(instance), "cpee_base": cpee_base,
                                         "instance_uuid": instance_uuid, "cpee_label": c.label,
                                         "cpee_activity": c.call_id, "orgmodel": orgmodel_url, })
        ET.SubElement(t, "process").text = f"{process_name} ({instance})"
        ET.SubElement(t, "label").text = c.label
        ET.SubElement(t, "role").text = c.role
        ET.SubElement(t, "unit").text = c.unit
        ET.SubElement(t, "user", {"uid": user_uid}).text = user_name

    ET.indent(root, space="  ")
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode")
