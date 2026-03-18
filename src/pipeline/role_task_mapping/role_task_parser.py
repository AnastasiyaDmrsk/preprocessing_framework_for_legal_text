from __future__ import annotations
import xml.etree.ElementTree as ET


# Maps deontic type → (CSS class, badge label)
DEONTIC_STYLE: dict[str, tuple[str, str]] = {
    "obligation":     ("deontic-obligation",     "shall"),
    "permission":     ("deontic-permission",      "may"),
    "prohibition":    ("deontic-prohibition",     "shall not"),
    "recommendation": ("deontic-recommendation", "should"),
}


def parse_role_task_mapping(xml_content: str) -> list[dict]:
    if not (xml_content or "").strip():
        return []

    try:
        root = ET.fromstring(xml_content.strip().lstrip("\ufeff"))
    except ET.ParseError:
        return []

    tasks: list[dict] = []

    for task_el in root.iter("task"):
        label_el = task_el.find("label")
        label    = label_el.text.strip() if label_el is not None and label_el.text else ""

        deontic_el   = task_el.find("deontic")
        deontic_type = (deontic_el.get("type", "") if deontic_el is not None else "").lower()
        css_class, badge = DEONTIC_STYLE.get(deontic_type, ("", deontic_el.get("modality", "") if deontic_el is not None else ""))

        actors: list[str] = []
        for performer in task_el.findall(".//performer"):
            role_el = performer.find("role")
            unit_el = performer.find("unit")
            name = (role_el.text if role_el is not None and role_el.text
                    else unit_el.text if unit_el is not None and unit_el.text
                    else None)
            if name:
                actors.append(name.strip())

        conditions: list[str] = [
            c.text.strip()
            for c in task_el.findall(".//condition")
            if c.text and c.text.strip()
        ]

        exceptions: list[str] = []
        for exc in task_el.findall(".//exception"):
            desc = exc.get("description", "").strip() or (exc.text or "").strip()
            if desc:
                exceptions.append(desc)

        src_el = task_el.find("source-ref")
        source = ""
        if src_el is not None:
            article   = src_el.get("article",   "")
            paragraph = src_el.get("paragraph", "")
            source    = f"Art. {article}" if article else ""
            if paragraph:
                source += f" §{paragraph}"

        tasks.append({
            "id":           task_el.get("id", ""),
            "label":        label,
            "deontic_type": deontic_type,
            "deontic_css":  css_class,
            "deontic_badge": badge,
            "actors":       actors,
            "conditions":   conditions,
            "exceptions":   exceptions,
            "source":       source,
        })

    return tasks
