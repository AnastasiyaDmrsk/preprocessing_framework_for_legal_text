from __future__ import annotations
import xml.etree.ElementTree as ET


def parse_organigram(xml_content: str) -> dict:
    """
    Parse an organigram XML and return vis.js-ready {nodes, edges}.
    """
    if not (xml_content or "").strip():
        return {"nodes": [], "edges": []}

    try:
        root = ET.fromstring(xml_content.strip().lstrip("\ufeff"))
    except ET.ParseError:
        return {"nodes": [], "edges": []}

    tag = root.tag
    ns_uri = tag[1:tag.index("}")] if tag.startswith("{") else ""
    ns = {"o": ns_uri} if ns_uri else {}

    def fa(el, path):
        return el.findall(path, ns) if ns else el.findall(path.replace("o:", ""))

    def fo(el, path):
        return el.find(path, ns) if ns else el.find(path.replace("o:", ""))

    def text(el):
        return el.text.strip() if el is not None and el.text else None

    def node_label(el, fallback_id: str) -> str:
        for attr in ("name", "label", "title"):
            v = el.get(attr)
            if v:
                return v.strip()
        for child_tag in ("o:name", "o:label"):
            child = fo(el, child_tag)
            if child is not None and child.text:
                return child.text.strip()
        return fallback_id

    unit_els: dict[str, ET.Element] = {}
    units: dict[str, str | None] = {}
    for u in fa(root, ".//o:units/o:unit"):
        uid = u.get("id")
        if uid:
            unit_els[uid] = u
            units[uid] = text(fo(u, "o:parent"))

    role_els: dict[str, ET.Element] = {}
    roles: dict[str, str | None] = {}
    for r in fa(root, ".//o:roles/o:role"):
        rid = r.get("id")
        if rid:
            role_els[rid] = r
            roles[rid] = text(fo(r, "o:parent"))

    unit_role_pairs: set[tuple[str, str]] = set()
    for subj in fa(root, ".//o:subjects/o:subject"):
        for rel in fa(subj, "o:relation"):
            u, r = rel.get("unit"), rel.get("role")
            if u and r:
                unit_role_pairs.add((u, r))

    _ctr: list[int] = [0]
    _map: dict[str, int] = {}

    def vid(key: str) -> int:
        if key not in _map:
            _ctr[0] += 1
            _map[key] = _ctr[0]
        return _map[key]

    nodes: list[dict] = []

    for uid, el in unit_els.items():
        lbl = node_label(el, uid)
        nodes.append({"id": vid(f"unit:{uid}"), "label": lbl, "title": f"Unit: {lbl}", "shape": "box",
            "color": {"background": "#AED6F1", "border": "#2471A3"},
            "font": {"color": "#1A5276", "bold": True, "size": 15}, "margin": 12, })

    for rid, el in role_els.items():
        lbl = node_label(el, rid)
        nodes.append({"id": vid(f"role:{rid}"), "label": lbl, "title": f"Role: {lbl}", "shape": "ellipse",
            "color": {"background": "#2471A3", "border": "#154360"}, "font": {"color": "#ffffff", "size": 13},
            "margin": 10, })

    edges: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add_edge(sk: str, dk: str, **kw) -> None:
        s, d = vid(sk), vid(dk)
        if (s, d) not in seen:
            seen.add((s, d))
            edges.append({"from": s, "to": d, **kw})

    for uid, parent in units.items():
        if parent and parent in units:
            add_edge(f"unit:{parent}", f"unit:{uid}", arrows="to", color={"color": "#2471A3"}, width=2)

    for rid, parent in roles.items():
        if parent and parent in roles:
            add_edge(f"role:{parent}", f"role:{rid}", arrows="to", color={"color": "#154360"}, dashes=True)

    for unit_id, role_id in unit_role_pairs:
        if unit_id not in units or role_id not in roles:
            continue
        if roles[role_id]:
            continue
        add_edge(f"unit:{unit_id}", f"role:{role_id}", arrows="to", color={"color": "#5D6D7E"})

    return {"nodes": nodes, "edges": edges}
