"""
Evaluates generated organigram XML files against gold standard organigrams.

Metrics:
  - Flat: Precision, Recall, F1 for units, roles, subjects (fuzzy-matched)
  - Hierarchy: Precision, Recall, F1 for parent-child relations (units + roles)
  - Graph: Normalized Graph Edit Distance (GED) per organigram
  - Subject binding: Precision, Recall, F1 for (unit, role) pairs in subjects

"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import spacy

NS = "http://cpee.org/ns/organisation/1.0"
SIMILARITY_THRESHOLD = 0.80

try:
    NLP = spacy.load("en_core_web_md")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_md' not found.\n"
        "Install with: python -m spacy download en_core_web_md"
    )


@dataclass
class OrgModel:
    """Parsed representation of one organigram XML file."""
    units: List[str] = field(default_factory=list)  # unit ids
    roles: List[str] = field(default_factory=list)  # role ids
    unit_parents: List[Tuple[str, str]] = field(default_factory=list)  # (child, parent)
    role_parents: List[Tuple[str, str]] = field(default_factory=list)  # (child, parent)
    # subjects as (subject_id, unit, role) triples
    subject_bindings: List[Tuple[str, str, str]] = field(default_factory=list)


# XML Parsing
def _tag(local: str) -> str:
    return f"{{{NS}}}{local}"


def parse_organigram(path: Path) -> OrgModel:
    tree = ET.parse(path)
    root = tree.getroot()
    model = OrgModel()

    for unit_el in root.findall(f".//{_tag('unit')}"):
        uid = unit_el.get("id", "").strip()
        if not uid:
            continue
        model.units.append(uid)
        for parent_el in unit_el.findall(_tag("parent")):
            if parent_el.text:
                model.unit_parents.append((uid, parent_el.text.strip()))

    for role_el in root.findall(f".//{_tag('role')}"):
        rid = role_el.get("id", "").strip()
        if not rid:
            continue
        model.roles.append(rid)
        for parent_el in role_el.findall(_tag("parent")):
            if parent_el.text:
                model.role_parents.append((rid, parent_el.text.strip()))

    for subj_el in root.findall(f".//{_tag('subject')}"):
        sid = subj_el.get("id", "").strip()
        for rel_el in subj_el.findall(_tag("relation")):
            u = rel_el.get("unit", "").strip()
            r = rel_el.get("role", "").strip()
            if u and r:
                model.subject_bindings.append((sid, u, r))

    return model


# Fuzzy matching helpers
def _doc(text: str):
    return NLP(text.lower())


def _best_match(
        candidate: str,
        pool: List[str],
        threshold: float = SIMILARITY_THRESHOLD,
) -> Optional[str]:
    """
    Return the best fuzzy match for `candidate` in `pool` using spaCy
    similarity, or None if no match exceeds threshold.
    """
    if not pool:
        return None

    cand_lower = candidate.lower()
    for name in pool:
        if name.lower() == cand_lower:
            return name

    cand_doc = _doc(candidate)
    if not cand_doc.has_vector:
        for name in pool:
            if cand_lower in name.lower() or name.lower() in cand_lower:
                return name
        return None

    best_score = -1.0
    best_name = None
    for name in pool:
        name_doc = _doc(name)
        if not name_doc.has_vector:
            continue
        score = cand_doc.similarity(name_doc)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name if best_score >= threshold else None


def _fuzzy_match_sets(
        gold: List[str],
        pred: List[str],
        threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[int, int, int]:
    """
    Returns (true_positives, false_positives, false_negatives) using
    greedy fuzzy matching between gold and pred name lists.
    Each gold item can be matched at most once.
    """
    gold_remaining = list(gold)
    tp = fp = 0
    for p in pred:
        match = _best_match(p, gold_remaining, threshold)
        if match is not None:
            tp += 1
            gold_remaining.remove(match)
        else:
            fp += 1
    fn = len(gold_remaining)
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, F1.
    Special case: if all counts are zero (nothing in gold, nothing predicted), return 1.0
    """
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


# Hierarchy evaluation
def _normalize_relation(
        child: str,
        parent: str,
        gold_names: List[str],
) -> Tuple[str, str]:
    """Fuzzy-resolve both sides of a relation to gold vocabulary."""
    matched_child = _best_match(child, gold_names) or child
    matched_parent = _best_match(parent, gold_names) or parent
    return matched_child, matched_parent


def _fuzzy_match_relations(
        gold_rels: List[Tuple[str, str]],
        pred_rels: List[Tuple[str, str]],
        gold_names: List[str],
) -> Tuple[int, int, int]:
    """
    Match predicted parent-child relations against gold relations using
    fuzzy name normalization.
    """
    gold_remaining = list(gold_rels)
    tp = fp = 0
    for (pc, pp) in pred_rels:
        norm = _normalize_relation(pc, pp, gold_names)
        if norm in gold_remaining:
            tp += 1
            gold_remaining.remove(norm)
        else:
            fp += 1
    fn = len(gold_remaining)
    return tp, fp, fn


# Graph Edit Distance
def _build_nx_graph(model: OrgModel) -> nx.DiGraph:
    """
    Build a directed graph from an OrgModel.
    Nodes: all units + roles (labelled by type).
    Edges: parent-child relations.
    """
    G = nx.DiGraph()
    for u in model.units:
        G.add_node(u, kind="unit")
    for r in model.roles:
        G.add_node(r, kind="role")
    for (child, parent) in model.unit_parents:
        G.add_edge(parent, child, rel="unit_hier")
    for (child, parent) in model.role_parents:
        G.add_edge(parent, child, rel="role_hier")
    for (_, unit, role) in model.subject_bindings:
        if unit and role:
            G.add_edge(unit, role, rel="subject_binding")
    return G


def _normalized_ged(gold: OrgModel, pred: OrgModel) -> float:
    """
    Compute normalized GED between gold and pred organigrams.
    Normalization: GED / (|nodes_gold| + |nodes_pred|).
    Uses optimize_graph_edit_distance for a fast upper bound.
    """
    G1 = _build_nx_graph(gold)
    G2 = _build_nx_graph(pred)

    def node_subst_cost(n1_attrs, n2_attrs):
        return 0.0 if n1_attrs.get("kind") == n2_attrs.get("kind") else 1.0

    def node_cost(attrs):
        return 1.0

    def edge_cost(attrs):
        return 1.0

    ged_gen = nx.optimize_graph_edit_distance(
        G1, G2,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_cost,
        node_ins_cost=node_cost,
        edge_del_cost=edge_cost,
        edge_ins_cost=edge_cost,
    )
    ged = next(ged_gen)
    denom = G1.number_of_nodes() + G2.number_of_nodes()
    return round(ged / denom, 4) if denom > 0 else 0.0


# Subject binding evaluation
def _fuzzy_match_bindings(
        gold_bindings: List[Tuple[str, str, str]],
        pred_bindings: List[Tuple[str, str, str]],
        gold_units: List[str],
        gold_roles: List[str],
) -> Tuple[int, int, int]:
    """
    Match (unit, role) pairs from subjects, ignoring subject id (dummy names).
    A predicted (_, pred_unit, pred_role) matches gold (_, gold_unit, gold_role)
    if both unit and role are fuzzy-matched to a gold entry.
    """
    gold_pairs = [(u, r) for (_, u, r) in gold_bindings]
    pred_pairs = [(u, r) for (_, u, r) in pred_bindings]

    gold_remaining = list(gold_pairs)
    tp = fp = 0

    for (pu, pr) in pred_pairs:
        matched_u = _best_match(pu, gold_units)
        matched_r = _best_match(pr, gold_roles)
        if matched_u and matched_r and (matched_u, matched_r) in gold_remaining:
            tp += 1
            gold_remaining.remove((matched_u, matched_r))
        else:
            fp += 1
    fn = len(gold_remaining)
    return tp, fp, fn


def evaluate_pair(gold: OrgModel, pred: OrgModel) -> Dict:
    all_gold_names = list(set(gold.units + gold.roles))

    # Units
    u_tp, u_fp, u_fn = _fuzzy_match_sets(gold.units, pred.units)
    u_p, u_r, u_f1 = _prf(u_tp, u_fp, u_fn)

    # Roles
    r_tp, r_fp, r_fn = _fuzzy_match_sets(gold.roles, pred.roles)
    r_p, r_r, r_f1 = _prf(r_tp, r_fp, r_fn)

    # Unit hierarchies
    uh_tp, uh_fp, uh_fn = _fuzzy_match_relations(
        gold.unit_parents, pred.unit_parents, all_gold_names
    )
    uh_p, uh_r, uh_f1 = _prf(uh_tp, uh_fp, uh_fn)

    # Role hierarchies
    rh_tp, rh_fp, rh_fn = _fuzzy_match_relations(
        gold.role_parents, pred.role_parents, all_gold_names
    )
    rh_p, rh_r, rh_f1 = _prf(rh_tp, rh_fp, rh_fn)

    # Subject bindings (unit+role pairs)
    sb_tp, sb_fp, sb_fn = _fuzzy_match_bindings(
        gold.subject_bindings, pred.subject_bindings,
        gold.units, gold.roles,
    )
    sb_p, sb_r, sb_f1 = _prf(sb_tp, sb_fp, sb_fn)

    # Graph Edit Distance
    nged = _normalized_ged(gold, pred)

    return {
        "gold_units": len(gold.units),
        "pred_units": len(pred.units),
        "gold_roles": len(gold.roles),
        "pred_roles": len(pred.roles),
        "gold_unit_parents": len(gold.unit_parents),
        "pred_unit_parents": len(pred.unit_parents),
        "gold_role_parents": len(gold.role_parents),
        "pred_role_parents": len(pred.role_parents),
        "gold_subject_bindings": len(gold.subject_bindings),
        "pred_subject_bindings": len(pred.subject_bindings),
        "unit_precision": u_p, "unit_recall": u_r, "unit_f1": u_f1,
        "role_precision": r_p, "role_recall": r_r, "role_f1": r_f1,
        "unit_hier_precision": uh_p, "unit_hier_recall": uh_r, "unit_hier_f1": uh_f1,
        "role_hier_precision": rh_p, "role_hier_recall": rh_r, "role_hier_f1": rh_f1,
        "subject_precision": sb_p, "subject_recall": sb_r, "subject_f1": sb_f1,
        "normalized_ged": nged,
    }


def _macro_avg(results: List[Dict]) -> Dict:
    if not results:
        return {}
    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    return {k: round(sum(r[k] for r in results) / len(results), 4) for k in keys}


def main():
    global SIMILARITY_THRESHOLD
    parser = argparse.ArgumentParser(
        description="Evaluate generated organigrams against gold standard."
    )
    parser.add_argument("--gold", required=True,
                        help="Gold standard XML file OR folder with gold XML files.")
    parser.add_argument("--pred", required=True,
                        help="Generated XML file OR folder with generated XML files.")
    parser.add_argument("--out", default="evaluation_results.csv",
                        help="Output CSV file.")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="spaCy similarity threshold for fuzzy matching (default: 0.82).")
    args = parser.parse_args()

    SIMILARITY_THRESHOLD = args.threshold
    gold_path = Path(args.gold)
    pred_path = Path(args.pred)

    if gold_path.is_file() and pred_path.is_file():
        # Single-file mode
        pairs = [(gold_path, pred_path)]
    elif gold_path.is_dir() and pred_path.is_dir():
        # Folder mode: match by filename
        pairs = [
            (gf, pred_path / gf.name)
            for gf in sorted(gold_path.glob("*.xml"))
            if (pred_path / gf.name).exists()
        ]
        if not pairs:
            print(f"No matching XML file pairs found between {gold_path} and {pred_path}")
            return
    else:
        print("ERROR: --gold and --pred must both be files or both be folders.")
        return

    all_results = []

    print(f"\n{'File':<35} {'Unit-F1':>8} {'Role-F1':>8} "
          f"{'UHier-F1':>10} {'RHier-F1':>10} {'Subj-F1':>8} {'NGED':>8}")
    print("-" * 95)

    for gf, pf in pairs:
        gold_model = parse_organigram(gf)
        pred_model = parse_organigram(pf)
        metrics = evaluate_pair(gold_model, pred_model)
        metrics["file"] = gf.name
        all_results.append(metrics)

        print(
            f"  {gf.name:<33} "
            f"{metrics['unit_f1']:>8.4f} "
            f"{metrics['role_f1']:>8.4f} "
            f"{metrics['unit_hier_f1']:>10.4f} "
            f"{metrics['role_hier_f1']:>10.4f} "
            f"{metrics['subject_f1']:>8.4f} "
            f"{metrics['normalized_ged']:>8.4f}"
        )

    if len(all_results) > 1:
        avg = _macro_avg(all_results)
        print("-" * 95)
        print(
            f"  {'MACRO AVERAGE':<33} "
            f"{avg.get('unit_f1', 0):>8.4f} "
            f"{avg.get('role_f1', 0):>8.4f} "
            f"{avg.get('unit_hier_f1', 0):>10.4f} "
            f"{avg.get('role_hier_f1', 0):>10.4f} "
            f"{avg.get('subject_f1', 0):>8.4f} "
            f"{avg.get('normalized_ged', 0):>8.4f}"
        )

    out_path = Path(args.out)
    fieldnames = ["file"] + [k for k in all_results[0] if k != "file"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
        if len(all_results) > 1:
            avg_row = {"file": "MACRO_AVG", **_macro_avg(all_results)}
            writer.writerow(avg_row)

    print(f"\nResults saved to: {out_path}")

    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"per_file": all_results,
                   **({"macro_avg": _macro_avg(all_results)} if len(all_results) > 1 else {})},
                  f, indent=2)
    print(f"JSON summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
