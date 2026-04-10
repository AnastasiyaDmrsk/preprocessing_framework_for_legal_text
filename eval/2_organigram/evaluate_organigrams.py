"""
Evaluates generated organigram XML files against gold standard organigrams.

Metrics:
  - Flat: Precision, Recall, F1 for units and roles (fuzzy-matched)
  - Hierarchy: Precision, Recall, F1 for parent-child relations (units + roles)
  - Subject binding: Precision, Recall, F1 for (unit, role) pairs

Security: requires defusedxml (pip install defusedxml) to prevent XXE attacks.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import spacy

try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

    warnings.warn(
        "defusedxml not installed — falling back to stdlib ET (XXE risk). "
        "Fix: pip install defusedxml",
        RuntimeWarning,
        stacklevel=1,
    )

NS = "http://cpee.org/ns/organisation/1.0"
_DEFAULT_THRESHOLD = 0.80

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
    units: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    unit_parents: List[Tuple[str, str]] = field(default_factory=list)  # (child, parent)
    role_parents: List[Tuple[str, str]] = field(default_factory=list)  # (child, parent)
    subject_bindings: List[Tuple[str, str, str]] = field(default_factory=list)  # (sid, unit, role)


# XML Parsing
def _tag(local: str) -> str:
    return f"{{{NS}}}{local}"


def parse_organigram(path: Path) -> OrgModel:
    tree = ET.parse(str(path))
    root = tree.getroot()
    model = OrgModel()

    for unit_el in root.findall(f".//{_tag('unit')}"):
        uid = unit_el.get("id", "").strip()
        if not uid:
            continue
        model.units.append(uid)
        for p_el in unit_el.findall(_tag("parent")):
            if p_el.text:
                model.unit_parents.append((uid, p_el.text.strip()))

    for role_el in root.findall(f".//{_tag('role')}"):
        rid = role_el.get("id", "").strip()
        if not rid:
            continue
        model.roles.append(rid)
        for p_el in role_el.findall(_tag("parent")):
            if p_el.text:
                model.role_parents.append((rid, p_el.text.strip()))

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


def _similarity(a: str, b: str) -> float:
    """Symmetric string similarity via spaCy vectors with substring tiebreaker."""
    al, bl = a.lower(), b.lower()
    if al == bl:
        return 1.0
    if al in bl or bl in al:
        # Substring containment: high but not perfect, avoids "AI" → "AI Office" at 1.0
        return 0.92
    da, db = NLP(al), NLP(bl)
    if da.has_vector and db.has_vector:
        return float(da.similarity(db))
    return 0.0


def _build_mapping(
        pred: List[str],
        gold: List[str],
        threshold: float,
) -> Dict[str, str]:
    """
    Build a deterministic 1-to-1 pred→gold mapping.

    Scores all (pred, gold) pairs, sorts descending by similarity, then greedily
    assigns highest-scoring unmatched pairs. This is order-independent and
    approximates the optimal bipartite matching (Hungarian algorithm) for the
    typical small sizes of organigram vocabularies.
    """
    scored: List[Tuple[float, str, str]] = []
    for p in pred:
        for g in gold:
            s = _similarity(p, g)
            if s >= threshold:
                scored.append((s, p, g))

    scored.sort(key=lambda x: x[0], reverse=True)

    mapping: Dict[str, str] = {}
    used_gold: set = set()
    for _, p, g in scored:
        if p not in mapping and g not in used_gold:
            mapping[p] = g
            used_gold.add(g)
    return mapping


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, F1.
    All-zero case (nothing in gold AND nothing predicted) → perfect score 1.0,
    meaning the model correctly produced nothing when nothing was expected.
    Any non-zero mismatch is penalised normally.
    """
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _prf_from_mapping(
        mapping: Dict[str, str],
        pred: List[str],
        gold: List[str],
) -> Tuple[float, float, float]:
    tp = len(mapping)
    fp = len(pred) - tp
    fn = len(gold) - tp
    return _prf(tp, fp, fn)


def _evaluate_relations(
        pred_rels: List[Tuple[str, str]],
        gold_rels: List[Tuple[str, str]],
        mapping: Dict[str, str],
) -> Tuple[int, int, int]:
    """
    Evaluate parent-child edges using a pre-built node mapping.
    Predicted names are translated to gold vocabulary before comparison,
    ensuring consistency with the flat entity evaluation.
    """
    gold_remaining = list(gold_rels)
    tp = fp = 0
    for child, parent in pred_rels:
        norm = (mapping.get(child, child), mapping.get(parent, parent))
        if norm in gold_remaining:
            tp += 1
            gold_remaining.remove(norm)
        else:
            fp += 1
    fn = len(gold_remaining)
    return tp, fp, fn


def _evaluate_subject_bindings(
        pred_bindings: List[Tuple[str, str, str]],
        gold_bindings: List[Tuple[str, str, str]],
        unit_map: Dict[str, str],
        role_map: Dict[str, str],
) -> Tuple[int, int, int]:
    """
    Evaluate (unit, role) subject-binding pairs using pre-built mappings.
    Subject ids are intentionally ignored (dummy placeholder names).
    """
    gold_pairs = [(u, r) for (_, u, r) in gold_bindings]
    gold_remaining = list(gold_pairs)
    tp = fp = 0
    for _, pu, pr in pred_bindings:
        mu = unit_map.get(pu)
        mr = role_map.get(pr)
        if mu and mr and (mu, mr) in gold_remaining:
            tp += 1
            gold_remaining.remove((mu, mr))
        else:
            fp += 1
    fn = len(gold_remaining)
    return tp, fp, fn


def evaluate_pair(
        gold: OrgModel,
        pred: OrgModel,
        threshold: float = _DEFAULT_THRESHOLD,
) -> Dict:
    """
    Evaluate a single (gold, pred) organigram pair.

    A single stable mapping is built first and reused across ALL metrics,
    guaranteeing that a predicted node resolves to the same gold node whether
    it appears in a flat list, a parent-child relation, or a subject binding.
    """
    unit_map = _build_mapping(pred.units, gold.units, threshold)
    role_map = _build_mapping(pred.roles, gold.roles, threshold)

    u_p, u_r, u_f1 = _prf_from_mapping(unit_map, pred.units, gold.units)
    r_p, r_r, r_f1 = _prf_from_mapping(role_map, pred.roles, gold.roles)

    uh_tp, uh_fp, uh_fn = _evaluate_relations(pred.unit_parents, gold.unit_parents, unit_map)
    uh_p, uh_r, uh_f1 = _prf(uh_tp, uh_fp, uh_fn)

    rh_tp, rh_fp, rh_fn = _evaluate_relations(pred.role_parents, gold.role_parents, role_map)
    rh_p, rh_r, rh_f1 = _prf(rh_tp, rh_fp, rh_fn)

    sb_tp, sb_fp, sb_fn = _evaluate_subject_bindings(
        pred.subject_bindings, gold.subject_bindings, unit_map, role_map
    )
    sb_p, sb_r, sb_f1 = _prf(sb_tp, sb_fp, sb_fn)

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
    }


def _macro_avg(results: List[Dict]) -> Dict:
    if not results:
        return {}
    float_keys = [k for k in results[0] if isinstance(results[0][k], float)]
    return {k: round(sum(r[k] for r in results) / len(results), 4) for k in float_keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated organigrams against gold standard."
    )
    parser.add_argument("--gold", required=True,
                        help="Gold standard XML file OR directory.")
    parser.add_argument("--pred", required=True,
                        help="Predicted XML file OR directory.")
    parser.add_argument("--out", default="evaluation_results.csv",
                        help="Output CSV file path.")
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD,
                        help=f"spaCy similarity threshold (default: {_DEFAULT_THRESHOLD}).")
    parser.add_argument("--ged-timeout", type=float, default=5.0, dest="ged_timeout",
                        help="GED timeout in seconds before falling back to upper bound (default: 5.0).")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    pred_path = Path(args.pred)

    if gold_path.is_file() and pred_path.is_file():
        pairs = [(gold_path, pred_path)]
    elif gold_path.is_dir() and pred_path.is_dir():
        pairs = [
            (gf, pred_path / gf.name)
            for gf in sorted(gold_path.glob("*.xml"))
            if (pred_path / gf.name).exists()
        ]
        if not pairs:
            print(f"No matching XML pairs found between '{gold_path}' and '{pred_path}'.")
            return
    else:
        print("ERROR: --gold and --pred must both be files or both be directories.")
        return

    all_results: List[Dict] = []
    print(
        f"\n{'File':<35} {'Unit-F1':>8} {'Role-F1':>8} "
        f"{'UHier-F1':>10} {'RHier-F1':>10} {'Subj-F1':>8}"
    )
    print("-" * 95)

    for gf, pf in pairs:
        gold_model = parse_organigram(gf)
        pred_model = parse_organigram(pf)
        metrics = evaluate_pair(
            gold_model, pred_model,
            threshold=args.threshold,
        )
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

        if len(all_results) > 1:
            avg_row: Dict = {"file": "MACRO_AVG", **_macro_avg(all_results)}
            for fn_key in fieldnames:
                avg_row.setdefault(fn_key, "")
            writer.writerow(avg_row)

    print(f"\nResults saved to: {out_path}")

    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_file": all_results,
                **({"macro_avg": _macro_avg(all_results)} if len(all_results) > 1 else {}),
            },
            f,
            indent=2,
        )
    print(f"JSON summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
