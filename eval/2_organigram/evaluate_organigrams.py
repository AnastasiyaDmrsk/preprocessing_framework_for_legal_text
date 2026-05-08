"""
Evaluates generated organigram XML files against gold standard organigrams.

Metrics:
  - Flat: Precision, Recall, F1 for units and roles (fuzzy-matched)
  - Hierarchy: Precision, Recall, F1 for parent-child relations (units + roles)
  - Subject binding (dependent): Precision, Recall, F1 for (unit, role) pairs
        using pre-built flat entity mappings — measures end-to-end pipeline quality.
  - Subject binding (independent): Precision, Recall, F1 for (unit, role) pairs
        via sentence-transformer bipartite matching — isolates binding quality from
        entity extraction errors and handles synonymous unit/role names.
  - Role-only binding: Precision, Recall, F1 matching pred bindings to gold
        on role name alone — isolates role assignment from unit naming errors.
  - Role coverage: % of gold roles that have ≥1 subject binding referencing them
        in the predicted output — sanity check orthogonal to binding correctness.
  - Actor identification: Precision, Recall, F1 for whether any actor (unit or role)
        was identified at all, regardless of unit/role classification.

Dependencies: pip install defusedxml spacy sentence-transformers
               python -m spacy download en_core_web_md
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import spacy
from sentence_transformers import SentenceTransformer, util as st_util

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
_BINDING_THRESHOLD = 0.75
_RUN_DIRS = ["run_1", "run_2", "run_3"]
_TEXT_TYPES = ["preprocessed_text", "raw_text"]
try:
    NLP = spacy.load("en_core_web_md")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_md' not found.\n"
        "Install with: python -m spacy download en_core_web_md"
    )

try:
    _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as exc:
    raise RuntimeError(
        "sentence-transformers model 'all-MiniLM-L6-v2' could not be loaded.\n"
        "Install with: pip install sentence-transformers\n"
        f"Original error: {exc}"
    )


@dataclass
class OrgModel:
    """Parsed representation of one organigram XML file."""
    units: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    unit_parents: List[Tuple[str, str]] = field(default_factory=list)  # (child, parent)
    role_parents: List[Tuple[str, str]] = field(default_factory=list)  # (child, parent)
    subject_bindings: List[Tuple[str, str, str]] = field(default_factory=list)  # (sid, unit, role)


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


# Similarity functions

@functools.lru_cache(maxsize=8192)
def _spacy_similarity(a: str, b: str) -> float:
    """
    spaCy vector similarity with exact-match and substring short-circuits.
    Cached to dramatically speed up bipartite matching loops.
    """
    al, bl = a.lower(), b.lower()
    if al == bl:
        return 1.0
    if al > bl:
        al, bl = bl, al
    if al in bl or bl in al:
        return 0.92

    da, db = NLP(al), NLP(bl)
    if da.has_vector and db.has_vector:
        return float(da.similarity(db))
    return 0.0


@functools.lru_cache(maxsize=8192)
def _sbert_similarity(a: str, b: str) -> float:
    """
    Sentence-transformer cosine similarity. Cached for performance.
    Handles synonyms ("organization" ≈ "company") better than spaCy vectors.
    """
    al, bl = a.lower(), b.lower()
    if al == bl:
        return 1.0
    # Normalize order to maximize cache hits (cosine sim is symmetric)
    if al > bl:
        al, bl = bl, al
    if al in bl or bl in al:
        return 0.92

    emb = _SBERT.encode([al, bl], convert_to_tensor=True, show_progress_bar=False)
    return float(st_util.cos_sim(emb[0], emb[1]))


# Bipartite matching helpers
def _build_mapping(
        pred: List[str],
        gold: List[str],
        threshold: float,
        sim_fn=_spacy_similarity,
) -> Dict[str, str]:
    """
    Greedy 1-to-1 pred→gold mapping approximating optimal bipartite matching.
    Scores all pairs, sorts descending, assigns highest-scoring unmatched pairs.
    """
    scored: List[Tuple[float, str, str]] = []
    for p in pred:
        for g in gold:
            s = sim_fn(p, g)
            if s >= threshold:
                scored.append((s, p, g))

    scored.sort(key=lambda x: x[0], reverse=True)

    mapping: Dict[str, str] = {}
    used_gold: Set[str] = set()
    for _, p, g in scored:
        if p not in mapping and g not in used_gold:
            mapping[p] = g
            used_gold.add(g)
    return mapping


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, F1.
    All-zero case → perfect score 1.0 (model correctly produced nothing).
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
    """Evaluate parent-child edges using a pre-built node mapping."""
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


# Subject binding — dependent (end-to-end, unchanged from original)
def _evaluate_subject_bindings_dependent(
        pred_bindings: List[Tuple[str, str, str]],
        gold_bindings: List[Tuple[str, str, str]],
        unit_map: Dict[str, str],
        role_map: Dict[str, str],
) -> Tuple[int, int, int]:
    """
    Evaluate (unit, role) pairs using pre-built flat entity mappings.
    Cascades errors from flat extraction: missed unit/role → binding FP/FN.
    Subject IDs are intentionally ignored (dummy placeholder names).
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


# Subject binding independent
def _evaluate_subject_bindings_independent(
        pred_bindings: List[Tuple[str, str, str]],
        gold_bindings: List[Tuple[str, str, str]],
        threshold: float = _BINDING_THRESHOLD,
) -> Tuple[int, int, int]:
    """
    Evaluate (unit, role) pairs via independent pair-level SBERT matching.

    Both the unit and role component must individually exceed `threshold`.
    The combined score is min(unit_sim, role_sim) — the weaker link governs.
    Greedy 1-to-1 bipartite assignment on descending score.

    Advantage over dependent metric: a binding can match even when the unit or
    role string was lost during the greedy flat mapping step, and synonymous
    names (e.g. "organization" / "company") are handled via SBERT embeddings.
    """
    pred_pairs = [(u, r) for (_, u, r) in pred_bindings]
    gold_pairs = [(u, r) for (_, u, r) in gold_bindings]

    scored: List[Tuple[float, int, int]] = []
    for i, (pu, pr) in enumerate(pred_pairs):
        for j, (gu, gr) in enumerate(gold_pairs):
            u_sim = _sbert_similarity(pu, gu)
            r_sim = _sbert_similarity(pr, gr)
            if u_sim >= threshold and r_sim >= threshold:
                scored.append((min(u_sim, r_sim), i, j))

    scored.sort(key=lambda x: x[0], reverse=True)

    matched_pred: Set[int] = set()
    matched_gold: Set[int] = set()
    tp = 0
    for _, i, j in scored:
        if i not in matched_pred and j not in matched_gold:
            tp += 1
            matched_pred.add(i)
            matched_gold.add(j)

    fp = len(pred_pairs) - tp
    fn = len(gold_pairs) - tp
    return tp, fp, fn


# Role-only binding evaluation
def _evaluate_role_only_binding(
        pred_bindings: List[Tuple[str, str, str]],
        gold_bindings: List[Tuple[str, str, str]],
        threshold: float = _BINDING_THRESHOLD,
) -> Tuple[int, int, int]:
    """
    Evaluate whether the predicted role in each subject binding matches any
    gold role binding, completely ignoring the unit component.
    """
    pred_roles = [r for (_, _u, r) in pred_bindings]
    gold_roles = [r for (_, _u, r) in gold_bindings]

    scored: List[Tuple[float, int, int]] = []
    for i, pr in enumerate(pred_roles):
        for j, gr in enumerate(gold_roles):
            s = _sbert_similarity(pr, gr)
            if s >= threshold:
                scored.append((s, i, j))

    scored.sort(key=lambda x: x[0], reverse=True)

    matched_pred: Set[int] = set()
    matched_gold: Set[int] = set()
    tp = 0
    for _, i, j in scored:
        if i not in matched_pred and j not in matched_gold:
            tp += 1
            matched_pred.add(i)
            matched_gold.add(j)

    fp = len(pred_roles) - tp
    fn = len(gold_roles) - tp
    return tp, fp, fn


def _evaluate_role_coverage(
        gold: OrgModel,
        pred: OrgModel,
        threshold: float = _BINDING_THRESHOLD,
) -> float:
    """
    For each gold role, check whether ≥1 predicted subject binding references
    a role that SBERT-matches it above threshold.
    Returns the fraction of gold roles covered (0.0–1.0).
    """
    if not gold.roles:
        return 1.0

    pred_roles_in_bindings = [r for (_, _u, r) in pred.subject_bindings]

    covered = 0
    for gold_role in gold.roles:
        for pred_role in pred_roles_in_bindings:
            if _sbert_similarity(gold_role, pred_role) >= threshold:
                covered += 1
                break

    return round(covered / len(gold.roles), 4)


def _evaluate_actor_identification(
        gold: OrgModel,
        pred: OrgModel,
        threshold: float,
) -> Tuple[float, float, float]:
    """
    Evaluate whether each actor (unit or role id) was identified at all,
    regardless of unit/role classification.
    """
    gold_actors = list(dict.fromkeys(gold.units + gold.roles))
    pred_actors = list(dict.fromkeys(pred.units + pred.roles))
    actor_map = _build_mapping(pred_actors, gold_actors, threshold)
    return _prf_from_mapping(actor_map, pred_actors, gold_actors)


def evaluate_pair(
        gold: OrgModel,
        pred: OrgModel,
        threshold: float = _DEFAULT_THRESHOLD,
) -> Dict:
    """
    Evaluate a single (gold, pred) organigram pair.
    """
    # --- Flat entity mapping (spaCy, reused for hierarchy + dependent) ----
    unit_map = _build_mapping(pred.units, gold.units, threshold, _spacy_similarity)
    role_map = _build_mapping(pred.roles, gold.roles, threshold, _spacy_similarity)

    u_p, u_r, u_f1 = _prf_from_mapping(unit_map, pred.units, gold.units)
    r_p, r_r, r_f1 = _prf_from_mapping(role_map, pred.roles, gold.roles)

    # --- Hierarchy ---------------------------------------------------------
    uh_tp, uh_fp, uh_fn = _evaluate_relations(
        pred.unit_parents, gold.unit_parents, unit_map
    )
    uh_p, uh_r, uh_f1 = _prf(uh_tp, uh_fp, uh_fn)

    rh_tp, rh_fp, rh_fn = _evaluate_relations(
        pred.role_parents, gold.role_parents, role_map
    )
    rh_p, rh_r, rh_f1 = _prf(rh_tp, rh_fp, rh_fn)

    # --- Subject binding: dependent (spaCy flat map, cascades errors) ------
    sb_tp, sb_fp, sb_fn = _evaluate_subject_bindings_dependent(
        pred.subject_bindings, gold.subject_bindings, unit_map, role_map,
    )
    sb_p, sb_r, sb_f1 = _prf(sb_tp, sb_fp, sb_fn)

    # --- Subject binding: independent (SBERT, lower threshold) ------------
    sbi_tp, sbi_fp, sbi_fn = _evaluate_subject_bindings_independent(
        pred.subject_bindings, gold.subject_bindings, _BINDING_THRESHOLD,
    )
    sbi_p, sbi_r, sbi_f1 = _prf(sbi_tp, sbi_fp, sbi_fn)

    # --- Role-only binding (SBERT, ignores unit component) ----------------
    rb_tp, rb_fp, rb_fn = _evaluate_role_only_binding(
        pred.subject_bindings, gold.subject_bindings, _BINDING_THRESHOLD,
    )
    rb_p, rb_r, rb_f1 = _prf(rb_tp, rb_fp, rb_fn)

    # --- Role coverage (% of gold roles with ≥1 subject in pred) ----------
    role_coverage = _evaluate_role_coverage(gold, pred, _BINDING_THRESHOLD)

    # --- Actor identification (unit OR role, any classification) ----------
    ac_p, ac_r, ac_f1 = _evaluate_actor_identification(gold, pred, threshold)

    return {
        # Counts
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
        # Flat entity metrics (spaCy)
        "unit_precision": u_p, "unit_recall": u_r, "unit_f1": u_f1,
        "role_precision": r_p, "role_recall": r_r, "role_f1": r_f1,
        # Hierarchy metrics
        "unit_hier_precision": uh_p, "unit_hier_recall": uh_r, "unit_hier_f1": uh_f1,
        "role_hier_precision": rh_p, "role_hier_recall": rh_r, "role_hier_f1": rh_f1,
        # Subject binding — dependent
        "subject_dep_precision": sb_p, "subject_dep_recall": sb_r, "subject_dep_f1": sb_f1,
        # Subject binding — independent
        "subject_indep_precision": sbi_p, "subject_indep_recall": sbi_r, "subject_indep_f1": sbi_f1,
        # Role-only binding
        "role_binding_precision": rb_p, "role_binding_recall": rb_r, "role_binding_f1": rb_f1,
        # Role coverage
        "role_coverage": role_coverage,
        # Actor identification
        "actor_precision": ac_p, "actor_recall": ac_r, "actor_f1": ac_f1,
    }


# Aggregation
def _macro_avg(results: List[Dict]) -> Dict:
    if not results:
        return {}
    float_keys = [k for k in results[0] if isinstance(results[0][k], float)]
    return {k: round(sum(r[k] for r in results) / len(results), 4) for k in float_keys}


_COL_HEADER = (
    f"{'File':<35} {'Unit-F1':>8} {'Role-F1':>8} "
    f"{'UHier-F1':>10} {'RHier-F1':>10} "
    f"{'SubjDep-F1':>11} {'SubjInd-F1':>11} "
    f"{'RoleBind-F1':>12} {'RoleCov':>8} {'Actor-F1':>10}"
)
_COL_SEP = "-" * len(_COL_HEADER)


def _fmt_row(label: str, m: Dict) -> str:
    return (
        f"  {label:<33} "
        f"{m.get('unit_f1', 0):>8.4f} "
        f"{m.get('role_f1', 0):>8.4f} "
        f"{m.get('unit_hier_f1', 0):>10.4f} "
        f"{m.get('role_hier_f1', 0):>10.4f} "
        f"{m.get('subject_dep_f1', 0):>11.4f} "
        f"{m.get('subject_indep_f1', 0):>11.4f} "
        f"{m.get('role_binding_f1', 0):>12.4f} "
        f"{m.get('role_coverage', 0):>8.4f} "
        f"{m.get('actor_f1', 0):>10.4f}"
    )


def _run_evaluation(
        gold_dir: Path,
        pred_dir: Path,
        threshold: float,
        label: str,
) -> List[Dict]:
    pairs = [
        (gf, pred_dir / gf.name)
        for gf in sorted(gold_dir.glob("*.xml"))
        if (pred_dir / gf.name).exists()
    ]
    if not pairs:
        print(f"  [WARNING] No matching XML pairs for '{label}' in '{pred_dir}'")
        return []

    results: List[Dict] = []
    for gf, pf in pairs:
        gold_model = parse_organigram(gf)
        pred_model = parse_organigram(pf)
        metrics = evaluate_pair(gold_model, pred_model, threshold=threshold)
        metrics["file"] = gf.name
        metrics["run_label"] = label
        results.append(metrics)
        print(_fmt_row(gf.name, metrics))

    return results


def _write_csv(
        path: Path,
        all_results: List[Dict],
        section_avgs: List[Dict],
        overall_avg: Dict,
) -> None:
    if not all_results:
        return
    base_keys = [k for k in all_results[0] if k not in ("file", "run_label", "section")]
    fieldnames = ["section", "run_label", "file"] + base_keys

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        for avg_row in section_avgs:
            writer.writerow({k: avg_row.get(k, "") for k in fieldnames})
        if overall_avg:
            writer.writerow({k: overall_avg.get(k, "") for k in fieldnames})


def _is_batch_pred(pred_path: Path) -> bool:
    for tt in _TEXT_TYPES:
        for run in _RUN_DIRS:
            if (pred_path / tt / run).is_dir():
                return True
    return False


def run_batch(
        gold_path: Path,
        pred_path: Path,
        out_path: Path,
        threshold: float,
) -> None:
    all_results: List[Dict] = []
    section_avgs: List[Dict] = []
    json_sections: Dict = {}

    print(f"\n{'=' * len(_COL_HEADER)}")
    print(f"  BATCH EVALUATION — {pred_path.name}")
    print(f"{'=' * len(_COL_HEADER)}")

    for text_type in _TEXT_TYPES:
        section_results: List[Dict] = []
        print(f"\n{'─' * len(_COL_HEADER)}")
        print(f"  SECTION: {text_type.upper()}")
        print(f"{'─' * len(_COL_HEADER)}")
        print(f"\n{_COL_HEADER}")

        for run in _RUN_DIRS:
            run_dir = pred_path / text_type / run
            if not run_dir.is_dir():
                print(f"  [SKIP] '{run_dir}' not found.")
                continue

            label = f"{text_type}/{run}"
            print(f"\n  -- {run} --")
            run_results = _run_evaluation(gold_path, run_dir, threshold, label)
            for r in run_results:
                r["section"] = text_type
            section_results.extend(run_results)

            if len(run_results) > 1:
                run_avg = _macro_avg(run_results)
                print(_fmt_row(f"  {run} MACRO AVG", run_avg))

        if section_results:
            sec_avg = _macro_avg(section_results)
            print(f"\n{_COL_SEP}")
            print(_fmt_row(f"  {text_type.upper()} OVERALL AVG", sec_avg))
            section_avgs.append({
                "section": text_type,
                "run_label": "SECTION_AVG",
                "file": "SECTION_MACRO_AVG",
                **sec_avg,
            })
            all_results.extend(section_results)
            json_sections[text_type] = {
                "per_file": section_results,
                "section_macro_avg": sec_avg,
            }

    overall_avg_row: Dict = {}
    if all_results:
        overall_avg = _macro_avg(all_results)
        print(f"\n{'=' * len(_COL_HEADER)}")
        print(_fmt_row("  OVERALL MACRO AVERAGE", overall_avg))
        print(f"{'=' * len(_COL_HEADER)}\n")
        overall_avg_row = {
            "section": "ALL",
            "run_label": "OVERALL_AVG",
            "file": "OVERALL_MACRO_AVG",
            **overall_avg,
        }

    _write_csv(out_path, all_results, section_avgs, overall_avg_row)
    print(f"Results saved to: {out_path}")

    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sections": json_sections,
                **({"overall_macro_avg": _macro_avg(all_results)} if all_results else {}),
            },
            f,
            indent=2,
        )
    print(f"JSON summary saved to: {summary_path}")


def run_single(
        gold_path: Path,
        pred_path: Path,
        out_path: Path,
        threshold: float,
) -> None:
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

    print(f"\n{_COL_HEADER}")
    print(_COL_SEP)

    for gf, pf in pairs:
        gold_model = parse_organigram(gf)
        pred_model = parse_organigram(pf)
        metrics = evaluate_pair(gold_model, pred_model, threshold=threshold)
        metrics["file"] = gf.name
        all_results.append(metrics)
        print(_fmt_row(gf.name, metrics))

    if len(all_results) > 1:
        avg = _macro_avg(all_results)
        print(_COL_SEP)
        print(_fmt_row("MACRO AVERAGE", avg))

    base_keys = [k for k in all_results[0] if k != "file"]
    fieldnames = ["file"] + base_keys
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
        if len(all_results) > 1:
            avg_row: Dict = {"file": "MACRO_AVG", **_macro_avg(all_results)}
            writer.writerow({k: avg_row.get(k, "") for k in fieldnames})

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated organigrams against gold standard."
    )
    parser.add_argument(
        "--gold", required=True,
        help="Gold standard XML file or directory.",
    )
    parser.add_argument(
        "--pred", required=True,
        help=(
            "Predicted XML file, run directory, OR model root directory. "
            "Model root mode expects <pred>/<text_type>/run_<N>/ structure "
            "(text types: preprocessed_text, raw_text; runs: run_1–run_3)."
        ),
    )
    parser.add_argument(
        "--out", default="evaluation_results.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--threshold", type=float, default=_DEFAULT_THRESHOLD,
        help=f"Flat entity spaCy similarity threshold (default: {_DEFAULT_THRESHOLD}). "
             f"Independent binding / role-only / coverage use a fixed {_BINDING_THRESHOLD} threshold.",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    pred_path = Path(args.pred)
    out_path = Path(args.out)

    if not gold_path.exists():
        print(f"ERROR: gold path '{gold_path}' does not exist.")
        return
    if not pred_path.exists():
        print(f"ERROR: pred path '{pred_path}' does not exist.")
        return

    if _is_batch_pred(pred_path):
        run_batch(gold_path, pred_path, out_path, args.threshold)
    else:
        run_single(gold_path, pred_path, out_path, args.threshold)


if __name__ == "__main__":
    main()
