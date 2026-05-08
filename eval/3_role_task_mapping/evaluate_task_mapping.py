from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

_DEFAULT_THRESHOLD = 0.80

try:
    NLP = spacy.load("en_core_web_md")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_md' not found.\n"
        "Install with: python -m spacy download en_core_web_md"
    )


@dataclass
class Performer:
    unit: str
    role: str


@dataclass
class Deontic:
    type: str
    modality: str


@dataclass
class SourceRef:
    article: str
    paragraph: str


@dataclass
class Task:
    id: str
    label: str
    performers: List[Performer] = field(default_factory=list)
    deontic: Optional[Deontic] = None
    conditions: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    source_ref: Optional[SourceRef] = None


@dataclass
class TaskModel:
    tasks: List[Task] = field(default_factory=list)


def parse_task_mapping(path: Path) -> TaskModel:
    tree = ET.parse(str(path))
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    def _t(local: str) -> str:
        return f"{ns}{local}"

    model = TaskModel()

    for task_el in root.findall(f".//{_t('task')}"):
        tid = task_el.get("id", "").strip()

        label_el = task_el.find(_t("label"))
        label = label_el.text.strip() if label_el is not None and label_el.text else ""
        if not label:
            continue

        task = Task(id=tid, label=label)

        # --- Performers ---
        performers_el = task_el.find(_t("performers"))
        if performers_el is not None:
            for perf_el in performers_el.findall(_t("performer")):
                u_el = perf_el.find(_t("unit"))
                r_el = perf_el.find(_t("role"))
                u = u_el.text.strip() if u_el is not None and u_el.text else ""
                r = r_el.text.strip() if r_el is not None and r_el.text else ""
                if u or r:
                    task.performers.append(Performer(unit=u, role=r))

        # --- Deontic ---
        deontic_el = task_el.find(_t("deontic"))
        if deontic_el is not None:
            task.deontic = Deontic(
                type=deontic_el.get("type", "").strip().lower(),
                modality=deontic_el.get("modality", "").strip().lower(),
            )

        # --- Conditions ---
        conditions_el = task_el.find(_t("conditions"))
        if conditions_el is not None:
            for cond_el in conditions_el.findall(_t("condition")):
                text = cond_el.text.strip() if cond_el.text else ""
                if text:
                    task.conditions.append(text)

        # --- Exceptions ---
        exceptions_el = task_el.find(_t("exceptions"))
        if exceptions_el is not None:
            for exc_el in exceptions_el:
                desc = exc_el.get("description", "").strip()
                if not desc and exc_el.text:
                    desc = exc_el.text.strip()
                if desc:
                    task.exceptions.append(desc)

        # --- Source reference ---
        src_el = task_el.find(_t("source-ref"))
        if src_el is not None:
            task.source_ref = SourceRef(
                article=src_el.get("article", "").strip(),
                paragraph=src_el.get("paragraph", "").strip(),
            )

        model.tasks.append(task)

    return model


def _similarity(a: str, b: str) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0

    al, bl = a.lower(), b.lower()
    if al == bl:
        return 1.0
    if al in bl or bl in al:
        return 0.92

    da, db = NLP(al), NLP(bl)
    if da.has_vector and db.has_vector:
        return float(da.similarity(db))
    return 0.0


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _match_tasks(
        pred_tasks: List[Task], gold_tasks: List[Task], threshold: float,
) -> List[Tuple[int, int, float]]:
    scored: List[Tuple[float, int, int]] = []
    for i, pt in enumerate(pred_tasks):
        for j, gt in enumerate(gold_tasks):
            sim = _similarity(pt.label, gt.label)
            if sim >= threshold:
                scored.append((sim, i, j))

    scored.sort(key=lambda x: x[0], reverse=True)
    matched: List[Tuple[int, int, float]] = []
    used_pred: Set[int] = set()
    used_gold: Set[int] = set()

    for sim, i, j in scored:
        if i not in used_pred and j not in used_gold:
            matched.append((i, j, sim))
            used_pred.add(i)
            used_gold.add(j)

    return matched


def _evaluate_performers(
        pred_perfs: List[Performer], gold_perfs: List[Performer], threshold: float,
) -> Tuple[int, int, int, float]:
    scored: List[Tuple[float, int, int]] = []
    for i, pp in enumerate(pred_perfs):
        for j, gp in enumerate(gold_perfs):
            u_sim = _similarity(pp.unit, gp.unit)
            r_sim = _similarity(pp.role, gp.role)
            if u_sim >= threshold and r_sim >= threshold:
                scored.append((min(u_sim, r_sim), i, j))

    scored.sort(key=lambda x: x[0], reverse=True)
    matched_p: Set[int] = set()
    matched_g: Set[int] = set()
    tp = 0
    for _, i, j in scored:
        if i not in matched_p and j not in matched_g:
            tp += 1
            matched_p.add(i)
            matched_g.add(j)

    fp = len(pred_perfs) - tp
    fn = len(gold_perfs) - tp

    if len(gold_perfs) > 0:
        correctness = tp / len(gold_perfs)
    else:
        correctness = 1.0 if len(pred_perfs) == 0 else 0.0

    return tp, fp, fn, round(correctness, 4)


def _evaluate_deontic(
        pred_deontic: Optional[Deontic], gold_deontic: Optional[Deontic],
) -> Tuple[int, int, int]:
    if gold_deontic is None and pred_deontic is None:
        return 1, 0, 0
    if gold_deontic is None and pred_deontic is not None:
        return 0, 1, 0
    if gold_deontic is not None and pred_deontic is None:
        return 0, 0, 1
    if (pred_deontic.type == gold_deontic.type
            and pred_deontic.modality == gold_deontic.modality):
        return 1, 0, 0
    else:
        return 0, 1, 1


def _evaluate_text_list(
        pred_texts: List[str], gold_texts: List[str], threshold: float,
) -> Tuple[int, int, int]:
    scored: List[Tuple[float, int, int]] = []
    for i, pt in enumerate(pred_texts):
        for j, gt in enumerate(gold_texts):
            sim = _similarity(pt, gt)
            if sim >= threshold:
                scored.append((sim, i, j))

    scored.sort(key=lambda x: x[0], reverse=True)
    matched_p: Set[int] = set()
    matched_g: Set[int] = set()
    tp = 0
    for _, i, j in scored:
        if i not in matched_p and j not in matched_g:
            tp += 1
            matched_p.add(i)
            matched_g.add(j)

    fp = len(pred_texts) - tp
    fn = len(gold_texts) - tp
    return tp, fp, fn


def _evaluate_source_ref(
        pred_ref: Optional[SourceRef], gold_ref: Optional[SourceRef],
) -> Tuple[int, int, int]:
    if gold_ref is None and pred_ref is None:
        return 1, 0, 0
    if gold_ref is None and pred_ref is not None:
        return 0, 1, 0
    if gold_ref is not None and pred_ref is None:
        return 0, 0, 1
    if (pred_ref.article == gold_ref.article
            and pred_ref.paragraph == gold_ref.paragraph):
        return 1, 0, 0
    else:
        return 0, 1, 1


def evaluate_pair(
        gold: TaskModel, pred: TaskModel, threshold: float = _DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    matches = _match_tasks(pred.tasks, gold.tasks, threshold)

    task_tp = len(matches)
    task_fp = len(pred.tasks) - task_tp
    task_fn = len(gold.tasks) - task_tp
    task_p, task_r, task_f1 = _prf(task_tp, task_fp, task_fn)

    micro = {
        "performer": [0, 0, 0],
        "deontic": [0, 0, 0],
        "condition": [0, 0, 0],
        "exception": [0, 0, 0],
        "source_ref": [0, 0, 0],
    }

    per_pair: List[Dict[str, Tuple[float, float, float]]] = []
    performer_correctness_scores = []

    for pred_idx, gold_idx, _ in matches:
        pt = pred.tasks[pred_idx]
        gt = gold.tasks[gold_idx]
        pair_metrics: Dict[str, Tuple[float, float, float]] = {}

        tp, fp, fn, p_corr = _evaluate_performers(pt.performers, gt.performers, threshold)
        micro["performer"][0] += tp;
        micro["performer"][1] += fp;
        micro["performer"][2] += fn
        pair_metrics["performer"] = _prf(tp, fp, fn)
        performer_correctness_scores.append(p_corr)

        tp, fp, fn = _evaluate_deontic(pt.deontic, gt.deontic)
        micro["deontic"][0] += tp;
        micro["deontic"][1] += fp;
        micro["deontic"][2] += fn
        pair_metrics["deontic"] = _prf(tp, fp, fn)

        tp, fp, fn = _evaluate_text_list(pt.conditions, gt.conditions, threshold)
        micro["condition"][0] += tp;
        micro["condition"][1] += fp;
        micro["condition"][2] += fn
        pair_metrics["condition"] = _prf(tp, fp, fn)

        tp, fp, fn = _evaluate_text_list(pt.exceptions, gt.exceptions, threshold)
        micro["exception"][0] += tp;
        micro["exception"][1] += fp;
        micro["exception"][2] += fn
        pair_metrics["exception"] = _prf(tp, fp, fn)

        tp, fp, fn = _evaluate_source_ref(pt.source_ref, gt.source_ref)
        micro["source_ref"][0] += tp;
        micro["source_ref"][1] += fp;
        micro["source_ref"][2] += fn
        pair_metrics["source_ref"] = _prf(tp, fp, fn)

        per_pair.append(pair_metrics)

    matched_gold_idxs = {g for _, g, _ in matches}
    for j, gt in enumerate(gold.tasks):
        if j in matched_gold_idxs: continue
        micro["performer"][2] += len(gt.performers)
        micro["deontic"][2] += 1 if gt.deontic is not None else 0
        micro["condition"][2] += len(gt.conditions)
        micro["exception"][2] += len(gt.exceptions)
        micro["source_ref"][2] += 1 if gt.source_ref is not None else 0
        performer_correctness_scores.append(0.0)

    matched_pred_idxs = {p for p, _, _ in matches}
    for i, pt in enumerate(pred.tasks):
        if i in matched_pred_idxs: continue
        micro["performer"][1] += len(pt.performers)
        micro["deontic"][1] += 1 if pt.deontic is not None else 0
        micro["condition"][1] += len(pt.conditions)
        micro["exception"][1] += len(pt.exceptions)
        micro["source_ref"][1] += 1 if pt.source_ref is not None else 0

    result: Dict[str, Any] = {
        "gold_tasks": len(gold.tasks),
        "pred_tasks": len(pred.tasks),
        "matched_tasks": task_tp,
        "task_precision": task_p,
        "task_recall": task_r,
        "task_f1": task_f1,
    }

    avg_performer_correctness = (
        sum(performer_correctness_scores) / len(performer_correctness_scores)
        if performer_correctness_scores else 0.0
    )
    result["performer_correctness"] = round(avg_performer_correctness, 4)

    for key in ("performer", "deontic", "condition", "exception", "source_ref"):
        tp, fp, fn = micro[key]
        p, r, f = _prf(tp, fp, fn)
        result[f"{key}_micro_precision"] = p
        result[f"{key}_micro_recall"] = r
        result[f"{key}_micro_f1"] = f

    if per_pair:
        for key in ("performer", "deontic", "condition", "exception", "source_ref"):
            vals = [pp[key] for pp in per_pair]
            result[f"{key}_macro_precision"] = round(sum(v[0] for v in vals) / len(vals), 4)
            result[f"{key}_macro_recall"] = round(sum(v[1] for v in vals) / len(vals), 4)
            result[f"{key}_macro_f1"] = round(sum(v[2] for v in vals) / len(vals), 4)
    else:
        for key in ("performer", "deontic", "condition", "exception", "source_ref"):
            result[f"{key}_macro_precision"] = 0.0
            result[f"{key}_macro_recall"] = 0.0
            result[f"{key}_macro_f1"] = 0.0

    return result


def _macro_avg(results: List[Dict]) -> Dict:
    if not results: return {}
    # Ensure we only try to average numeric metrics (int or float)
    num_keys = [k for k, v in results[0].items() if isinstance(v, (int, float))]
    avg_dict = {}
    for k in num_keys:
        vals = [r[k] for r in results if isinstance(r.get(k), (int, float))]
        if vals:
            avg_dict[k] = round(sum(vals) / len(vals), 4)
    return avg_dict


_CONSOLE_COLS = (
    ("Task-F1", "task_f1"),
    ("Perf-Corr", "performer_correctness"),
    ("Deont-F1", "deontic_micro_f1"),
    ("Cond-F1", "condition_micro_f1"),
    ("Excep-F1", "exception_micro_f1"),
)


def _print_header() -> str:
    parts = [f"{'File':<50}"]
    for label, _ in _CONSOLE_COLS:
        parts.append(f"{label:>10}")
    line = " ".join(parts)
    print(f"\n{line}")
    print("-" * len(line))
    return line


def _print_row(name: str, metrics: Dict) -> None:
    parts = [f"  {name:<48}"]
    for _, key in _CONSOLE_COLS:
        parts.append(f"{metrics.get(key, 0):>10.4f}")
    print(" ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated role-task mappings against gold standard.")
    parser.add_argument("--gold", required=True, help="Gold standard XML file OR directory.")
    parser.add_argument("--pred", required=True, help="Predicted XML file OR directory.")
    parser.add_argument("--out", default="evaluation_results.csv", help="Output CSV file path.")
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD,
                        help=f"Similarity threshold (default: {_DEFAULT_THRESHOLD}).")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    pred_path = Path(args.pred)
    pairs = []

    if gold_path.is_file() and pred_path.is_file():
        pairs = [(gold_path, pred_path)]
    elif gold_path.is_dir() and pred_path.is_dir():
        for gf in sorted(gold_path.glob("*.xml")):
            matched_preds = list(pred_path.rglob(gf.name))
            for pf in matched_preds:
                pairs.append((gf, pf))
        if not pairs:
            print(f"No identically named XML pairs found between '{gold_path}' and recursive '{pred_path}'.")
            return
    else:
        print("ERROR: --gold and --pred must both be files or both be directories.")
        return

    all_results: List[Dict] = []
    header_line = _print_header()

    for gf, pf in pairs:
        gold_model = parse_task_mapping(gf)
        pred_model = parse_task_mapping(pf)
        metrics = evaluate_pair(gold_model, pred_model, threshold=args.threshold)

        if pred_path.is_dir():
            rel_parts = pf.relative_to(pred_path).parts
            if len(rel_parts) >= 3:
                section = rel_parts[0]
                run_label = f"{rel_parts[0]}/{rel_parts[1]}"
                file_name = rel_parts[-1]
            elif len(rel_parts) == 2:
                section = rel_parts[0]
                run_label = rel_parts[0]
                file_name = rel_parts[-1]
            else:
                section = "default_section"
                run_label = "default_run"
                file_name = pf.name
        else:
            section = "default_section"
            run_label = "default_run"
            file_name = pf.name

        metrics["section"] = section
        metrics["run_label"] = run_label
        metrics["file"] = file_name

        all_results.append(metrics)

        rel_path = str(pf.relative_to(pred_path)) if pred_path.is_dir() else pf.name
        _print_row(rel_path, metrics)

    sections = list(dict.fromkeys(r["section"] for r in all_results))
    final_results = []

    for sec in sections:
        sec_results = [r for r in all_results if r["section"] == sec]
        final_results.extend(sec_results)

        if len(sec_results) > 1:
            sec_avg = _macro_avg(sec_results)
            avg_row = {
                "section": sec,
                "run_label": "SECTION_AVG",
                "file": "SECTION_MACRO_AVG",
                **sec_avg
            }
            final_results.append(avg_row)

    if len(all_results) > 1:
        overall_avg = _macro_avg(all_results)
        print("-" * len(header_line))
        _print_row("OVERALL MACRO AVERAGE", overall_avg)

    out_path = Path(args.out)
    if final_results:
        metric_keys = [k for k in final_results[0].keys() if k not in ("section", "run_label", "file")]
        fieldnames = ["section", "run_label", "file"] + metric_keys

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_results)

    print(f"\nResults saved to: {out_path}")
    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_file": final_results,
                **({"overall_macro_avg": _macro_avg(all_results)} if len(all_results) > 1 else {}),
            },
            f, indent=2,
        )
    print(f"JSON summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
