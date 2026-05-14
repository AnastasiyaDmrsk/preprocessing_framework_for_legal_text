from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import nltk
import numpy as np
import spacy
from scipy.optimize import linear_sum_assignment

for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

try:
    NLP = spacy.load("en_core_web_md")
except OSError:
    raise RuntimeError("spaCy model 'en_core_web_md' not found.\n"
                       "Install with: python -m spacy download en_core_web_md")

_DEFAULT_THRESHOLD = 0.80


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return (precision, recall, F1) from confusion counts."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _similarity(a: str, b: str) -> float:
    """Semantic similarity via spaCy word vectors with fast-path shortcuts."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    al, bl = a.lower().strip(), b.lower().strip()
    if al == bl:
        return 1.0
    if al in bl or bl in al:
        return 0.92
    da, db = NLP(al), NLP(bl)
    if da.has_vector and db.has_vector:
        return float(da.similarity(db))
    return 0.0


def _hungarian_match(pred_list: List[str], gold_list: List[str], threshold: float) -> List[Tuple[int, int, float]]:
    """
    Globally optimal 1-to-1 bipartite matching via the Hungarian algorithm.
    Returns [(pred_idx, gold_idx, similarity), …] for pairs above *threshold*.
    """
    if not pred_list or not gold_list:
        return []

    sim_matrix = np.zeros((len(pred_list), len(gold_list)))
    for i, p in enumerate(pred_list):
        for j, g in enumerate(gold_list):
            sim_matrix[i, j] = _similarity(p, g)

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)  # negate → maximise

    return [(int(i), int(j), float(sim_matrix[i, j])) for i, j in zip(row_ind, col_ind) if
        sim_matrix[i, j] >= threshold]


def compute_token_stats(text: str) -> Dict[str, Any]:
    tokens = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    sent_lens = [len(nltk.word_tokenize(s)) for s in sentences]
    avg = sum(sent_lens) / len(sent_lens) if sent_lens else 0.0
    return {"total_tokens": len(tokens), "total_sentences": len(sentences), "avg_tokens_per_sentence": round(avg, 4), }


_ACTORS_RE = re.compile(r"actors?\s*:\s*(.+?)\.?\s*$", re.IGNORECASE)


def extract_actors(text: str) -> List[str]:
    """
    Parse the first sentence to extract actor names from
    "The process contains the following actors: A, B, C, and D."
    """
    first = nltk.sent_tokenize(text)[0] if text.strip() else ""
    m = _ACTORS_RE.search(first)
    if not m:
        return []
    parts = re.split(r",\s*|\s+and\s+", m.group(1), flags=re.IGNORECASE)
    return [p.strip(" .'\"") for p in parts if p.strip(" .'\"")]


def evaluate_actors(gold: List[str], pred: List[str], threshold: float) -> Dict[str, Any]:
    matches = _hungarian_match(pred, gold, threshold)
    tp = len(matches)
    fp, fn = len(pred) - tp, len(gold) - tp
    p, r, f1 = _prf(tp, fp, fn)
    return {"gold_actor_count": len(gold), "pred_actor_count": len(pred), "actor_precision": p, "actor_recall": r,
        "actor_f1": f1, }


_XOR_PATTERNS = [re.compile(r"\bif\b", re.IGNORECASE),
    re.compile(r"\bunless\b", re.IGNORECASE),
    re.compile(r"\bwhether\b", re.IGNORECASE), re.compile(r"\b(?<!\w)OR(?!\w)\b"),
    re.compile(r"\botherwise\b", re.IGNORECASE), ]

_AND_PATTERNS = [re.compile(r"\bin parallel\b", re.IGNORECASE), re.compile(r"\bsimultaneously\b", re.IGNORECASE),
    re.compile(r"\bactivities occur in parallel\b", re.IGNORECASE),
    re.compile(r"\bactivities? in parallel\b", re.IGNORECASE)]


def count_gateways(text: str) -> Dict[str, int]:
    return {"xor_gateway_count": sum(len(p.findall(text)) for p in _XOR_PATTERNS),
        "and_gateway_count": sum(len(p.findall(text)) for p in _AND_PATTERNS), }


def evaluate_gateways(gold: Dict[str, int], pred: Dict[str, int]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for key in ("xor_gateway_count", "and_gateway_count"):
        g, p = gold[key], pred[key]
        res[f"gold_{key}"] = g
        res[f"pred_{key}"] = p
        res[f"{key}_ratio"] = round(min(g, p) / max(g, p) if max(g, p) > 0 else 1.0, 4)
    return res


_TASK_RE = re.compile(r"\b(?P<actor>[A-Z][A-Za-z\s\-/()]+?)\s+"
                      r"(?P<deontic>shall not|may not|shall|may|must|should)\s+"
                      r"(?P<activity>.+?)(?=[;.\n]|$)", re.MULTILINE, )


def _split_clauses(text: str) -> List[str]:
    """
    Flatten text into candidate task strings by splitting on
    sentences → semicolons → bullet / dash markers.
    """
    candidates: List[str] = []
    for sent in nltk.sent_tokenize(text):
        for part in re.split(r";\s*", sent):
            for sub in re.split(r"\n\s*[-•*]\s+", part):
                if sub.strip():
                    candidates.append(sub.strip())
    return candidates


def extract_tasks(text: str) -> List[Dict[str, str]]:
    """
    Return one dict per task clause: actor / deontic / activity / full.
    Deduplication is applied on (actor, deontic, first-60-chars-of-activity).
    """
    tasks: List[Dict[str, str]] = []
    seen: Set[tuple] = set()

    for candidate in _split_clauses(text):
        for m in _TASK_RE.finditer(candidate):
            actor = m.group("actor").strip()
            deontic = m.group("deontic").strip().lower()
            activity = m.group("activity").strip()

            key = (actor.lower(), deontic, activity[:60].lower())
            if key in seen:
                continue
            seen.add(key)

            tasks.append({"actor": actor, "deontic": deontic, "activity": activity, "full": candidate, })
    return tasks


def evaluate_tasks(gold: List[Dict], pred: List[Dict], threshold: float) -> Dict[str, Any]:
    gold_act = [t["activity"] for t in gold]
    pred_act = [t["activity"] for t in pred]

    matches = _hungarian_match(pred_act, gold_act, threshold)
    tp = len(matches)
    fp, fn = len(pred) - tp, len(gold) - tp
    t_p, t_r, t_f1 = _prf(tp, fp, fn)
    avg_sim = round(sum(m[2] for m in matches) / tp if tp > 0 else 0.0, 4)

    d_tp = d_fp = d_fn = 0
    a_tp = a_fp = a_fn = 0

    matched_pred_idx = {m[0] for m in matches}
    matched_gold_idx = {m[1] for m in matches}

    for pred_i, gold_j, _ in matches:
        pt, gt = pred[pred_i], gold[gold_j]

        if pt["deontic"] == gt["deontic"]:
            d_tp += 1
        else:
            d_fp += 1
            d_fn += 1

        if _similarity(pt["actor"], gt["actor"]) >= threshold:
            a_tp += 1
        else:
            a_fp += 1
            a_fn += 1

    unmatched_gold = len(gold) - len(matched_gold_idx)
    unmatched_pred = len(pred) - len(matched_pred_idx)
    d_fn += unmatched_gold
    a_fn += unmatched_gold
    d_fp += unmatched_pred
    a_fp += unmatched_pred

    d_p, d_r, d_f1 = _prf(d_tp, d_fp, d_fn)

    return {"gold_task_count": len(gold), "pred_task_count": len(pred), "matched_tasks": tp,
        "avg_task_similarity": avg_sim, "task_precision": t_p, "task_recall": t_r,
        "task_f1": t_f1, "deontic_precision": d_p, "deontic_recall": d_r, "deontic_f1": d_f1}


def evaluate_pair(gold_text: str, pred_text: str, threshold: float = _DEFAULT_THRESHOLD) -> Dict[str, Any]:
    res: Dict[str, Any] = {}

    # Surface stats reported separately for gold and pred
    for prefix, text in (("gold", gold_text), ("pred", pred_text)):
        for k, v in compute_token_stats(text).items():
            res[f"{prefix}_{k}"] = v

    res.update(evaluate_actors(extract_actors(gold_text), extract_actors(pred_text), threshold))
    res.update(evaluate_gateways(count_gateways(gold_text), count_gateways(pred_text)))
    res.update(evaluate_tasks(extract_tasks(gold_text), extract_tasks(pred_text), threshold))
    return res


_CONSOLE_COLS: List[Tuple[str, str]] = [("Task-F1", "task_f1"), ("Task-Sim", "avg_task_similarity"),
    ("Deont-F1", "deontic_f1"), ("Actor-F1", "actor_f1"),
    ("XOR-ratio", "xor_gateway_count_ratio"), ("AND-ratio", "and_gateway_count_ratio"), ]


def _print_header() -> int:
    line = f"{'File':<40} " + " ".join(f"{lbl:>10}" for lbl, _ in _CONSOLE_COLS)
    print(f"\n{line}\n" + "-" * len(line))
    return len(line)


def _print_row(name: str, metrics: Dict[str, Any]) -> None:
    vals = " ".join(f"{metrics.get(k, 0.0):>10.4f}" for _, k in _CONSOLE_COLS)
    print(f"  {name[:38]:<38} {vals}")


def _macro_avg(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    num_keys = [k for k, v in results[0].items() if isinstance(v, (int, float))]
    return {k: round(sum(r[k] for r in results if isinstance(r.get(k), (int, float))) / len(results), 4, ) for k in
        num_keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated process descriptions against gold standard.")
    parser.add_argument("--gold", required=True, help="Gold-standard .txt file OR directory of .txt files.")
    parser.add_argument("--pred", required=True, help="Predicted .txt file OR directory of .txt files.")
    parser.add_argument("--out", default="evaluation_results_process_description.csv",
                        help="Output CSV path (default: evaluation_results_process_description.csv).")
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD,
                        help=f"Similarity threshold (default: {_DEFAULT_THRESHOLD}).")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    pred_path = Path(args.pred)
    pairs: List[Tuple[Path, Path]] = []

    if gold_path.is_file() and pred_path.is_file():
        pairs = [(gold_path, pred_path)]
    elif gold_path.is_dir() and pred_path.is_dir():
        for gf in sorted(gold_path.glob("*.txt")):
            pairs.extend((gf, pf) for pf in pred_path.rglob(gf.name))
        if not pairs:
            print(f"No identically named .txt pairs found between "
                  f"'{gold_path}' and '{pred_path}'.")
            return
    else:
        print("ERROR: --gold and --pred must both be files or both be directories.")
        return

    all_results: List[Dict[str, Any]] = []
    width = _print_header()

    for gf, pf in pairs:
        metrics = evaluate_pair(gf.read_text(encoding="utf-8"), pf.read_text(encoding="utf-8"),
            threshold=args.threshold, )

        if pred_path.is_dir():
            rel_parts = pf.relative_to(pred_path).parts
            if len(rel_parts) >= 3:
                section = rel_parts[0]
                run_label = f"{rel_parts[0]}/{rel_parts[1]}"
                file_name = rel_parts[-1]
            elif len(rel_parts) == 2:
                section, run_label, file_name = rel_parts[0], rel_parts[0], rel_parts[-1]
            else:
                section, run_label, file_name = "default_section", "default_run", pf.name
        else:
            section, run_label, file_name = "default_section", "default_run", pf.name

        metrics.update(section=section, run_label=run_label, file=file_name)
        all_results.append(metrics)
        _print_row(str(pf.relative_to(pred_path)) if pred_path.is_dir() else pf.name, metrics, )

    sections = list(dict.fromkeys(r["section"] for r in all_results))
    final_results: List[Dict[str, Any]] = []

    for sec in sections:
        sec_results = [r for r in all_results if r["section"] == sec]
        final_results.extend(sec_results)
        if len(sec_results) > 1:
            final_results.append(
                {"section": sec, "run_label": "SECTION_AVG", "file": "SECTION_MACRO_AVG", **_macro_avg(sec_results), })

    if len(all_results) > 1:
        print("-" * width)
        _print_row("OVERALL MACRO AVERAGE", _macro_avg(all_results))

    out_path = Path(args.out)
    if final_results:
        meta = ["section", "run_label", "file"]
        fields = meta + [k for k in final_results[0] if k not in meta]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(final_results)
    print(f"\nResults saved to: {out_path}")

    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"per_file": final_results,
            **({"overall_macro_avg": _macro_avg(all_results)} if len(all_results) > 1 else {}), }, f, indent=2, )
    print(f"JSON summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
