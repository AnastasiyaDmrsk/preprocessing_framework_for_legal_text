import argparse
import re
import sys
import pandas as pd

METRIC_COLS = ["gold_total_tokens", "pred_total_tokens", "gold_total_sentences", "pred_total_sentences",
    "actor_precision", "actor_recall", "actor_f1", "gold_xor_gateway_count", "pred_xor_gateway_count",
    "gold_and_gateway_count", "pred_and_gateway_count", "gold_task_count", "pred_task_count", "matched_tasks",
    "avg_task_similarity", "task_precision", "task_recall", "task_f1", "deontic_precision", "deontic_recall",
    "deontic_f1", ]


def _doc_number(filename: str) -> int:
    m = re.match(r"(\d+)", filename)
    return int(m.group(1)) if m else 0


def _run_number(run_label: str) -> str:
    m = re.search(r"run_(\d+)", run_label)
    return m.group(1) if m else run_label


def _prf(row: pd.Series, p: str, r: str, f: str) -> str:
    return f"{row[p]:.2f}/{row[r]:.2f}/{row[f]:.2f}"


def _f(row: pd.Series, col: str) -> str:
    return f"{row[col]:.2f}"


def _ratio(pred_val: float, gold_val: float) -> str:
    return f"{int(round(pred_val))}/{int(round(gold_val))}"


def _format_row(doc_label: str, row: pd.Series) -> str:
    cells = [doc_label, str(int(round(row["gold_total_tokens"]))), str(int(round(row["pred_total_tokens"]))),
        str(int(round(row["gold_total_sentences"]))), str(int(round(row["pred_total_sentences"]))),
        _prf(row, "actor_precision", "actor_recall", "actor_f1"),
        _ratio(row["pred_xor_gateway_count"], row["gold_xor_gateway_count"]),
        _ratio(row["pred_and_gateway_count"], row["gold_and_gateway_count"]), str(int(round(row["gold_task_count"]))),
        str(int(round(row["pred_task_count"]))), str(int(round(row["matched_tasks"]))), _f(row, "avg_task_similarity"),
        _prf(row, "task_precision", "task_recall", "task_f1"),
        _prf(row, "deontic_precision", "deontic_recall", "deontic_f1"), ]
    return " & ".join(cells) + r" \\"


def _format_avg_row(row: pd.Series) -> str:
    cells = [r"\textbf{Ø}", f"{row['gold_total_tokens']:.0f}", f"{row['pred_total_tokens']:.0f}",
        f"{row['gold_total_sentences']:.0f}", f"{row['pred_total_sentences']:.0f}",
        f"\\textbf{{{_prf(row, 'actor_precision', 'actor_recall', 'actor_f1')}}}",
        f"\\textbf{{{row['pred_xor_gateway_count']:.1f}/{row['gold_xor_gateway_count']:.1f}}}",
        f"\\textbf{{{row['pred_and_gateway_count']:.1f}/{row['gold_and_gateway_count']:.1f}}}",
        f"{row['gold_task_count']:.0f}", f"{row['pred_task_count']:.0f}", f"{row['matched_tasks']:.0f}",
        f"\\textbf{{{row['avg_task_similarity']:.2f}}}",
        f"\\textbf{{{_prf(row, 'task_precision', 'task_recall', 'task_f1')}}}",
        f"\\textbf{{{_prf(row, 'deontic_precision', 'deontic_recall', 'deontic_f1')}}}", ]
    return " & ".join(cells) + r" \\"


HEADER = r"""\begin{table}[h]
\centering
\caption{%(caption)s}
\label{%(label)s}
\resizebox{\textwidth}{!}{
\begin{tabular}{l cc cc c cc ccc c c c}
\toprule
\multirow{2}{*}{\textbf{Doc}} 
& \multicolumn{2}{c}{\textbf{\# Tokens}}
& \multicolumn{2}{c}{\textbf{\# Sentences}}
& \multirow{2}{*}{\textbf{Actor P/R/F1}}
& \textbf{XOR} & \textbf{AND}
& \multicolumn{3}{c}{\textbf{Tasks}}
& \multirow{2}{*}{\textbf{Sim}}
& \multirow{2}{*}{\textbf{Task P/R/F1}}
& \multirow{2}{*}{\textbf{Deontic P/R/F1}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){9-11}
& \textbf{G} & \textbf{P} & \textbf{G} & \textbf{P} & & \textbf{P/G} & \textbf{P/G} & \textbf{G} & \textbf{P} & \textbf{M} & & & \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}
}
\end{table}"""


def _load_clean(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data[data["file"] != "SECTION_MACRO_AVG"]
    data = data[~data["run_label"].str.contains("SECTION_AVG", na=False)]
    data["_doc_num"] = data["file"].apply(_doc_number)
    data["_run_num"] = data["run_label"].apply(_run_number)
    return data.sort_values(["_doc_num", "_run_num"]).reset_index(drop=True)


def method1_all_runs(df: pd.DataFrame) -> str:
    data = _load_clean(df)
    if data.empty:
        return "% No data rows found."

    doc_nums = sorted(data["_doc_num"].unique())
    doc_index = {n: i + 1 for i, n in enumerate(doc_nums)}

    lines = [HEADER % {"caption": "Process Description Evaluation --- All Runs", "label": "tab:proc_desc_all_runs", }]

    for _, row in data.iterrows():
        seq_idx = doc_index[row["_doc_num"]]
        doc_label = f"{seq_idx} (run {row['_run_num']})"
        lines.append(_format_row(doc_label, row))

    lines.append(r"\midrule")
    avg = data[METRIC_COLS].mean()
    lines.append(_format_avg_row(avg))
    lines.append(FOOTER)
    return "\n".join(lines)


def method2_averaged(df: pd.DataFrame) -> str:
    data = _load_clean(df)
    if data.empty:
        return "% No data rows found."

    data = data[data["_run_num"].isin(["1", "2", "3"])]

    doc_nums = sorted(data["_doc_num"].unique())
    doc_index = {n: i + 1 for i, n in enumerate(doc_nums)}

    grouped = (data.groupby("_doc_num", sort=True)[METRIC_COLS].mean().reset_index())

    lines = [
        HEADER % {"caption": r"Process Description Evaluation --- avg.\ over 3 runs", "label": "tab:proc_desc_avg", }]

    for _, row in grouped.iterrows():
        seq_idx = doc_index[row["_doc_num"]]
        lines.append(_format_row(str(seq_idx), row))

    lines.append(r"\midrule")
    avg = grouped[METRIC_COLS].mean()
    lines.append(_format_avg_row(avg))
    lines.append(FOOTER)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert process description evaluation CSV to LaTeX.")
    parser.add_argument("csv_file")
    parser.add_argument("--method", choices=["1", "2", "both"], default="both")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: '{args.csv_file}' not found.", file=sys.stderr)
        sys.exit(1)

    required = {"section", "run_label", "file", "gold_total_tokens", "pred_total_tokens", "gold_total_sentences",
        "pred_total_sentences", "actor_precision", "actor_recall", "actor_f1", "gold_xor_gateway_count",
        "pred_xor_gateway_count", "gold_and_gateway_count", "pred_and_gateway_count", "gold_task_count",
        "pred_task_count", "matched_tasks", "avg_task_similarity", "task_precision", "task_recall", "task_f1",
        "deontic_precision", "deontic_recall", "deontic_f1", }
    missing = required - set(df.columns)
    if missing:
        print(f"Error: missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.method in ("1", "both"):
        print(f"\n% Per-run table\n")
        print(method1_all_runs(df))

    if args.method in ("2", "both"):
        print(f"\n%Averaged over runs table\n")
        print(method2_averaged(df))


if __name__ == "__main__":
    main()
