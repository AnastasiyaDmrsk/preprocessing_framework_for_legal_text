import argparse
import sys
import pandas as pd

METRIC_COLS = ["task_precision", "task_recall", "task_f1", "performer_micro_precision", "performer_micro_recall",
    "performer_micro_f1", "performer_macro_precision", "performer_macro_recall", "performer_macro_f1",
    "deontic_micro_f1", "deontic_macro_f1", "condition_micro_f1", "condition_macro_f1", "exception_micro_f1",
    "exception_macro_f1", "source_ref_micro_f1", "source_ref_macro_f1", ]


def _prf(row: pd.Series, p: str, r: str, f: str) -> str:
    return f"{row[p]:.2f}/{row[r]:.2f}/{row[f]:.2f}"


def _f(row: pd.Series, col: str) -> str:
    return f"{row[col]:.2f}"


def _format_row(row: pd.Series, doc_label: str, bold: bool = False) -> str:
    task = _prf(row, "task_precision", "task_recall", "task_f1")
    perf_mi = _prf(row, "performer_micro_precision", "performer_micro_recall", "performer_micro_f1")
    perf_ma = _prf(row, "performer_macro_precision", "performer_macro_recall", "performer_macro_f1")
    deon_mi = _f(row, "deontic_micro_f1")
    deon_ma = _f(row, "deontic_macro_f1")
    cond_mi = _f(row, "condition_micro_f1")
    cond_ma = _f(row, "condition_macro_f1")
    exc_mi = _f(row, "exception_micro_f1")
    exc_ma = _f(row, "exception_macro_f1")
    src_mi = _f(row, "source_ref_micro_f1")
    src_ma = _f(row, "source_ref_macro_f1")

    cells = [doc_label, task, perf_mi, perf_ma, deon_mi, deon_ma, cond_mi, cond_ma, exc_mi, exc_ma, src_mi, src_ma]
    if bold:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    return " & ".join(cells) + r" \\"


def _macro_avg(df: pd.DataFrame) -> pd.Series:
    return df[METRIC_COLS].mean()


HEADER = r"""\begin{table}[h]
\centering
\caption{Role-task mapping evaluation results}
\label{%(label)s}
\resizebox{\textwidth}{!}{
\begin{tabular}{l c cc cc cc cc cc}
\toprule
\multirow{2}{*}{\textbf{Doc}} & \multirow{2}{*}{\textbf{Task P/R/F1}}
& \multicolumn{2}{c}{\textbf{Performers P/R/F1}}
& \multicolumn{2}{c}{\textbf{Deontic F1}}
& \multicolumn{2}{c}{\textbf{Conditions F1}}
& \multicolumn{2}{c}{\textbf{Exceptions F1}}
& \multicolumn{2}{c}{\textbf{Source F1}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}
& & \textbf{Micro} & \textbf{Macro} & \textbf{Micro} & \textbf{Macro} & \textbf{Micro} & \textbf{Macro} & \textbf{Micro} & \textbf{Macro} & \textbf{Micro} & \textbf{Macro} \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}
}
\end{table}"""


def method1_single_run(df: pd.DataFrame, section: str) -> str:
    subset = df[df["section"] == section].copy()
    if subset.empty:
        return f"% No data for section: {section}"

    lines = [HEADER % {"label": f"tab:rtm_eval_{section.replace(' ', '_')}", }]

    for _, row in subset.sort_values(["file", "run_label"]).iterrows():
        run_stem = row["run_label"].split("/")[-1]
        doc_label = f"{row['file'].split("_")[0]} ({run_stem.replace("_", " ")})"
        lines.append(_format_row(row, doc_label))

    lines.append(r"\midrule")
    avg = _macro_avg(subset)
    lines.append(_format_row(avg, r"\textbf{Ø}", bold=True))
    lines.append(FOOTER)
    return "\n".join(lines)


def method2_averaged(df: pd.DataFrame, section: str) -> str:
    subset = df[df["section"] == section].copy()
    if subset.empty:
        return f"% No data for section: {section}"

    run_mask = subset["run_label"].str.endswith(("run_1", "run_2", "run_3"))
    subset = subset[run_mask]

    grouped = (subset.groupby("file", sort=True)[METRIC_COLS].mean().reset_index())

    lines = [HEADER % {"label": f"tab:rtm_eval_{section.replace(' ', '_')}_avg", }]

    for _, row in grouped.iterrows():
        lines.append(_format_row(row, row["file"].split("_")[0]))

    lines.append(r"\midrule")
    avg = _macro_avg(grouped)
    lines.append(_format_row(avg, r"\textbf{Ø}", bold=True))
    lines.append(FOOTER)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert process evaluation CSV to LaTeX tables.")
    parser.add_argument("csv_file", help="Path to the evaluation CSV.")
    parser.add_argument("--section", default=None, help="Filter by section. Omit to process all sections.", )
    parser.add_argument("--method", choices=["1", "2", "both"], default="both",
        help="1=per-run, 2=averaged, both=both (default).", )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: '{args.csv_file}' not found.", file=sys.stderr)
        sys.exit(1)

    required = {"section", "run_label", "file", "task_precision", "task_recall", "task_f1", "performer_micro_precision",
        "performer_micro_recall", "performer_micro_f1", "performer_macro_precision", "performer_macro_recall",
        "performer_macro_f1", "deontic_micro_f1", "deontic_macro_f1", "condition_micro_f1", "condition_macro_f1",
        "exception_micro_f1", "exception_macro_f1", "source_ref_micro_f1", "source_ref_macro_f1", }
    missing = required - set(df.columns)
    if missing:
        print(f"Error: missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    sections = [args.section] if args.section else df["section"].unique().tolist()

    for sec in sections:
        if args.method in ("1", "both"):
            print(f"\n% Per-run table\n")
            print(method1_single_run(df, sec))

        if args.method in ("2", "both"):
            print(f"\n%Averaged over runs table\n")
            print(method2_averaged(df, sec))


if __name__ == "__main__":
    main()
