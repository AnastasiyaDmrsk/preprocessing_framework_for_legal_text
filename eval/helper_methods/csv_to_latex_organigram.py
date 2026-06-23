import argparse
import sys
import pandas as pd

METRIC_COLS = {
    "unit":         ("unit_precision",       "unit_recall",       "unit_f1"),
    "role":         ("role_precision",       "role_recall",       "role_f1"),
    "unit_hier_f1": ("unit_hier_f1",),
    "role_hier_f1": ("role_hier_f1",),
    "subject_f1":   ("subject_indep_f1",),
    "binding":      ("role_binding_precision", "role_binding_recall", "role_binding_f1"),
    "coverage":     ("role_coverage",),
    "actor":        ("actor_precision",      "actor_recall",      "actor_f1"),
}

def _prf(row: pd.Series, p_col: str, r_col: str, f_col: str) -> str:
    return f"{row[p_col]:.2f}/{row[r_col]:.2f}/{row[f_col]:.2f}"


def _f(row: pd.Series, col: str) -> str:
    return f"{row[col]:.2f}"


def _format_row(row: pd.Series, doc_label: str, bold: bool = False) -> str:
    unit    = _prf(row, "unit_precision",         "unit_recall",         "unit_f1")
    role    = _prf(row, "role_precision",         "role_recall",         "role_f1")
    u_hier  = _f(row,  "unit_hier_f1")
    r_hier  = _f(row,  "role_hier_f1")
    subj    = _f(row,  "subject_indep_f1")
    binding = _f(row, "role_binding_f1")
    cov     = _f(row,  "role_coverage")
    actor   = _prf(row, "actor_precision",        "actor_recall",        "actor_f1")

    cells = [doc_label, unit, role, u_hier, r_hier, subj, binding, cov, actor]
    if bold:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    return " & ".join(cells) + r" \\"


def _avg_row(df: pd.DataFrame) -> pd.Series:
    """Macro-average over all numeric metric columns."""
    metric_cols = [
        "unit_precision", "unit_recall", "unit_f1",
        "role_precision", "role_recall", "role_f1",
        "unit_hier_f1", "role_hier_f1",
        "subject_indep_f1", "role_binding_f1",
        "role_coverage",
        "actor_precision", "actor_recall", "actor_f1",
    ]
    return df[metric_cols].mean()


HEADER = r"""\begin{table}[h]
\centering
\caption{Organigram evaluation results}
\label{tab:eval_%(label)s}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccc}
\toprule
\textbf{Doc} & \textbf{Unit P/R/F1} & \textbf{Role P/R/F1} & \textbf{Unit H F1} & \textbf{Role H F1} & \textbf{Subject F1} & \textbf{Role Binding F1} & \textbf{Role Coverage} & \textbf{Actor P/R/F1} \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}
}
\end{table}"""

def method1_single_run(df: pd.DataFrame, section: str) -> str:
    """
    Generates a LaTeX table for all runs.
    """
    subset = df[df["section"] == section].copy()
    if subset.empty:
        return f"% No data found for section: {section}"

    lines = [HEADER % {"caption": f"Section: {section}", "label": section.replace(" ", "_")}]

    for _, row in subset.sort_values(["file", "run_label"]).iterrows():
        run_stem = row["run_label"].split("/")[-1]
        doc_label = f"{row['file'].split("_")[0]} ({run_stem.replace("_", " ")})"
        lines.append(_format_row(row, doc_label))

    lines.append(r"\midrule")
    avg = _avg_row(subset)
    lines.append(_format_row(avg, r"\textbf{Ø}", bold=True))
    lines.append(FOOTER)
    return "\n".join(lines)

def method2_averaged(df: pd.DataFrame, section: str) -> str:
    """
    Generates a LaTeX table with average over runs.
    """
    subset = df[df["section"] == section].copy()
    if subset.empty:
        return f"% No data found for section: {section}"

    run_filter = subset["run_label"].str.endswith(("run_1", "run_2", "run_3"))
    subset = subset[run_filter]

    metric_cols = [
        "unit_precision", "unit_recall", "unit_f1",
        "role_precision", "role_recall", "role_f1",
        "unit_hier_f1", "role_hier_f1",
        "subject_indep_f1", "role_binding_f1",
        "role_coverage",
        "actor_precision", "actor_recall", "actor_f1",
    ]

    grouped = (
        subset.groupby("file", sort=True)[metric_cols]
        .mean()
        .reset_index()
    )

    lines = [HEADER % {
        "caption": f"Section: {section} (avg. over 3 runs)",
        "label": section.replace(" ", "_") + "_avg",
    }]

    for _, row in grouped.iterrows():
        lines.append(_format_row(row, row["file"].split("_")[0]))

    lines.append(r"\midrule")
    avg = _avg_row(grouped)
    lines.append(_format_row(avg, r"\textbf{Ø}", bold=True))
    lines.append(FOOTER)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert evaluation CSV to LaTeX tables.")
    parser.add_argument("csv_file", help="Path to the evaluation results CSV file.")
    parser.add_argument(
        "--section",
        default=None,
        help="Filter by section name. If omitted, all sections are processed.",
    )
    parser.add_argument(
        "--method",
        choices=["1", "2", "both"],
        default="both",
        help="Which table to generate: 1=single-run, 2=averaged, both=both (default).",
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: file '{args.csv_file}' not found.", file=sys.stderr)
        sys.exit(1)

    required_cols = {
        "section", "run_label", "file",
        "unit_precision", "unit_recall", "unit_f1",
        "role_precision", "role_recall", "role_f1",
        "unit_hier_f1", "role_hier_f1",
        "subject_indep_f1",
        "role_binding_precision", "role_binding_recall", "role_binding_f1",
        "role_coverage",
        "actor_precision", "actor_recall", "actor_f1",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: CSV is missing required columns: {missing}", file=sys.stderr)
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