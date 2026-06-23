import argparse
import re
import sys
import pandas as pd
import numpy as np

CATEGORIES = ["Actors", "Activities", "Events", "Data Objects", "Conditions", "AND Gateways", "XOR Gateways"]
RECALL_COLS = [f"{c} Recall" for c in CATEGORIES]
ABS_COLS = [f"{c} Absolute" for c in CATEGORIES]
SHORT = ["Actors", "Activities", "Events", "Data Objects", "Conditions", "AND", "XOR"]


def _doc_number(filename: str) -> int:
    m = re.match(r"(\d+)", str(filename))
    return int(m.group(1)) if m else 0


def _run_number(run_folder: str) -> str:
    m = re.search(r"run(\d+)", str(run_folder))
    return m.group(1) if m else "?"


def _parse_absolute(abs_str: str) -> tuple[int, int]:
    s = str(abs_str).strip()
    m = re.match(r"(\d+)/(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def _recall_str(r: float) -> str:
    return f"{r:.2f}"


def _abs_str(matched: int, gold: int) -> str:
    return f"{matched}/{gold}"


def _bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


def _format_cell(recall: float, abs_s: str, do_bold: bool) -> tuple[str, str]:
    rs = _recall_str(recall)
    if do_bold:
        return _bold(rs), _bold(abs_s)
    return rs, abs_s


def _build_row(type_label: str, recalls: list, abs_pairs: list, bold_flags: list) -> str:
    cells = [type_label]
    for r, (matched, gold), bf in zip(recalls, abs_pairs, bold_flags):
        rc, ab = _format_cell(float(r), _abs_str(matched, gold), bf)
        cells += [rc, ab]
    return " & ".join(cells) + r" \\"


def _bold_flags(raw_recalls: list, prep_recalls: list) -> tuple[list, list]:
    bf_raw, bf_prep = [], []
    for rv, pv in zip(raw_recalls, prep_recalls):
        if float(rv) > float(pv):
            bf_raw.append(True);
            bf_prep.append(False)
        elif float(pv) > float(rv):
            bf_raw.append(False);
            bf_prep.append(True)
        else:
            bf_raw.append(False);
            bf_prep.append(False)
    return bf_raw, bf_prep


def _header(caption: str, label: str) -> str:
    cmidrules = "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(CATEGORIES)))
    ra_headers = " & ".join(r"\textbf{R} & \textbf{A}" for _ in CATEGORIES)
    mc_headers = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{s}}}}}" for s in SHORT)
    col_spec = "l l" + " cc" * len(CATEGORIES)
    return rf"""\begin{{table}}[h]
\centering
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{{col_spec}}}
\toprule
\multirow{{2}}{{*}}{{\textbf{{Doc}}}} & \multirow{{2}}{{*}}{{\textbf{{Type}}}}
& {mc_headers} \\
{cmidrules}
& & {ra_headers} \\
\midrule"""


FOOTER = r"""\bottomrule
\end{tabular}
}
\end{table}"""


def _load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_doc_num"] = df["Gold Standard"].apply(_doc_number)
    df["_run_num"] = df["Run Folder"].apply(_run_number)
    return df.sort_values(["_doc_num", "Run Folder"]).reset_index(drop=True)


def method1_all_runs(df: pd.DataFrame) -> str:
    data = _clean(df)
    data = data[data["_run_num"].isin(["1", "2", "3"])]
    raw_data = data[data["Run Folder"].str.startswith("raw")]
    prep_data = data[data["Run Folder"].str.startswith("preprocessed")]
    doc_nums = sorted(data["_doc_num"].unique())
    doc_index = {n: i + 1 for i, n in enumerate(doc_nums)}

    lines = [_header(caption="BPMN completeness for all runs. R = Recall, A = Absolute.", label="tab:bpmn_all_runs", )]

    for doc_num in doc_nums:
        doc_raw = raw_data[raw_data["_doc_num"] == doc_num]
        doc_prep = prep_data[prep_data["_doc_num"] == doc_num]
        display = doc_index[doc_num]
        n_rows = len(doc_raw) + len(doc_prep)
        first = True

        run_nums = sorted(set(doc_raw["_run_num"].tolist() + doc_prep["_run_num"].tolist()))

        for run_num in run_nums:
            raw_row = doc_raw[doc_raw["_run_num"] == run_num]
            prep_row = doc_prep[doc_prep["_run_num"] == run_num]

            if not raw_row.empty and not prep_row.empty:
                bf_raw, bf_prep = _bold_flags([raw_row.iloc[0][rc] for rc in RECALL_COLS],
                    [prep_row.iloc[0][rc] for rc in RECALL_COLS], )
            else:
                bf_raw = bf_prep = [False] * len(CATEGORIES)

            for row, bf, type_label in [(raw_row, bf_raw, "Raw"), (prep_row, bf_prep, "Prep"), ]:
                if row.empty:
                    continue
                r = row.iloc[0]
                recalls = [r[rc] for rc in RECALL_COLS]
                pairs = [_parse_absolute(r[ac]) for ac in ABS_COLS]
                row_str = _build_row(type_label, recalls, pairs, bf)
                if first:
                    lines.append(f"\\multirow{{{n_rows}}}{{*}}{{{display}}} & " + row_str)
                    first = False
                else:
                    lines.append(" & " + row_str)

        lines.append(r"\midrule")

    for i, (type_label, tdata) in enumerate([("Raw", raw_data), ("Prep", prep_data)]):

        avg_recalls = [float(tdata[rc].mean()) for rc in RECALL_COLS]

        avg_pairs = []
        for ac in ABS_COLS:
            total_matched = 0
            total_gold = 0
            for _dn, rows in tdata.groupby("_doc_num"):
                ms = [_parse_absolute(v)[0] for v in rows[ac]]
                gs = [_parse_absolute(v)[1] for v in rows[ac]]
                total_matched += sum(ms)
                total_gold += gs[0] if gs else 0
            avg_pairs.append((total_matched, total_gold))

        row_str = _build_row(type_label, avg_recalls, avg_pairs, [False] * len(CATEGORIES))
        if i == 0:
            lines.append(f"\\multirow{{2}}{{*}}{{\\textbf{{Ø}}}} & " + row_str)
        else:
            lines.append(" & " + row_str)

    lines.append(FOOTER)
    return "\n".join(lines)


def method2_averaged(df: pd.DataFrame) -> str:
    data = _clean(df)
    data = data[data["_run_num"].isin(["1", "2", "3"])]
    raw_data = data[data["Run Folder"].str.startswith("raw")]
    prep_data = data[data["Run Folder"].str.startswith("preprocessed")]
    doc_nums = sorted(data["_doc_num"].unique())
    doc_index = {n: i + 1 for i, n in enumerate(doc_nums)}

    raw_per_doc = {}
    for doc_num in doc_nums:
        rows = raw_data[raw_data["_doc_num"] == doc_num]
        if rows.empty:
            continue
        avg_recalls = [float(rows[rc].mean()) for rc in RECALL_COLS]
        avg_pairs = []
        for ac in ABS_COLS:
            parsed = [_parse_absolute(v) for v in rows[ac]]
            ms = [p[0] for p in parsed]
            gs = [p[1] for p in parsed]
            avg_pairs.append((int(round(np.mean(ms))), int(round(np.mean(gs)))))
        raw_per_doc[doc_num] = (avg_recalls, avg_pairs)

    prep_per_doc = {}
    for doc_num in doc_nums:
        rows = prep_data[prep_data["_doc_num"] == doc_num]
        if rows.empty:
            continue
        avg_recalls = [float(rows[rc].mean()) for rc in RECALL_COLS]
        avg_pairs = []
        for ac in ABS_COLS:
            parsed = [_parse_absolute(v) for v in rows[ac]]
            ms = [p[0] for p in parsed]
            gs = [p[1] for p in parsed]
            avg_pairs.append((int(round(np.mean(ms))), int(round(np.mean(gs)))))
        prep_per_doc[doc_num] = (avg_recalls, avg_pairs)

    lines = [
        _header(caption=r"BPMN completeness --- avg.\ over 3 runs. R = Recall, A = Absolute.", label="tab:bpmn_avg", )]

    for doc_num in doc_nums:
        display = doc_index[doc_num]
        has_raw = doc_num in raw_per_doc
        has_prep = doc_num in prep_per_doc
        n_types = int(has_raw) + int(has_prep)
        first = True

        if has_raw and has_prep:
            bf_raw, bf_prep = _bold_flags(raw_per_doc[doc_num][0], prep_per_doc[doc_num][0])
        else:
            bf_raw = bf_prep = [False] * len(CATEGORIES)

        if has_raw:
            recalls, pairs = raw_per_doc[doc_num]
            row_str = _build_row("Raw", recalls, pairs, bf_raw)
            lines.append(f"\\multirow{{{n_types}}}{{*}}{{{display}}} & " + row_str)
            first = False

        if has_prep:
            recalls, pairs = prep_per_doc[doc_num]
            row_str = _build_row("Prep", recalls, pairs, bf_prep)
            if first:
                lines.append(f"\\multirow{{{n_types}}}{{*}}{{{display}}} & " + row_str)
            else:
                lines.append(" & " + row_str)

        lines.append(r"\midrule")

    n_docs = len(doc_nums)
    for i, (type_label, per_doc) in enumerate([("Raw", raw_per_doc), ("Prep", prep_per_doc)]):

        avg_recalls = [sum(per_doc[dn][0][j] for dn in per_doc) / n_docs for j in range(len(CATEGORIES))]
        avg_pairs = [(sum(per_doc[dn][1][j][0] for dn in per_doc), sum(per_doc[dn][1][j][1] for dn in per_doc)) for j in
            range(len(CATEGORIES))]

        row_str = _build_row(type_label, avg_recalls, avg_pairs, [False] * len(CATEGORIES))
        if i == 0:
            lines.append(f"\\multirow{{2}}{{*}}{{\\textbf{{Ø}}}} & " + row_str)
        else:
            lines.append(" & " + row_str)

    lines.append(FOOTER)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert BPMN completeness Excel to LaTeX tables.")
    parser.add_argument("excel_file", help="Path to the .xlsx file.")
    parser.add_argument("--method", choices=["1", "2", "both"], default="both")
    args = parser.parse_args()

    try:
        df = _load(args.excel_file)
    except FileNotFoundError:
        print(f"Error: '{args.excel_file}' not found.", file=sys.stderr)
        sys.exit(1)

    required = {"Gold Standard", "Run Folder"} | set(RECALL_COLS) | set(ABS_COLS)
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
