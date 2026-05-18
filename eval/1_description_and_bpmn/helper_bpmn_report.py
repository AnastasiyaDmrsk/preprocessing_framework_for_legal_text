"""
Reads the Excel file and produces:
  • Per-gold-standard averages (over 3 runs) for raw and preprocessed (and approach)
  • Grand averages across all gold-standard files for raw and preprocessed

Two column types are handled:
  - Float columns  (e.g. "Actors Recall")    -> plain numeric mean
  - Fraction columns (e.g. "Actors Absolute") -> numerators and denominators
    are averaged independently, each rounded to 2 decimal places.
    e.g.  3/5  and  5/7  ->  (3+5)/2 = 4.00  and  (5+7)/2 = 6.00  ->  "4.00/6.00"

Console output uses " & " as column separator (LaTeX-friendly).

Usage:
    python helper_bpmn_report.py                          # default filename
    python helper_bpmn_report.py path/to/report.xlsx
    python helper_bpmn_report.py path/to/report.xlsx Sheet2
"""

import re
import sys
import numpy as np
import pandas as pd

FILE_PATH           = sys.argv[1] if len(sys.argv) > 1 else "bpmn_completeness_report.xlsx"
SHEET_NAME          = sys.argv[2] if len(sys.argv) > 2 else 0
GOLD_STD_COL        = "Gold Standard"
RUN_FOLDER_COL      = "Run Folder"
RAW_PREFIX          = "raw"
PREPROCESSED_PREFIX = "preprocessed"
APPROACH_PREFIX = "approach"
FRACTION_DECIMALS   = 2          # decimal places for fraction num/denom averages
FLOAT_DECIMALS      = 2          # decimal places for plain numeric averages
SEP                 = " & "      # column separator in console output

FRACTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")


def load_data(path: str, sheet) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from '{path}'")
    print(f"Columns: {list(df.columns)}\n")
    return df


def find_column(df: pd.DataFrame, target: str) -> str:
    """Case-insensitive column lookup."""
    mapping = {c.lower(): c for c in df.columns}
    key = target.lower()
    if key not in mapping:
        raise KeyError(f"Column '{target}' not found. Available: {list(df.columns)}")
    return mapping[key]


def classify_column(series: pd.Series) -> str:
    """
    Returns 'numeric', 'fraction', or 'ignore'.
    Fraction columns have >=80% of non-null values matching N/M.
    """
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    sample = series.dropna().astype(str)
    if sample.empty:
        return "ignore"
    if sample.apply(lambda v: bool(FRACTION_RE.match(v))).mean() >= 0.8:
        return "fraction"
    if pd.to_numeric(sample, errors="coerce").notna().mean() >= 0.8:
        return "numeric"
    return "ignore"


def extract_run_type(run_folder_value: str) -> str:
    """'raw_run1' -> 'raw',  'preprocessed_run3' -> 'preprocessed'"""
    val = str(run_folder_value).strip().lower()
    if val.startswith(PREPROCESSED_PREFIX):
        return PREPROCESSED_PREFIX
    if val.startswith(RAW_PREFIX):
        return RAW_PREFIX
    if val.startswith(APPROACH_PREFIX):
        return APPROACH_PREFIX
    return val


def parse_fraction(value: str):
    """Return (numerator, denominator) as ints, or (NaN, NaN) if unparseable."""
    m = FRACTION_RE.match(str(value))
    if m:
        return float(m.group(1)), float(m.group(2))
    return np.nan, np.nan


def fraction_mean(series: pd.Series) -> str:
    """
    Average numerators and denominators independently, each rounded to
    FRACTION_DECIMALS decimal places.
    e.g. ["3/5", "5/7"] -> "4.00/6.00"
    """
    pairs = series.apply(parse_fraction)
    nums = [n for n, _ in pairs if not (isinstance(n, float) and np.isnan(n))]
    dens = [d for _, d in pairs if not (isinstance(d, float) and np.isnan(d))]
    if not nums:
        return "N/A"
    avg_num = round(sum(nums) / len(nums), FRACTION_DECIMALS)
    avg_den = round(sum(dens) / len(dens), FRACTION_DECIMALS)
    return f"{avg_num:.{FRACTION_DECIMALS}f}/{avg_den:.{FRACTION_DECIMALS}f}"


def fraction_group_mean(df: pd.DataFrame, group_cols: list, frac_cols: list) -> pd.DataFrame:
    """Compute fraction_mean per (group_cols) group for every fraction column."""
    records = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for col in frac_cols:
            row[col] = fraction_mean(grp[col])
        records.append(row)
    return pd.DataFrame(records)


def fraction_grand_mean(df: pd.DataFrame, frac_cols: list) -> pd.Series:
    """Re-parse already-averaged fraction strings and average them again."""
    return pd.Series({col: fraction_mean(df[col]) for col in frac_cols})


def fmt_value(v, decimals: int = FLOAT_DECIMALS) -> str:
    """Format a single cell value for console output."""
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def print_ampersand_table(df: pd.DataFrame, label_cols: list, metric_cols: list) -> None:
    """
    Print df rows as:   LabelCol1 & LabelCol2 & metric1 & metric2 & ...
    Header row first, then one data row per DataFrame row.
    """
    all_cols = label_cols + metric_cols
    header = SEP.join(all_cols)
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        parts = [str(row[c]) for c in label_cols] + \
                [fmt_value(row[c]) for c in metric_cols]
        print(SEP.join(parts))

def main():
    df = load_data(FILE_PATH, SHEET_NAME)

    gold_col   = find_column(df, GOLD_STD_COL)
    folder_col = find_column(df, RUN_FOLDER_COL)
    df["_run_type"] = df[folder_col].apply(extract_run_type)

    # Classify columns
    skip_cols = {gold_col.lower(), folder_col.lower(), "_run_type"}
    num_cols, frac_cols, ignored = [], [], []
    for col in df.columns:
        if col.lower() in skip_cols:
            continue
        kind = classify_column(df[col])
        if kind == "numeric":
            num_cols.append(col)
        elif kind == "fraction":
            frac_cols.append(col)
        else:
            ignored.append(col)

    print(f"Float columns    ({len(num_cols)}):  {num_cols}")
    print(f"Fraction columns ({len(frac_cols)}): {frac_cols}")
    print(f"Ignored columns  ({len(ignored)}):   {ignored}\n")

    all_metric_cols = num_cols + frac_cols
    group_cols      = [gold_col, "_run_type"]
    num_grp = (
        df.groupby(group_cols)[num_cols].mean().reset_index()
        if num_cols else pd.DataFrame(columns=group_cols)
    )
    frac_grp = (
        fraction_group_mean(df, group_cols, frac_cols)
        if frac_cols else pd.DataFrame(columns=group_cols)
    )
    per_group_avg = (
        num_grp.merge(frac_grp, on=group_cols, how="outer") if frac_cols else num_grp
    )
    per_group_avg = per_group_avg.rename(columns={"_run_type": "Run Type"})

    raw_avg  = per_group_avg[per_group_avg["Run Type"] == RAW_PREFIX].copy()
    prep_avg = per_group_avg[per_group_avg["Run Type"] == PREPROCESSED_PREFIX].copy()
    app_avg = per_group_avg[per_group_avg["Run Type"] == APPROACH_PREFIX].copy()

    label_cols = [gold_col, "Run Type"]
    sep_line   = "=" * 70

    print(sep_line)
    print(f"Average per gold-standard -- RAW  ({len(raw_avg)} groups)")
    print(sep_line)
    print_ampersand_table(raw_avg, label_cols, all_metric_cols)

    print()
    print(sep_line)
    print(f"Average per gold-standard -- PREPROCESSED  ({len(prep_avg)} groups)")
    print(sep_line)
    print_ampersand_table(prep_avg, label_cols, all_metric_cols)

    print()
    print(sep_line)
    print(f"Average per gold-standard -- APPROACH  ({len(app_avg)} groups)")
    print(sep_line)
    print_ampersand_table(app_avg, label_cols, all_metric_cols)

    grand_num_raw   = raw_avg[num_cols].mean()  if num_cols  else pd.Series(dtype=float)
    grand_num_prep  = prep_avg[num_cols].mean() if num_cols  else pd.Series(dtype=float)
    grand_num_app = app_avg[num_cols].mean() if num_cols else pd.Series(dtype=float)
    grand_frac_raw  = fraction_grand_mean(raw_avg,  frac_cols) if frac_cols else pd.Series(dtype=object)
    grand_frac_prep = fraction_grand_mean(prep_avg, frac_cols) if frac_cols else pd.Series(dtype=object)
    grand_app_prep = fraction_grand_mean(app_avg, frac_cols) if frac_cols else pd.Series(dtype=object)

    summary = pd.DataFrame(
        {
            RAW_PREFIX:          pd.concat([grand_num_raw,  grand_frac_raw]),
            PREPROCESSED_PREFIX: pd.concat([grand_num_prep, grand_frac_prep]),
            APPROACH_PREFIX: pd.concat([grand_num_app, grand_app_prep]),
        }
    ).T
    summary.index.name = "Run Type"

    print()
    print(sep_line)
    print("Grand average across all gold-standard groups")
    print(sep_line)
    # Print header
    grand_metric_cols = list(summary.columns)
    print("Run Type" + SEP + SEP.join(grand_metric_cols))
    print("-" * 70)
    for run_type, row in summary.iterrows():
        parts = [str(run_type)] + [fmt_value(row[c]) for c in grand_metric_cols]
        print(SEP.join(parts))

    out_path = FILE_PATH.replace(".xlsx", "_analysis.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        raw_avg.to_excel(writer,  sheet_name="Raw_per_group",          index=False)
        prep_avg.to_excel(writer, sheet_name="Preprocessed_per_group", index=False)
        app_avg.to_excel(writer, sheet_name="Approach_per_group", index=False)
        summary.to_excel(writer,  sheet_name="Grand_averages")
    print(f"\nResults saved to '{out_path}'")


if __name__ == "__main__":
    main()