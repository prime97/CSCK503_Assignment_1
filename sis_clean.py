from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ----------------------------- I/O helpers ----------------------------- #
def read_csv_fallback(path: Path) -> pd.DataFrame:
    """Read CSV trying common encodings."""
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # last attempt with default
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save CSV with UTF-8 encoding."""
    df.to_csv(path, index=False, encoding="utf-8")


# --------------------------- cleaning helpers -------------------------- #
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy column names:
    - collapse whitespace, remove newlines and slashes
    - lower-case
    - spaces -> underscores
    """
    cols = []
    for c in df.columns:
        c2 = re.sub(r"\s+", " ", str(c)).strip()
        c2 = c2.replace("/", " ").replace("\n", " ")
        c2 = re.sub(r"\s+", " ", c2).lower().replace(" ", "_")
        cols.append(c2)
    out = df.copy()
    out.columns = cols
    return out


def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and collapse spaces in all string-like columns."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "O" or pd.api.types.is_string_dtype(out[c]):
            out[c] = (
                out[c].astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
    return out


def find_date_columns(df: pd.DataFrame) -> List[str]:
    """Heuristic: any column name containing 'date'."""
    return [c for c in df.columns if "date" in c]


def parse_dates_best_effort(series: pd.Series) -> pd.Series:
    """
    Parse dates using several formats, then a generic fallback.
    Safe: returns NaT for unparseable values.
    """
    s = (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]
    best = pd.Series(pd.NaT, index=s.index)
    for fmt in formats:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        best = best.fillna(parsed)
    # generic last pass
    best = best.fillna(pd.to_datetime(s, errors="coerce"))
    return best


def constant_columns(df: pd.DataFrame) -> List[str]:
    """Columns with only one distinct non-null value."""
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1]


def high_missing_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """Columns with missing ratio above threshold (default 95%)."""
    return [c for c in df.columns if df[c].isna().mean() > threshold]


def standardise_qualifications(s: pd.Series) -> pd.Series:
    """
    Map common variants to tidy labels.
    Extend this mapping as needed for your data.
    """
    if s.dtype != "O" and not pd.api.types.is_string_dtype(s):
        return s
    x = s.astype("string").str.strip()
    x = x.str.replace(r"\bph\.?\s*d\.?\b", "PhD", flags=re.IGNORECASE, regex=True)
    x = x.str.replace(r"\bm\.?\s*sc\.?\b", "MSc", flags=re.IGNORECASE, regex=True)
    x = x.str.replace(r"\bb\.?\s*sc\.?\b", "BSc", flags=re.IGNORECASE, regex=True)
    x = x.str.replace(r"\bm\.?\s*ba\b", "MBA", flags=re.IGNORECASE, regex=True)
    x = x.str.replace(r"\bmasters?\b", "Master", flags=re.IGNORECASE, regex=True)
    return x


def ensure_identifier(df: pd.DataFrame, id_candidates: List[str]) -> Tuple[pd.DataFrame, str, Dict[str, int]]:
    """
    Validate an identifier column.
    - Accepts first available candidate (e.g., 'id', 'staff_id', etc.)
    - Treats '', '0', 'nan', 'none' as missing
    - If non-unique or missing, creates 'faculty_uid' surrogate key
    Returns: (df, id_col_used, diagnostics)
    """
    out = df.copy()
    diag = {}

    id_col = None
    for cand in id_candidates:
        if cand in out.columns:
            id_col = cand
            break

    if id_col is None:
        out["faculty_uid"] = np.arange(1, len(out) + 1)
        diag.update({"id_col": "faculty_uid", "created_uid": len(out)})
        return out, "faculty_uid", diag

    out[id_col] = out[id_col].astype("string")
    invalid_mask = (
        out[id_col].str.strip().isin(["", "0", "nan", "none"]) | out[id_col].isna()
    )
    diag["invalid_ids_found"] = int(invalid_mask.sum())
    out.loc[invalid_mask, id_col] = pd.NA

    nunique = out[id_col].nunique(dropna=True)
    missing = int(out[id_col].isna().sum())
    diag["unique_ids"] = int(nunique)
    diag["missing_ids"] = missing
    diag["rows"] = len(out)

    # If not fully unique or has missing, create a clean surrogate
    if (nunique + missing) != len(out):
        out["faculty_uid"] = np.arange(1, len(out) + 1)
        diag["created_uid"] = len(out)
        return out, "faculty_uid", diag

    # If only missing remain, still add surrogate to preserve row identity
    if missing:
        out["faculty_uid"] = np.arange(1, len(out) + 1)
        diag["created_uid_for_missing"] = missing
        return out, "faculty_uid", diag

    return out, id_col, diag


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute numeric with median, categorical with mode.
    Simple, transparent, and suitable for baseline ML prep.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].median())
        else:
            if out[c].isna().any():
                m = out[c].mode(dropna=True)
                if len(m):
                    out[c] = out[c].fillna(m.iloc[0])
    return out


def summarise(df: pd.DataFrame) -> Dict[str, object]:
    """Key quality metrics for quick before/after checks."""
    return {
        "shape": df.shape,
        "duplicates": int(df.duplicated().sum()),
        "missing_top10": df.isna().sum().sort_values(ascending=False).head(10),
        "constant_cols": constant_columns(df),
        "high_missing_cols": high_missing_columns(df),
    }


# ------------------------------- pipeline ------------------------------ #
def clean_pipeline(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict, List[str]]:
    """
    Full cleaning pipeline.
    Returns: cleaned_df, before_summary, after_summary, dropped_columns
    """
    # 1) column names + trim
    df = normalise_columns(df_raw)
    df = trim_strings(df)

    # 2) before snapshot
    before = summarise(df)

    # 3) robust date parsing (by name)
    for c in find_date_columns(df):
        df[c] = parse_dates_best_effort(df[c])

    # 4) identifier handling
    df, id_col, id_diag = ensure_identifier(
        df, id_candidates=["id", "staff_id", "employee_id", "faculty_id"]
    )

    # 5) drop bad columns (very sparse + constant)
    drop_cols = sorted(set(high_missing_columns(df) + constant_columns(df)))
    df = df.drop(columns=drop_cols, errors="ignore")

    # 6) standardise likely qualification columns
    for c in df.columns:
        if "qualification" in c:
            df[c] = standardise_qualifications(df[c])

    # 7) impute
    df = impute_missing(df)

    # 8) remove duplicates
    df = df.drop_duplicates()

    # 9) after snapshot
    after = summarise(df)

    return df, before, after, drop_cols


# --------------------------------- main -------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Clean SIS Faculty CSV (no report)."
    )
    parser.add_argument(
        "--in", dest="input_csv", default="SIS_Faculty-List.csv",
        help="Input CSV path (default: SIS_Faculty-List.csv)"
    )
    parser.add_argument(
        "--out", dest="output_csv", default="SIS_Faculty-List_clean.csv",
        help="Output cleaned CSV path (default: SIS_Faculty-List_clean.csv)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_raw = read_csv_fallback(input_path)
    cleaned, before, after, dropped = clean_pipeline(df_raw)

    save_csv(cleaned, output_path)

    # Console summary
    print("=== BEFORE ===")
    print(f"shape: {before['shape']}")
    print(f"duplicates: {before['duplicates']}")
    print("missing (top 10):")
    print(before["missing_top10"])
    print("constant columns:", before["constant_cols"])
    print("high-missing columns:", before["high_missing_cols"])

    print("\n=== AFTER ===")
    print(f"shape: {after['shape']}")
    print(f"duplicates: {after['duplicates']}")
    print("missing (top 10):")
    print(after["missing_top10"])
    print("Dropped columns:", dropped)
    print(f"\nSaved cleaned CSV -> {output_path.resolve()}")


if __name__ == "__main__":
    main()
