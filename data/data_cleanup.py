"""data.data_cleanup

Convert raw market-data CSVs into the **long-format OHLCV** layout expected by
the backtester engine (see data_structure_template.md in this folder).

Target schema
-------------
    Date, Symbol, Open, High, Low, Close, Volume

Supported input formats
-----------------------
1. **Real-sample per-file CSVs** (``data/real_samples/*.csv``)
   - Columns: Time, Open, Close, Volume, High, Low, …, Dividend, Split
   - Symbol is extracted from the filename  (e.g. ``ADRC_BABA_day.csv`` → ``BABA``).

2. **Synthetic generator output** (``data/synthetic.csv``)
   - Already nearly correct; may contain extra columns (e.g. ``Sigma``).
   - ``Date`` may be the index rather than a column.

3. **Generic / user-supplied CSV**
   - Any CSV that contains at least Date/Time + OHLCV columns (names may differ).
   - A column-mapping dict lets the user translate arbitrary headers.

Usage
-----
    python -m data.data_cleanup                       # clean real_samples → combined CSV
    python -m data.data_cleanup --input my_file.csv   # clean a single file
"""

from __future__ import annotations

import os
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

#target schema (from data_structure_template.md) ──────────────────────────
REQUIRED_COLUMNS = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

def _extract_symbol_from_filename(filepath: str) -> str:
    """Extract the ticker symbol from a real-sample filename.

    Convention: ``<prefix>_<SYMBOL>_<freq>.csv``
    Example:    ``ADRC_BABA_day.csv``  →  ``BABA``
    Falls back to the full stem if the pattern doesn't match.
    """
    stem = Path(filepath).stem  # e.g. "ADRC_BABA_day"
    parts = stem.split("_")
    if len(parts) >= 3:
        # Middle token(s) are the symbol; last token is freq (day/min/…)
        return "_".join(parts[1:-1])
    return stem

def _coerce_datetime(series: pd.Series) -> pd.Series:
    """Best-effort parse of a date/time column into proper datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, infer_datetime_format=True, utc=False)

def read_real_sample(filepath: str) -> pd.DataFrame:
    """Read one real-sample CSV and return a cleaned DataFrame.

    The real-sample files use ``Time`` instead of ``Date``, have columns
    in a non-standard order, and encode the symbol in the filename.
    """
    df = pd.read_csv(filepath)

    symbol = _extract_symbol_from_filename(filepath)

    # Build the column mapping from raw → target
    rename_map: Dict[str, str] = {}
    if "Time" in df.columns and "Date" not in df.columns:
        rename_map["Time"] = "Date"

    df = df.rename(columns=rename_map)

    # Ensure required columns exist after rename
    for col in PRICE_COLUMNS + ["Volume"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {filepath}")

    df["Symbol"] = symbol
    df["Date"] = _coerce_datetime(df["Date"])

    return df[REQUIRED_COLUMNS].copy()

def read_synthetic(filepath: str) -> pd.DataFrame:
    """Read a synthetic-generator CSV (``synthetic.csv``).

    Handles the case where ``Date`` is stored as the index.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"] if "Date" in pd.read_csv(filepath, nrows=0).columns else True)

    # If Date was the index, reset it
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()

    df["Date"] = _coerce_datetime(df["Date"])

    # synthetic.csv already contains Symbol; just keep required cols
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in synthetic CSV: {missing}")

    return df[REQUIRED_COLUMNS].copy()

def read_generic(
    filepath: str,
    column_map: Optional[Dict[str, str]] = None,
    symbol_override: Optional[str] = None,
) -> pd.DataFrame:
    """Read an arbitrary CSV and attempt to map it to the target schema.

    Parameters
    ----------
    filepath : str
        Path to the CSV.
    column_map : dict, optional
        Mapping from *source* column names to *target* names.  For example:
        ``{"timestamp": "Date", "ticker": "Symbol", "vol": "Volume"}``
    symbol_override : str, optional
        If the file has no Symbol column, use this value for every row.
    """
    df = pd.read_csv(filepath)

    if column_map:
        df = df.rename(columns=column_map)

    # Auto-detect common aliases
    alias_map: Dict[str, str] = {
        "Time": "Date",
        "Timestamp": "Date",
        "datetime": "Date",
        "time": "Date",
        "date": "Date",
        "Ticker": "Symbol",
        "ticker": "Symbol",
        "symbol": "Symbol",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "Vol": "Volume",
        "vol": "Volume",
    }
    for raw, target in alias_map.items():
        if raw in df.columns and target not in df.columns:
            df = df.rename(columns={raw: target})

    # If Date came from the index
    if "Date" not in df.columns and df.index.name in ("Date", "Time", "Timestamp"):
        df = df.reset_index().rename(columns={df.index.name: "Date"})

    if "Symbol" not in df.columns:
        if symbol_override:
            df["Symbol"] = symbol_override
        else:
            df["Symbol"] = Path(filepath).stem

    df["Date"] = _coerce_datetime(df["Date"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns after mapping: {missing} in {filepath}")

    return df[REQUIRED_COLUMNS].copy()

def clean(df: pd.DataFrame, *, drop_duplicates: bool = True, sort: bool = True) -> pd.DataFrame:
    """Apply the validation / cleanup steps from data_structure_template.md.

    Steps performed
    ---------------
    1. Ensure all required columns are present.
    2. Parse ``Date`` to datetime.
    3. Coerce OHLC to float and Volume to int (NaN → 0).
    4. Drop rows where any price column is NaN.
    5. Clamp negative prices to 0.01.
    6. Enforce High >= max(Open, Close) and Low <= min(Open, Close).
    7. Fill missing Volume with 0.
    8. Drop duplicate (Date, Symbol) rows (keeping first).
    9. Sort by Date then Symbol.
    10. Convert Symbol to category dtype for memory efficiency.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Date
    df["Date"] = _coerce_datetime(df["Date"])

    # Numeric coercion
    for col in PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)

    # Drop rows with NaN prices (as recommended by template)
    df = df.dropna(subset=PRICE_COLUMNS).copy()

    # Clamp negatives
    for col in PRICE_COLUMNS:
        df[col] = df[col].clip(lower=0.01)

    # Enforce OHLC consistency
    bar_max = df[["Open", "Close"]].max(axis=1)
    bar_min = df[["Open", "Close"]].min(axis=1)
    df["High"] = df["High"].clip(lower=bar_max)
    df["Low"] = df["Low"].clip(upper=bar_min)
    df["Low"] = df["Low"].clip(lower=0.01)

    # Duplicates
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["Date", "Symbol"], keep="first")
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped} duplicate (Date, Symbol) rows.")

    # Sort
    if sort:
        df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    # Category dtype for Symbol
    df["Symbol"] = df["Symbol"].astype("category")

    return df

def validate(df: pd.DataFrame) -> List[str]:
    """Run the template's validation checklist and return a list of warnings.

    Returns an empty list if everything passes.
    """
    warnings: List[str] = []

    # Required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            warnings.append(f"Missing column: {col}")

    if warnings:
        return warnings  # can't check further without columns

    # NaN prices
    for col in PRICE_COLUMNS:
        n = df[col].isna().sum()
        if n:
            warnings.append(f"{col} has {n} NaN value(s).")

    # Volume dtype
    if not pd.api.types.is_integer_dtype(df["Volume"]):
        warnings.append("Volume is not integer-like.")

    # Date dtype
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        warnings.append("Date is not datetime dtype.")

    # Duplicates
    dup_count = df.duplicated(subset=["Date", "Symbol"]).sum()
    if dup_count:
        warnings.append(f"{dup_count} duplicate (Date, Symbol) row(s).")

    # OHLC consistency
    bad_high = (df["High"] < df[["Open", "Close"]].max(axis=1)).sum()
    bad_low = (df["Low"] > df[["Open", "Close"]].min(axis=1)).sum()
    if bad_high:
        warnings.append(f"{bad_high} row(s) where High < max(Open, Close).")
    if bad_low:
        warnings.append(f"{bad_low} row(s) where Low > min(Open, Close).")

    # Negative prices
    for col in PRICE_COLUMNS:
        neg = (df[col] < 0).sum()
        if neg:
            warnings.append(f"{col} has {neg} negative value(s).")

    return warnings

def clean_real_samples(
    folder: str = os.path.join(os.path.dirname(__file__), "real_samples"),
    output: str = os.path.join(os.path.dirname(__file__), "real_samples_clean.csv"),
) -> pd.DataFrame:
    """Read every CSV in ``real_samples/``, clean, combine, and save.

    Returns the combined DataFrame.
    """
    pattern = os.path.join(folder, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = read_real_sample(fp)
            frames.append(df)
        except Exception as exc:
            print(f"  ⚠ Skipping {os.path.basename(fp)}: {exc}")

    if not frames:
        raise RuntimeError("No files could be read successfully.")

    combined = pd.concat(frames, ignore_index=True)
    combined = clean(combined)

    issues = validate(combined)
    if issues:
        print("Validation warnings after cleaning:")
        for w in issues:
            print(f"  - {w}")
    else:
        print("Validation passed — no issues found.")

    combined.to_csv(output, index=False)
    print(f"Saved {len(combined):,} rows ({combined['Symbol'].nunique()} symbols) → {output}")
    return combined

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean raw market CSVs into the engine's long-format OHLCV layout.",
    )
    p.add_argument(
        "--input", "-i",
        help="Path to a single CSV to clean. If omitted, cleans all real_samples/*.csv.",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path. Defaults to <input>_clean.csv or data/real_samples_clean.csv.",
    )
    p.add_argument(
        "--format", "-f",
        choices=["real_sample", "synthetic", "generic"],
        default="generic",
        help="Reader to use for --input (default: generic).",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="Override symbol name (for single-asset files without a Symbol column).",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation on an already-cleaned CSV without modifying it.",
    )
    return p

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Validate-only mode
    if args.validate_only:
        if not args.input:
            parser.error("--validate-only requires --input <file>")
        df = pd.read_csv(args.input, parse_dates=["Date"])
        issues = validate(df)
        if issues:
            print(f"Validation issues in {args.input}:")
            for w in issues:
                print(f"  - {w}")
        else:
            print(f"{args.input}: all checks passed.")
        return

    # Single-file mode
    if args.input:
        reader = {
            "real_sample": read_real_sample,
            "synthetic": read_synthetic,
            "generic": lambda fp: read_generic(fp, symbol_override=args.symbol),
        }[args.format]

        df = reader(args.input)
        df = clean(df)

        output = args.output or (Path(args.input).stem + "_clean.csv")
        df.to_csv(output, index=False)
        print(f"Saved {len(df):,} rows → {output}")

        issues = validate(df)
        if issues:
            for w in issues:
                print(f"  - {w}")
        else:
            print("Validation passed.")
        return

    # Batch mode — process everything in real_samples/
    out = args.output or os.path.join(os.path.dirname(__file__), "real_samples_clean.csv")
    clean_real_samples(output=out)

if __name__ == "__main__":
    main()