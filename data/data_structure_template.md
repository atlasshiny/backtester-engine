
# Long-format OHLCV data template

This document describes the expected long-format OHLCV (Open/High/Low/Close/Volume) CSV layout used by the backtester. "Long format" means one row per timestamp per symbol (as opposed to wide format where each symbol is a separate column).

## Required columns

| Column | Type | Description | Notes |
|---|---:|---|---|
| `Date` | datetime | Timestamp for the bar (ISO 8601 recommended) | Use UTC or document timezone; often parsed as the index |
| `Symbol` | string / category | Ticker or instrument identifier | Keep consistent tickers across files; convert to `category` to save memory |
| `Open` | float | Opening price for the interval | Non-negative; same currency/units as other price fields |
| `High` | float | Highest price during the interval | Should be >= `Open` and `Close` |
| `Low` | float | Lowest price during the interval | Should be <= `Open` and `Close` |
| `Close` | float | Closing price for the interval | Non-negative |
| `Volume` | int | Traded quantity for the interval | Use `0` if unknown; avoid empty cells to preserve integer dtype |

Optional/auxiliary columns (useful but not required):

- `Adj Close` : Adjusted close price when available.
- Any indicator columns (e.g. `Sigma`, `VWAP`) — these are allowed and will be ignored by default by the engine unless a strategy references them.

All numeric price fields should be non-negative and in a consistent currency/units. Missing numeric values should be avoided; if present, document how they should be handled.

## File conventions

- File format: CSV (UTF-8). Comma-separated by default.
- Row order: Not strictly required, but recommendation is ascending `Date` then `Symbol`.
- Indexing: The backtester often sets `Date` as the DataFrame index; keep `Date` present as a column so files are self-describing.
- Frequency: File may contain any frequency (daily, hourly, minute). Ensure the `Date` values reflect the exact timestamp for the bar.
- Trading vs calendar days: If using business-day frequency, be explicit (e.g. `2020-01-01` might be a holiday).

## Example CSV header and a few rows

Example (three symbols over three hourly bars):

```
Date,Symbol,Open,High,Low,Close,Volume
1980-01-01 00:00:00,SYNTH,200.00,201.05,199.80,200.50,82000
1980-01-01 01:00:00,SYNTH,200.50,201.10,200.10,200.90,79000
1980-01-01 00:00:00,TEST,325.00,326.20,324.90,325.40,51000
1980-01-01 01:00:00,TEST,325.40,326.00,324.80,325.75,49500
1980-01-01 00:00:00,THEORY,86.00,86.60,85.90,86.20,115000
```

Note: The above is sorted by time for readability. The file can contain interleaved symbols for each timestamp; stable sorting and consistent ordering make downstream processing faster.

## Recommended dtypes and memory tips (pandas)

- Parse `Date` column as datetime on read: `parse_dates=['Date']` and optionally `infer_datetime_format=True`.
- Convert `Symbol` to `category` after loading to reduce memory: `df['Symbol'] = df['Symbol'].astype('category')`.
- Use `float32` for price columns when memory is constrained: `df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].astype('float32')`.
- Use `int32` for `Volume` if values fit: `df['Volume'] = df['Volume'].astype('int32')`.

## Example: reading the CSV with pandas

```python
import pandas as pd

df = pd.read_csv(
	'data/synthetic.csv',
	parse_dates=['Date'],
	infer_datetime_format=True,
)

# Optional: set index, sort, and convert dtypes
df = df.sort_values(['Date', 'Symbol'])
df = df.set_index('Date')
df['Symbol'] = df['Symbol'].astype('category')
df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].astype('float32')
df['Volume'] = df['Volume'].astype('int32')

print(df.head())
```

If you prefer the index to be a MultiIndex with `Date` and `Symbol`:

```python
df = df.reset_index().set_index(['Date','Symbol']).sort_index()
```

## Common pitfalls and recommendations

- Timezones: Be consistent. Use UTC or ensure all files use the same timezone. Avoid mixing naive and timezone-aware datetimes.
- Missing bars: Some symbols may have missing timestamps (e.g., illiquid instruments). Strategies should handle sparse series gracefully.
- Duplicate rows: Ensure no duplicate (Date, Symbol) pairs in the file. If duplicates exist, decide whether to aggregate or drop them.
- Non-trading intervals: For higher-frequency data (minutes/seconds), include only actual trading bars (or mark non-trading explicitly).
- Volume: If volume is unknown, use `0` rather than empty cells to avoid dtype inference issues.

## Minimal validation checklist before using a file with the engine

- Columns present and spelled correctly: `Date, Symbol, Open, High, Low, Close, Volume`.
- No NaNs in price columns (or documented handling).
- `Volume` is integer-like.
- `Date` parsed correctly (sample a few rows to verify timezone and format).
- No duplicate (Date, Symbol) rows.

## Engine expectations & clarifications (code mapping)

The repository contains a few behaviors that affect how input CSVs are treated. The notes below map those behaviors to the relevant runtime assumptions so you can prepare files that work predictably with the engine.

- `Symbol` injection: If a DataFrame does not contain a `Symbol` column, the engine will inject `Symbol='SINGLE'` so single-asset datasets are supported transparently (see `backtest/engine.py`).
- Data input types: The engine accepts either a pandas `DataFrame` or a dict-of-arrays produced by `df_to_arrays()` (see `backtest/array_utils.py`). Both paths preserve additional columns (e.g. indicator columns like `Sigma`).
- `Date` vs index: Many components expect a `Date` column (datetime). The engine can fall back to a positional index when `Date` is absent, but including a parsed `Date` column is strongly recommended for correctness and reproducibility.
- `Close` is functionally required: Several features (technical indicators, Monte Carlo simulations, performance analytics) operate on the `Close` series. Monte Carlo mode raises if `Close` is missing (`backtest/monte_carlo.py`).
- Broker/fill logic dependencies: The `Broker` relies on OHLC fields for fill decisions and timestamping. MARKET fills use `Open`, LIMIT fills inspect `Low`/`High`, and the broker updates last-known prices from `Close` (see `backtest/broker.py`). If you plan to execute orders, include valid `Open`, `High`, `Low`, and `Close` values.
- Sorting: The engine assumes the dataset is processed in ascending time order. For deterministic multi-asset behavior, sort by `Date` then `Symbol` before running a backtest.
- Missing / empty values: Prefer using explicit values (e.g. `0` for unknown `Volume`) rather than empty cells to avoid dtype inference problems when converting to arrays.
- Example reading: Some example scripts in the repo read CSVs without `parse_dates=['Date']` (e.g. `main.py`). Use `pd.read_csv(..., parse_dates=['Date'])` when loading to ensure proper datetime parsing.
