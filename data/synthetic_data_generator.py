
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# - `start_date`: starting timestamp for the generated series
# - `freq`: pandas frequency string controlling granularity (business days, days,
#           hourly, minutes, seconds, etc.)
# - `horizon_years`: target length in years used to derive `num_rows` to avoid
#                    accidentally generating extremely long time ranges
# Change these values to control the timescale and length of the synthetic data.
# ---------------------------------------------------------------------------
start_date = "2000-01-01"
freq = "D"  # Options: 'B' business days, 'D' days, 'W' weeks, 'M' months, 'H' hours, 'T' minutes, 'S' seconds
horizon_years = 5  # Adjust to control total span (in years)

# Derive number of rows from horizon + frequency to avoid excessively long spans
if freq == "B":
    num_rows = int(horizon_years * 252)
elif freq == "D":
    num_rows = int(horizon_years * 365)
elif freq == "W":
    num_rows = int(horizon_years * 52)
elif freq == "M":
    num_rows = int(horizon_years * 12)
elif freq == "H":
    num_rows = int(horizon_years * 365 * 24)
elif freq == "T":  # minutes
    num_rows = int(horizon_years * 365 * 24 * 60)
elif freq == "S":  # seconds
    num_rows = int(horizon_years * 365 * 24 * 60 * 60)
else:
    num_rows = 100_000  # fallback

# Build the pandas `Date` index for the requested range and frequency. The
# rest of the generator uses vectorized numpy arrays of shape
# `(num_rows, num_symbols)` for efficient simulation.
dates = pd.date_range(start=start_date, periods=num_rows, freq=freq)

# Symbols and per-symbol configuration. Keep these small when testing to
# reduce memory usage; the generator creates arrays sized (num_rows * num_symbols).
symbols = ["SYNTH", "TEST", "THEORY"]  # Add more symbols as needed
num_symbols = len(symbols)
symbol_indices = np.arange(num_symbols)

# Starting prices per symbol and a base volume vector used to synthesize
# volume that correlates with absolute returns.
S0_vec = np.array([200.0, 325.0, 86])  # starting prices for each symbol
base_volume_vec = np.array([80000, 50000, 112000])

# Annualized drift used in the GBM; small default for near-zero drift
mu_default = 0.0002
mu_vec = np.array([mu_default for _ in symbols])

# Time step in years-per-bar. Compute automatically from `freq` so volatility/drift scale
# sensibly when using daily/hours/minutes/seconds data.
if freq == "B":
    dt = 1.0 / 252.0
elif freq == "D":
    dt = 1.0 / 365.0
elif freq == "W":
    dt = 1.0 / 52.0
elif freq == "M":
    dt = 1.0 / 12.0
elif freq == "H":
    dt = 1.0 / (365.0 * 24.0)
elif freq == "T":
    dt = 1.0 / (365.0 * 24.0 * 60.0)
elif freq == "S":
    dt = 1.0 / (365.0 * 24.0 * 3600.0)
else:
    # Fallback to trading-day convention
    dt = 1.0 / 252.0

# Basic validation to catch accidental huge arrays or nonsensical dt values
if num_rows <= 0:
    raise ValueError("num_rows must be positive")
if not (0.0 < dt <= 1.0):
    raise ValueError(f"computed dt looks wrong: {dt}")
vol_low = 0.008
vol_high = 0.015
regime_probs = [0.8, 0.2]
market_corr = 0.35
gap_sigma_mult = 0.25
intraday_range_mult = 1.00

seed = None
rng = np.random.default_rng(seed)

# ---------------------------------------------------------------------------
# Volatility regime per-symbol time series
# `sigma_series` has shape (num_rows, num_symbols). Each column chooses between
# two volatility regimes independently for each symbol. This lets some symbols
# be volatile for stretches while others remain calm.
# ---------------------------------------------------------------------------
sigma_series = np.stack([
    rng.choice([vol_low, vol_high], size=num_rows, p=regime_probs)
    for _ in range(num_symbols)
], axis=1)  # shape: [num_rows, num_symbols]

# Shocks
Z_market = rng.normal(0, 1, num_rows)
Z_idio = rng.normal(0, 1, (num_rows, num_symbols))
Z = market_corr * Z_market[:, None] + np.sqrt(max(0.0, 1.0 - market_corr**2)) * Z_idio

# ---------------------------------------------------------------------------
# Generate log-return increments for a Geometric Brownian Motion (GBM).
# - We use the standard discretized GBM step: drift - 0.5 * sigma^2 + sigma * noise.
# - `gbm_increments` is (num_rows-1, num_symbols). We exponentiate the
#   cumulative sum to produce a price path anchored at `S0_vec`.
# ---------------------------------------------------------------------------
gbm_increments = (mu_vec - 0.5 * sigma_series[1:, :] ** 2) * dt + sigma_series[1:, :] * np.sqrt(dt) * Z[1:, :]
close = np.zeros((num_rows, num_symbols))
close[0, :] = S0_vec
close[1:, :] = S0_vec * np.exp(np.cumsum(gbm_increments, axis=0))

# --- REFACTORED TREND LOGIC ---
# Instead of 0.15 * S0 (Fixed Dollars), we use a percentage oscillation.
# 0.05 means the trend causes a +/- 5% swing in the price level.
# Make the sinusoidal trend less dominant and add per-symbol variation + LF noise
# Reduce amplitude so GBM noise still shows through, and add random phase per symbol
trend_pct_amplitude = 0.005  # +/-x% swing
trend_period = 1200

# Per-symbol random phase so symbols don't move identically
phases = rng.uniform(0, 2 * np.pi, size=(1, num_symbols))

# Base sinusoidal percentage factor (shape: [num_rows, num_symbols])
trend_t = (np.arange(num_rows)[:, None] * 2 * np.pi / trend_period) + phases
trend_sine = trend_pct_amplitude * np.sin(trend_t)

# Add a small low-frequency noise component (cumulative small shocks),
# normalized per-symbol and given a small weight so it doesn't dominate
lf_noise = rng.normal(loc=0.0, scale=0.2, size=(num_rows, num_symbols))
lf = np.cumsum(lf_noise, axis=0)
max_abs = np.max(np.abs(lf), axis=0)
max_abs[max_abs == 0] = 1.0
lf_scaled = lf / max_abs
lf_weight = 0.005

# Combined trend factor
trend_factor = trend_sine + lf_weight * lf_scaled

# Apply multiplicatively: Close = Close * (1 + trend_factor)
close = close * (1.0 + trend_factor)


# ---------------------------------------------------------------------------
# Create OHLC from the simulated close series
# - `gap_pct` introduces overnight/interval gaps between previous close and
#   current open.
# - `intraday_range` is sampled from an exponential to produce asymmetric
#   intraday moves; high/low are computed relative to open/close.
# All operations are vectorized across symbols and time for speed.
# ---------------------------------------------------------------------------
gap_pct = rng.normal(loc=0.0, scale=gap_sigma_mult * sigma_series, size=(num_rows, num_symbols))
open_ = np.empty((num_rows, num_symbols))
open_[0, :] = close[0, :]
open_[1:, :] = close[:-1, :] * (1.0 + gap_pct[1:, :])

intraday_range = rng.exponential(scale=intraday_range_mult * sigma_series * close, size=(num_rows, num_symbols))
high = np.maximum(open_, close) + intraday_range
low = np.minimum(open_, close) - intraday_range

# Ensure numerical safety
open_ = np.maximum(open_, 0.01)
close = np.maximum(close, 0.01)
high = np.maximum(high, np.maximum(open_, close))
low = np.maximum(low, 0.01)
low = np.minimum(low, np.minimum(open_, close))

# ---------------------------------------------------------------------------
# Volume: scale a base volume by the magnitude of the absolute return to create
# a simple relationship between volatility and traded volume.
# ---------------------------------------------------------------------------
abs_return = np.abs(np.log(close / open_))
volume = (base_volume_vec * (1 + 20 * abs_return)).astype(int)

# ---------------------------------------------------------------------------
# Flatten to long format DataFrame suitable for the backtester. We iterate
# over symbols to build per-symbol DataFrames and then concatenate. For very
# large `num_rows * num_symbols` this may require a lot of memory; consider
# streaming or writing per-symbol CSVs for extremely large simulations.
# ---------------------------------------------------------------------------
records = []
for j, symbol in enumerate(symbols):
    df = pd.DataFrame({
        "Date": dates,
        "Symbol": symbol,
        "Open": open_[:, j],
        "High": high[:, j],
        "Low": low[:, j],
        "Close": close[:, j],
        "Volume": volume[:, j],
        "Sigma": sigma_series[:, j],
    })
    records.append(df)

full_ohlcv = pd.concat(records).sort_values(["Date", "Symbol"]).set_index("Date")

# Show a small preview and write to disk
print(full_ohlcv.head())
full_ohlcv.to_csv("./data/synthetic.csv")
