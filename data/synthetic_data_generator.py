
import pandas as pd
import numpy as np

# general configuration
num_rows = 100_000
dates = pd.date_range(start="1960-01-01", periods=num_rows, freq="B", unit="s")
symbols = ["SYNTH", "TEST"]  # Add more symbols as needed
num_symbols = len(symbols)
symbol_indices = np.arange(num_symbols)
S0_vec = np.array([200.0, 325.0])  # or [start_prices[s] for s in symbols]
base_volume_vec = np.array([500, 700])
mu_default = 0.0002
mu_vec = np.array([mu_default for _ in symbols])

dt = 1.0
vol_low = 0.008
vol_high = 0.015
regime_probs = [0.8, 0.2]
market_corr = 0.35
gap_sigma_mult = 0.25
intraday_range_mult = 1.00

seed = None
rng = np.random.default_rng(seed)

# Regime switching per symbol
sigma_series = np.stack([
    rng.choice([vol_low, vol_high], size=num_rows, p=regime_probs)
    for _ in range(num_symbols)
], axis=1)  # shape: [num_rows, num_symbols]

# Shocks
Z_market = rng.normal(0, 1, num_rows)
Z_idio = rng.normal(0, 1, (num_rows, num_symbols))
Z = market_corr * Z_market[:, None] + np.sqrt(max(0.0, 1.0 - market_corr**2)) * Z_idio

# GBM increments
gbm_increments = (mu_vec - 0.5 * sigma_series[1:, :] ** 2) * dt + sigma_series[1:, :] * np.sqrt(dt) * Z[1:, :]
close = np.zeros((num_rows, num_symbols))
close[0, :] = S0_vec
close[1:, :] = S0_vec * np.exp(np.cumsum(gbm_increments, axis=0))

# Add long-term trend channel
trend_amplitude = 0.15 * S0_vec
trend_period = 1200
trend = trend_amplitude * np.sin(np.arange(num_rows)[:, None] * 2 * np.pi / trend_period)
close *= (1 + trend / S0_vec)

# Generate OHLC vectorized
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

# volume
abs_return = np.abs(np.log(close / open_))
volume = (base_volume_vec * (1 + 20 * abs_return)).astype(int)

# Flatten to long format DataFrame
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

print(full_ohlcv.head())
full_ohlcv.to_csv("./data/synthetic.csv")
