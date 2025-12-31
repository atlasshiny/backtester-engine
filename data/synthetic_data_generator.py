import pandas as pd
import numpy as np

# general configuration
num_rows = 25_000
dates = pd.date_range(start="1960-01-01", periods=num_rows, freq="B")
symbols = ["SYNTH", "TEST"]  # Add more symbols as needed
start_prices = {"SYNTH": 200.0,
                "TEST": 325.0}  # keys must match symbols
base_volumes = {"SYNTH": 500,
                "TEST": 700}  # keys must match symbols

# per-symbol drift (defaults to mu_default if not specified)
mu_default = 0.0002  # Daily drift
mu_by_symbol = {}

dt = 1.0  # Time step (1 trading day)

# volitility regime
vol_low = 0.008
vol_high = 0.015
regime_probs = [0.8, 0.2]

# Optional cross-asset correlation (0.0 = independent, 1.0 = perfectly correlated)
market_corr = 0.35

# Gap and intraday range shaping
gap_sigma_mult = 0.25  # controls typical open gap size vs sigma
intraday_range_mult = 1.00  # controls intraday range magnitude

all_ohlcv = []

# Reproducibility
seed = None  # set to an int (e.g., 42) for deterministic output
rng = np.random.default_rng(seed)

# Shared market shock for correlation across symbols
Z_market = rng.normal(0, 1, num_rows)

for symbol in symbols:
    S0 = start_prices[symbol]
    base_volume = base_volumes[symbol]
    mu = mu_by_symbol.get(symbol, mu_default)

    # Regime switching
    sigma_series = rng.choice([vol_low, vol_high], size=num_rows, p=regime_probs)

    # Generate price series using GBM
    Z_idio = rng.normal(0, 1, num_rows)
    Z = market_corr * Z_market + np.sqrt(max(0.0, 1.0 - market_corr**2)) * Z_idio
    close = np.zeros(num_rows)
    close[0] = S0
    close[1:] = close[0] * np.exp(
        np.cumsum((mu - 0.5 * sigma_series[1:] ** 2) * dt + sigma_series[1:] * np.sqrt(dt) * Z[1:])
    )

    # Generate OHLC vectorized
    # Open is previous close plus a percentage gap (more realistic than absolute gap)
    gap_pct = rng.normal(loc=0.0, scale=gap_sigma_mult * sigma_series, size=num_rows)
    open_ = np.empty(num_rows)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1.0 + gap_pct[1:])

    # Intraday range scales with price level
    intraday_range = rng.exponential(scale=intraday_range_mult * sigma_series * close)
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
    volume = (base_volume * (1 + 20 * abs_return)).astype(int)

    # assemble dataframe
    ohlcv = pd.DataFrame({
        "Symbol": symbol,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Sigma": sigma_series,
    }, index=dates)

    all_ohlcv.append(ohlcv)

# combine all symbols into one DataFrame (long format)
full_ohlcv = pd.concat(all_ohlcv).reset_index().rename(columns={"index": "Date"})
full_ohlcv = full_ohlcv.sort_values(["Date", "Symbol"]).set_index("Date")

print(full_ohlcv.head())
full_ohlcv.to_csv("./data/synthetic.csv")
