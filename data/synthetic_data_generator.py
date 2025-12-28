import pandas as pd
import numpy as np

# general configuration
num_rows = 25_000
dates = pd.date_range(start="1960-01-01", periods=num_rows, freq="B")
symbols = ["SYNTH"]  # Add more symbols as needed
start_prices = {"SYNTH": 200.0}  # set keys to match symbols
base_volumes = {"SYNTH": 500} # set keys to match symbols

S0 = 200.0 # Initial price
mu = 0.0002 # Daily drift
dt = 1.0 # Time step (1 trading day)

# volitility regime
vol_low = 0.008
vol_high = 0.015
regime_probs = [0.8, 0.2]

all_ohlcv = []

#np.random.seed(42) # for deterministic testing/debugging

for symbol in symbols:
    S0 = start_prices[symbol]
    base_volume = base_volumes[symbol]

    # Regime switching
    sigma_series = np.random.choice([vol_low, vol_high], size=num_rows, p=regime_probs)

    # Generate price series using GBM
    Z = np.random.normal(0, 1, num_rows)
    price = np.zeros(num_rows)
    price[0] = S0
    price[1:] = price[0] * np.exp(
        np.cumsum((mu - 0.5 * sigma_series[1:]**2) * dt + sigma_series[1:] * np.sqrt(dt) * Z[1:])
    )

    # Generate OHLC vectorized
    gap = np.random.normal(0.0, 0.2 * sigma_series * price)
    open_ = np.empty(num_rows)
    close = price.copy()
    open_[0] = price[0]
    open_[1:] = price[:-1] + gap[1:]

    intraday_range = np.random.exponential(scale=sigma_series * price)
    high = np.maximum(open_, close) + intraday_range
    low = np.minimum(open_, close) - intraday_range

    # Ensure numerical safety
    open_ = np.maximum(open_, 0.01)
    high = np.maximum(high, open_)
    low = np.maximum(low, 0.01)
    close = np.maximum(close, 0.01)

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

print(full_ohlcv)
full_ohlcv.to_csv("./data/synthetic.csv")
