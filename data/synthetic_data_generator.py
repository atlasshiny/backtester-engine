import pandas as pd
import numpy as np

num_rows = 25000  # instead of 100
dates = pd.date_range(start="1960-01-01", periods=num_rows)
price = np.cumsum(np.random.randn(num_rows)) + 100 # simple random walk. can be swapped for brownian motion for better synthetic data.

# generate synthetic data

# Generate OHLC, ensuring no negative values
open = np.maximum(price + np.random.rand(num_rows), 0)
close = np.maximum(price + np.random.rand(num_rows), 0)
high = np.maximum(np.maximum(open, close) + np.random.rand(num_rows) * 2, 0)
low = np.maximum(np.minimum(open, close) - np.random.rand(num_rows) * 2, 0)

ohlcv = pd.DataFrame({
    'Symbol': "SYNTH",
    'Open': open,
    'High': high,
    'Low': low,
    'Close': close,
    'Volume': np.random.randint(100, 1000, size=num_rows)
}, index=dates)

print(ohlcv)

ohlcv.to_csv("./data/synthetic.csv")