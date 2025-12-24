import pandas as pd
import numpy as np

num_rows = 1000  # instead of 100
dates = pd.date_range(start="2025-01-01", periods=num_rows)
price = np.cumsum(np.random.randn(num_rows)) + 100 # simple random walk. can be swapped for brownian motion for better synthetic data.

# generate synthetic data
ohlcv = pd.DataFrame({
    'Open': price + np.random.rand(num_rows),
    'High': price + np.random.rand(num_rows) * 2,
    'Low': price - np.random.rand(num_rows) * 2,
    'Close': price + np.random.rand(num_rows),
    'Volume': np.random.randint(100, 1000, size=num_rows)
}, index=dates)

print(ohlcv)

ohlcv.to_csv("./data/synthetic.csv")