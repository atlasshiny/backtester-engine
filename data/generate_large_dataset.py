import pandas as pd
import numpy as np

def generate_large_dataset(n_rows=10_000_000, n_symbols=100):
    print(f"Generating {n_rows} rows across {n_symbols} symbols...")
    
    # Create symbols (e.g., SYM00, SYM01...)
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    
    # Generate dates (10 million minutes is ~19 years of 24/7 data)
    dates = pd.date_range("2000-01-01", periods=n_rows // n_symbols, freq="min")
    
    # repeat dates for each symbol to maintain the long-format structure
    data = {
        'Date': np.tile(dates, n_symbols),
        'Symbol': np.repeat(symbols, len(dates)),
        'Close': np.cumprod(1 + np.random.normal(0, 0.0001, n_rows)),
        'High': 0, 'Low': 0, 'Open': 0, 'Volume': 1000
    }
    
    df = pd.DataFrame(data)
    # Important: Engine expects data sorted by Date then Symbol
    df = df.sort_values(['Date', 'Symbol'])
    df.to_csv("./data/stress_test.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    generate_large_dataset()