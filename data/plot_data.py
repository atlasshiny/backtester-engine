import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, 'synthetic.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path, parse_dates=['Date'])
    symbols = df['Symbol'].unique()

    for symbol in symbols:
        df_sym = df[df['Symbol'] == symbol].copy()
        df_sym.set_index('Date', inplace=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_sym.index, df_sym['Close'], label=f'{symbol} Close')
        ax.set_title(f'{symbol} Close Price')
        ax.set_ylabel('Price')
        ax.legend()
        fig.tight_layout()

        filename = f'synthetic_plot_{symbol}.png'
        out = os.path.join(base, filename)
        fig.savefig(out)
        plt.close(fig)
        print(f'Saved plot for {symbol} to {out}')


if __name__ == '__main__':
    main()