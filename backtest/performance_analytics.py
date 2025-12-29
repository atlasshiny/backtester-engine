import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PerformanceAnalytics:
    def __init__(self):
        pass

    def analyze_and_plot(self, portfolio, data_set, plot: bool = True, save: bool = False, risk_free_rate: float = 0.0):
        # convert value history to numpy array for calculations 
        equity = np.array(portfolio.value_history)
        returns = np.diff(equity) / equity[:-1]

        # basic metrics 
        final_value = equity[-1]
        pnl = final_value - portfolio.initial_cash
        print(f"Final Portfolio Value: {final_value:.2f}")
        print(f"PnL: {pnl:.2f}")

        if not portfolio.positions:
            print("No outstanding positions")
        else:
            for sym, pos in portfolio.positions.items():
                print(f"Remaining {sym}: {pos['amount']} units at avg price {pos['price']:.2f}")

        print("\nNOTICE: All trades include slippage and commission.")

        # drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_drawdown = drawdowns.max()
        print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")

        # sharpe ratio (robust to division by zero or nan)
        std_ret = np.std(returns)
        if std_ret > 0 and np.isfinite(std_ret):
            sharpe = (np.mean(returns) - risk_free_rate) / std_ret * np.sqrt(252)
        else:
            sharpe = float('nan')
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")

        # sortino ratio (robust to division by zero or nan)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_dev = np.std(downside_returns)
            if downside_dev > 0 and np.isfinite(downside_dev):
                sortino = (np.mean(returns) - risk_free_rate) / downside_dev * np.sqrt(252)
            else:
                sortino = float('nan')
        else:
            sortino = float('nan')
        print(f"Annualized Sortino Ratio: {sortino:.2f}")

        # trade statistics
        trades = pd.DataFrame(portfolio.trade_log)
        total_trades = len(trades[trades['side'].isin(['BUY','SELL'])])

        # Improved Win Rate calculation: pair each SELL with its corresponding BUY, compute per-trade PnL
        # Assumptions
        # - Long-only
        # - FIFO matching
        # - No partial exits
        # - Single-symbol trading
        # - Commission/slippage ignored in per-trade PnL
        trade_pairs = []
        open_trades = []
        for _, row in trades.iterrows():
            if row['side'] == 'BUY' and row['qty'] > 0 and row['price'] is not None:
                open_trades.append(row)
            elif row['side'] == 'SELL' and row['qty'] > 0 and row['price'] is not None and open_trades:
                entry = open_trades.pop(0)
                # PnL = (sell price - buy price) * qty (assume qty matches)
                pnl = (row['price'] - entry['price']) * min(row['qty'], entry['qty'])
                trade_pairs.append({'entry': entry, 'exit': row, 'pnl': pnl})
        if trade_pairs:
            wins = [tp for tp in trade_pairs if tp['pnl'] > 0]
            win_rate = (len(wins) / len(trade_pairs)) * 100
        else:
            win_rate = float('nan')
        print(f"Win Rate: {win_rate:.2f}%")

        # plots
        if plot:
            # equity curve
            plt.figure(figsize=(10, 5))
            plt.plot(equity, label='Equity Curve', color='tab:blue')
            plt.xlabel('Time (events)')
            plt.ylabel('Portfolio Value')
            plt.title('Equity Curve')
            plt.legend()
            plt.show()

            # price vs equity
            plt.figure(figsize=(12, 6))
            ax1 = plt.gca()
            ax1.plot(equity, color='tab:blue', label='Equity Curve')
            ax1.set_xlabel('Time (events)')
            ax1.set_ylabel('Portfolio Value', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.plot(data_set['Close'].values, color='tab:orange', label='Price (Close)')
            ax2.set_ylabel('Price', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            plt.title('Equity and Price Curve')
            fig = plt.gcf()
            fig.tight_layout()
            plt.show()

        if save:
            trades.to_csv("./trade_log.csv", index=False)
            metrics = {
                "Final Value": final_value,
                "PnL": pnl,
                "Max Drawdown": max_drawdown,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Total Trades": total_trades
            }
            pd.DataFrame([metrics]).to_csv("./backtest_metrics.csv", index=False)