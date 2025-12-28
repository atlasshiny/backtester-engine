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

        # sharpe ratio (include case for division by zero)
        sharpe = (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252)
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")

        # sortino ratio (include case for division by zero)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_dev = np.std(downside_returns)
            sortino = (np.mean(returns) - risk_free_rate) / downside_dev * np.sqrt(252)
        else:
            sortino = np.nan
        print(f"Annualized Sortino Ratio: {sortino:.2f}")

        # trade statistics
        trades = pd.DataFrame(portfolio.trade_log)
        total_trades = len(trades[trades['side'].isin(['BUY','SELL'])])

        # FIX LATER

        # winning_trades = trades[(trades['side'] == 'SELL') & (trades['price'] > 0)]
        # losing_trades = trades[(trades['side'] == 'SELL') & (trades['price'] > 0)]  # can refine by PnL
        # print(f"Total trades: {total_trades}")
        # print(f"Winning trades: {len(winning_trades)}")
        # print(f"Losing trades: {len(losing_trades)}")

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