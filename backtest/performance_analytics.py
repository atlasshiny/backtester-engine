import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PerformanceAnalytics:
    def __init__(self):
        pass

    def analyze_and_plot(self, portfolio, data_set, plot: bool = True, save: bool = False, risk_free_rate: float = 0.0, trade_log=None):
        """
        Analyze portfolio performance, print statistics, and plot results.
        Args:
            portfolio: The Portfolio instance.
            data_set (pd.DataFrame): The market data.
            plot (bool): Whether to plot results.
            save (bool): Whether to save results to file.
            risk_free_rate (float): Risk-free rate for Sharpe/Sortino.
            trade_log (list): List of trade dictionaries for analytics.
        """
        # convert value history to numpy array for calculations
        equity = np.array(portfolio.value_history, dtype=float)
        if len(equity) >= 2:
            returns = np.diff(equity) / equity[:-1]
        else:
            returns = np.array([])

        # basic metrics
        final_value = equity[-1] if len(equity) else float('nan')
        pnl = (final_value - portfolio.initial_cash) if np.isfinite(final_value) else float('nan')
        print(f"Final Portfolio Value: {final_value:.2f}")
        print(f"PnL: {pnl:.2f}")

        if not portfolio.positions:
            print("No outstanding positions")
        else:
            for sym, pos in portfolio.positions.items():
                print(f"Remaining {sym}: {pos.qty} units at avg price {pos.avg_price:.2f}")

        print("\nNOTICE: All trades include slippage and commission.")

        # drawdown
        if len(equity):
            running_max = np.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max
            max_drawdown = float(drawdowns.max())
        else:
            drawdowns = np.array([])
            max_drawdown = float('nan')
        print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")

        # sharpe ratio (robust to division by zero or nan)
        std_ret = np.std(returns) if len(returns) else float('nan')
        if len(returns) and std_ret > 0 and np.isfinite(std_ret):
            sharpe = (np.mean(returns) - risk_free_rate) / std_ret * np.sqrt(252)
        else:
            sharpe = float('nan')
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")

        # sortino ratio (robust to division by zero or nan)
        downside_returns = returns[returns < 0] if len(returns) else np.array([])
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
        trades = pd.DataFrame(trade_log) if trade_log is not None else pd.DataFrame()
        if not trades.empty:
            # normalize timestamp if present
            if 'timestamp' in trades.columns:
                trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors='coerce')
            total_trades = int(len(trades[trades['side'].isin(['BUY', 'SELL'])])) if 'side' in trades.columns else 0
            total_commission = float(trades['commission'].fillna(0.0).sum()) if 'commission' in trades.columns else 0.0
        else:
            total_trades = 0
            total_commission = 0.0

        # Win Rate + trade PnL: FIFO match per symbol (long-only)
        trade_pairs = []
        if not trades.empty and 'symbol' in trades.columns:
            open_trades_by_symbol = {}
            for _, row in trades.iterrows():
                sym = row.get('symbol')
                if sym is None:
                    continue
                if row['side'] == 'BUY' and row['qty'] > 0 and row['price'] is not None:
                    open_trades_by_symbol.setdefault(sym, []).append(row)
                elif row['side'] == 'SELL' and row['qty'] > 0 and row['price'] is not None:
                    opens = open_trades_by_symbol.get(sym, [])
                    if opens:
                        entry = opens.pop(0)
                        matched_qty = min(row['qty'], entry['qty'])
                        gross_pnl = (row['price'] - entry['price']) * matched_qty
                        entry_comm = float(entry.get('commission', 0.0) or 0.0)
                        exit_comm = float(row.get('commission', 0.0) or 0.0)
                        net_pnl = gross_pnl - entry_comm - exit_comm
                        trade_pairs.append({'entry': entry, 'exit': row, 'qty': matched_qty, 'gross_pnl': gross_pnl, 'net_pnl': net_pnl})

        if trade_pairs:
            net_pnls = np.array([tp['net_pnl'] for tp in trade_pairs], dtype=float)
            wins = net_pnls[net_pnls > 0]
            losses = net_pnls[net_pnls < 0]

            win_rate = (len(wins) / len(net_pnls)) * 100
            avg_win = float(np.mean(wins)) if len(wins) else float('nan')
            avg_loss = float(np.mean(losses)) if len(losses) else float('nan')

            gross_profit = float(np.sum(wins)) if len(wins) else 0.0
            gross_loss = float(np.sum(losses)) if len(losses) else 0.0
            profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else float('nan')
            expectancy = float(np.mean(net_pnls)) if len(net_pnls) else float('nan')

            # streaks
            max_consec_wins = 0
            max_consec_losses = 0
            current_wins = 0
            current_losses = 0
            for p in net_pnls:
                if p > 0:
                    current_wins += 1
                    current_losses = 0
                elif p < 0:
                    current_losses += 1
                    current_wins = 0
                else:
                    current_wins = 0
                    current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
                max_consec_losses = max(max_consec_losses, current_losses)

            print(f"Total Trades: {len(net_pnls)}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Profit Factor (net): {profit_factor:.3f}")
            print(f"Avg Win (net): {avg_win:.2f}")
            print(f"Avg Loss (net): {avg_loss:.2f}")
            print(f"Expectancy / Trade (net): {expectancy:.2f}")
            print(f"Total Commission: {total_commission:.2f}")
            print(f"Max Consecutive Wins: {max_consec_wins}")
            print(f"Max Consecutive Losses: {max_consec_losses}")

            # holding time (requires timestamps)
            entry_ts = pd.to_datetime([tp['entry'].get('timestamp') for tp in trade_pairs], errors='coerce')
            exit_ts = pd.to_datetime([tp['exit'].get('timestamp') for tp in trade_pairs], errors='coerce')
            holding = (exit_ts - entry_ts)
            holding = holding[~pd.isna(holding)]
            if len(holding):
                print(f"Avg Holding Time: {holding.mean()}")
                print(f"Median Holding Time: {holding.median()}")
        else:
            print(f"Total Trades: {total_trades}")
            print("Win Rate: nan%")
            print(f"Total Commission: {total_commission:.2f}")

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

            # drawdown (underwater)
            if len(drawdowns):
                plt.figure(figsize=(10, 3))
                plt.plot(drawdowns * -100.0, color='tab:red')
                plt.xlabel('Time (events)')
                plt.ylabel('Drawdown (%)')
                plt.title('Drawdown (Underwater)')
                plt.tight_layout()
                plt.show()

            # returns distribution
            if len(returns):
                plt.figure(figsize=(8, 4))
                plt.hist(returns, bins=50, color='tab:purple', alpha=0.8)
                plt.xlabel('Per-step returns')
                plt.ylabel('Frequency')
                plt.title('Returns Distribution')
                plt.tight_layout()
                plt.show()

            # trade PnL distribution + per-symbol PnL
            if trade_pairs:
                net_pnls = np.array([tp['net_pnl'] for tp in trade_pairs], dtype=float)
                plt.figure(figsize=(8, 4))
                plt.hist(net_pnls, bins=40, color='tab:green', alpha=0.8)
                plt.xlabel('Trade PnL (net)')
                plt.ylabel('Frequency')
                plt.title('Trade PnL Distribution (Net)')
                plt.tight_layout()
                plt.show()

                # per-symbol PnL (net)
                sym_pnl = {}
                for tp in trade_pairs:
                    sym = tp['entry'].get('symbol')
                    if sym is None:
                        continue
                    sym_pnl[sym] = sym_pnl.get(sym, 0.0) + float(tp['net_pnl'])
                if sym_pnl:
                    plt.figure(figsize=(10, 4))
                    keys = list(sym_pnl.keys())
                    vals = [sym_pnl[k] for k in keys]
                    plt.bar(keys, vals, color='tab:blue', alpha=0.8)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('PnL (net)')
                    plt.title('PnL by Symbol (Net)')
                    plt.tight_layout()
                    plt.show()

            # price vs equity (only plot a single symbol's Close if multi-asset)
            if 'Symbol' in data_set.columns:
                first_symbol = data_set['Symbol'].iloc[0]
                close_series = data_set[data_set['Symbol'] == first_symbol]['Close'].values
                title_suffix = f" ({first_symbol})"
            else:
                close_series = data_set['Close'].values
                title_suffix = ""

            plt.figure(figsize=(12, 6))
            ax1 = plt.gca()
            ax1.plot(equity, color='tab:blue', label='Equity Curve')
            ax1.set_xlabel('Time (events)')
            ax1.set_ylabel('Portfolio Value', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.plot(close_series, color='tab:orange', label='Price (Close)')
            ax2.set_ylabel('Price', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            plt.title('Equity and Price Curve' + title_suffix)
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
                "Total Trades": total_trades,
                "Total Commission": total_commission
            }
            pd.DataFrame([metrics]).to_csv("./backtest_metrics.csv", index=False)