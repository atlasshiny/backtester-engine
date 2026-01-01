import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PerformanceAnalytics:
    """Compute performance statistics and generate plots.

    Inputs
    ------
    - Portfolio.value_history: equity series updated by the broker.
    - trade_log: list of dictionaries (typically Broker.trade_log).

    Notes
    -----
    - If the engine is run in row-by-row multi-asset mode (one step per (Date, Symbol)),
      then "returns" are computed per event, not per date. For per-date returns,
      prefer running the engine with group_by_date=True.
    - Trade pairing logic is a simple FIFO match per symbol and assumes long-only.
      It does not fully support partial fills/scale-in/scale-out accounting.
    """

    def __init__(self):
        """Create a PerformanceAnalytics instance."""
        pass

    def analyze_and_plot(self, portfolio, data_set, plot: bool = True, save: bool = False, risk_free_rate: float = 0.0, trade_log=None, annualization_factor: int = 252):
        """
        Analyze portfolio performance, print statistics, and (optionally) plot results.

        Parameters
        ----------
        portfolio:
            Portfolio instance containing cash, positions, and value_history.
        data_set:
            Market data DataFrame used for the run (used for price plotting).
        plot:
            When True, show Matplotlib figures.
        save:
            When True, write ./trade_log.csv and ./backtest_metrics.csv.
        risk_free_rate:
            Risk-free rate used in Sharpe/Sortino calculations. This is treated as a
            per-period rate aligned to the return series being computed.
        trade_log:
            List[dict] with Broker log schema. Expected keys include:
            timestamp, symbol, side, qty, price, commission, slippage, order_type,
            limit_price, comment.

        Output
        ------
        Prints summary statistics to stdout and optionally shows/saves plots.

        Caveats
        -------
        Annualization uses sqrt(252) by default. This is only appropriate when the
        returns series represents daily steps.
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
            sharpe = (np.mean(returns) - risk_free_rate) / std_ret * np.sqrt(annualization_factor)
        else:
            sharpe = float('nan')
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")

        # sortino ratio (robust to division by zero or nan)
        downside_returns = returns[returns < 0] if len(returns) else np.array([])
        if len(downside_returns) > 0:
            downside_dev = np.std(downside_returns)
            if downside_dev > 0 and np.isfinite(downside_dev):
                sortino = (np.mean(returns) - risk_free_rate) / downside_dev * np.sqrt(annualization_factor)
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

        # Win Rate + trade PnL: Robust FIFO match per symbol
        trade_pairs = []
        if not trades.empty and 'symbol' in trades.columns:
            # Convert to list of dicts once for faster iteration
            trade_list = trades.to_dict('records')
            open_trades_by_symbol = {} # Stores lists of BUY dicts
            for trade_dict in trade_list:
                sym = trade_dict.get('symbol')
                side = trade_dict.get('side')
                qty_to_match = trade_dict.get('qty', 0)
                if sym is None or qty_to_match <= 0 or trade_dict['price'] is None:
                    continue
                if side == 'BUY':
                    # Add to the queue for this symbol
                    open_trades_by_symbol.setdefault(sym, []).append(trade_dict.copy())
                elif side == 'SELL':
                    opens = open_trades_by_symbol.get(sym, [])
                    # Loop until the SELL quantity is fully matched or we run out of BUYs
                    while qty_to_match > 0 and opens:
                        entry = opens[0]  # Look at the oldest BUY
                        available_qty = entry['qty']
                        # Determine how much we can match in this iteration
                        matched_qty = min(qty_to_match, available_qty)
                        # Calculate proportional PnL and commissions
                        share_of_entry = matched_qty / entry['qty']
                        share_of_exit = matched_qty / trade_dict['qty']
                        gross_pnl = (trade_dict['price'] - entry['price']) * matched_qty
                        entry_comm = float(entry.get('commission', 0.0) or 0.0) * share_of_entry
                        exit_comm = float(trade_dict.get('commission', 0.0) or 0.0) * share_of_exit
                        net_pnl = gross_pnl - entry_comm - exit_comm
                        trade_pairs.append({
                            'entry': entry,
                            'exit': trade_dict.copy(),
                            'qty': matched_qty,
                            'gross_pnl': gross_pnl,
                            'net_pnl': net_pnl
                        })
                        # Update remaining quantities
                        qty_to_match -= matched_qty
                        entry['qty'] -= matched_qty
                        # If the original BUY is fully used up, remove it from the queue
                        if entry['qty'] <= 0:
                            opens.pop(0)

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