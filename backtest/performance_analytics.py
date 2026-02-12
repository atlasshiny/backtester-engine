import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Literal
from numba import njit

try:  # Optional GPU acceleration
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:  # noqa: BLE001 - import-time probe only
    cp = None
    _CUPY_AVAILABLE = False


def _select_array_module(prefer_gpu: Literal["auto", True, False], length: int, gpu_min_size: int) -> object:
    if prefer_gpu is False:
        return np
    if _CUPY_AVAILABLE and (prefer_gpu is True or (prefer_gpu == "auto" and length >= gpu_min_size)):
        return cp
    return np


def _to_numpy(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

@njit
def _calculate_consecutive_streaks(is_win: np.ndarray, is_loss: np.ndarray) -> tuple:
    """Calculate maximum consecutive wins and losses using Numba."""
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for win, loss in zip(is_win, is_loss):
        if win:
            current_wins += 1
            current_losses = 0
            if current_wins > max_consec_wins:
                max_consec_wins = current_wins
        elif loss:
            current_losses += 1
            current_wins = 0
            if current_losses > max_consec_losses:
                max_consec_losses = current_losses
        else:
            current_wins = 0
            current_losses = 0
    return max_consec_wins, max_consec_losses

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

    def analyze_and_plot(self, portfolio, data_set, plot: bool = True, save: bool = False, risk_free_rate: float = 0.0, trade_log=None, annualization_factor: int | None = 252, max_points: int = 1000, prefer_gpu: Literal["auto", True, False] = "auto", gpu_min_size: int = 10000):
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
        annualization_factor:
            Integer factor for annualizing Sharpe/Sortino ratios (default: 252).
        max_points:
            Maximum number of points to plot in charts (downsampling for performance).
        prefer_gpu:
            "auto" (default) uses CuPy when available and the equity array is at least gpu_min_size.
            True forces GPU when possible; False keeps CPU only.
        gpu_min_size:
            Minimum array length required before attempting GPU acceleration in auto mode.

        Output
        ------
        Prints summary statistics to stdout and optionally shows/saves plots.

        Caveats
        -------
        Annualization uses sqrt(252) by default. This is only appropriate when the
        returns series represents daily steps.
        """
        # Determine annualization: if caller provided None, try to infer from the
        # data_set['Date'] timestamps. Otherwise use the provided factor.
        def _infer_annualization(dates) -> float:
            try:
                idx = pd.to_datetime(dates)
                if len(idx) < 2:
                    return 252.0
                deltas = (idx[1:] - idx[:-1]).to_series().dt.total_seconds()
                med = deltas.median()
                if med <= 0 or pd.isna(med):
                    return 252.0
                seconds_per_year = 365.0 * 24.0 * 3600.0
                periods_per_year = seconds_per_year / med
                return float(periods_per_year)
            except Exception:
                return 252.0

        if annualization_factor is None:
            # Support either a pandas DataFrame or a dict of arrays for data_set
            if isinstance(data_set, pd.DataFrame) and 'Date' in data_set.columns:
                annualization = _infer_annualization(data_set['Date'].values)
            elif isinstance(data_set, dict) and 'Date' in data_set:
                annualization = _infer_annualization(data_set['Date'])
            else:
                annualization = 252.0
        else:
            annualization = float(annualization_factor)
        xp = _select_array_module(prefer_gpu, len(portfolio.value_history), gpu_min_size)

        # convert value history to chosen array module for calculations
        equity = xp.asarray(portfolio.value_history, dtype=xp.float64)
        equity_np = _to_numpy(equity)
        if equity.size >= 2:
            returns = xp.diff(equity) / equity[:-1]
            returns_np = _to_numpy(returns)
        else:
            returns = xp.asarray([], dtype=xp.float64)
            returns_np = np.array([])

        # basic metrics
        final_value = float(equity_np[-1]) if equity_np.size else float('nan')
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
        if equity.size:
            # running_max = xp.maximum.accumulate(equity) can't use cupy because feature hasn't been added yet from numpy
            if hasattr(equity, 'get'):  # Use this to check if it's a CuPy array, else continute with standard numpy
                running_max = np.maximum.accumulate(equity.get()) # Move to CPU, calculate, move back
                running_max = xp.asarray(running_max) 
            else:
                running_max = xp.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max
            max_drawdown = float(_to_numpy(drawdowns).max())
        else:
            drawdowns = xp.asarray([], dtype=xp.float64)
            max_drawdown = float('nan')
        print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")

        # Prepare arrays from data_set early to avoid repeated pandas indexing later.
        symbol_arr = None
        close_arr = np.asarray([])
        date_arr = None
        if isinstance(data_set, pd.DataFrame):
            symbol_arr = data_set['Symbol'].to_numpy() if 'Symbol' in data_set.columns else None
            close_arr = data_set['Close'].to_numpy() if 'Close' in data_set.columns else np.asarray([])
            date_arr = pd.to_datetime(data_set['Date']).to_numpy() if 'Date' in data_set.columns else None
        elif isinstance(data_set, dict):
            # assume dict of arrays
            symbol_arr = np.asarray(data_set.get('Symbol')) if data_set.get('Symbol') is not None else None
            close_arr = np.asarray(data_set.get('Close')) if data_set.get('Close') is not None else np.asarray([])
            try:
                date_arr = pd.to_datetime(data_set.get('Date')) if data_set.get('Date') is not None else None
            except Exception:
                date_arr = None

        # sharpe ratio (robust to division by zero or nan)
        if len(returns_np):
            # Use GPU computations when `returns` is a CuPy array (has `get`)
            if hasattr(returns, 'get'):
                mean_ret = float(xp.mean(returns))
                std_ret = float(xp.std(returns))
            else:
                mean_ret = float(np.mean(returns_np))
                std_ret = float(np.std(returns_np))
        else:
            mean_ret = float('nan')
            std_ret = float('nan')

        if len(returns_np) and std_ret > 0 and np.isfinite(std_ret):
            sharpe = (mean_ret - risk_free_rate) / std_ret * np.sqrt(annualization)
        else:
            sharpe = float('nan')
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")

        # sortino ratio (robust to division by zero or nan)
        if len(returns_np):
            if hasattr(returns, 'get'):
                downside = returns[returns < 0]
                downside_dev = float(xp.std(downside)) if downside.size else float('nan')
            else:
                downside = returns_np[returns_np < 0]
                downside_dev = float(np.std(downside)) if len(downside) else float('nan')
            if downside_dev > 0 and np.isfinite(downside_dev):
                sortino = (mean_ret - risk_free_rate) / downside_dev * np.sqrt(annualization)
            else:
                sortino = float('nan')
        else:
            sortino = float('nan')
        print(f"Annualized Sortino Ratio: {sortino:.2f}")

        # trade statistics
        trades = pd.DataFrame(trade_log) if trade_log is not None else pd.DataFrame()
        if not trades.empty:
            # Work with numpy arrays for faster processing
            sides = trades['side'].values if 'side' in trades.columns else np.array([])
            commissions = trades['commission'].fillna(0.0).values if 'commission' in trades.columns else np.array([])
            
            total_trades = int(np.sum(np.isin(sides, ['BUY', 'SELL']))) if len(sides) else 0
            total_commission = float(np.sum(commissions)) if len(commissions) else 0.0
        else:
            total_trades = 0
            total_commission = 0.0
            sides = np.array([])
            commissions = np.array([])

        # Win Rate + trade PnL: Robust FIFO match per symbol using NumPy for speed
        trade_pairs = []
        if trade_log and len(trade_log) > 0:
            # Work directly with list of dicts (already provided by broker)
            open_trades_by_symbol = {} # Stores lists of BUY dicts
            for trade_dict in trade_log:
                sym = trade_dict.get('symbol')
                side = trade_dict.get('side')
                qty_to_match = trade_dict.get('qty', 0)
                if sym is None or qty_to_match <= 0 or trade_dict.get('price') is None:
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

            # streaks - optimized with Numba
            is_win = net_pnls > 0
            is_loss = net_pnls < 0
            max_consec_wins, max_consec_losses = _calculate_consecutive_streaks(is_win, is_loss)

            print(f"Total Trades: {len(net_pnls)}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Profit Factor (net): {profit_factor:.3f}")
            print(f"Avg Win (net): {avg_win:.2f}")
            print(f"Avg Loss (net): {avg_loss:.2f}")
            print(f"Expectancy / Trade (net): {expectancy:.2f}")
            print(f"Total Commission: {total_commission:.2f}")
            print(f"Max Consecutive Wins: {max_consec_wins}")
            print(f"Max Consecutive Losses: {max_consec_losses}")

            # holding time (requires timestamps) - use NumPy for speed
            if trade_pairs:
                entry_timestamps = [tp['entry'].get('timestamp') for tp in trade_pairs]
                exit_timestamps = [tp['exit'].get('timestamp') for tp in trade_pairs]
                # Convert to numpy datetime64 for fast operations
                entry_ts = pd.to_datetime(entry_timestamps, errors='coerce')
                exit_ts = pd.to_datetime(exit_timestamps, errors='coerce')
                holding = (exit_ts - entry_ts)
                valid_holding = holding[~pd.isna(holding)]
                if len(valid_holding):
                    print(f"Avg Holding Time: {valid_holding.mean()}")
                    print(f"Median Holding Time: {valid_holding.median()}")
        else:
            print(f"Total Trades: {total_trades}")
            print("Win Rate: nan%")
            print(f"Total Commission: {total_commission:.2f}")

        # plots
        if plot:
            # When there are too many lines of data, use the 'fast' preset for better performance
            if len(data_set) > 25000:
                plt.style.use('fast')
                # Further optimize the Agg backend for large financial arrays
                plt.rcParams['agg.path.chunksize'] = 10000

            # Prepare NumPy arrays for faster vectorized operations and
            # deduplicate equity curve: in group_by_date mode with N symbols,
            # `value_history` is updated N times per date.
            if isinstance(data_set, pd.DataFrame):
                symbol_arr = data_set['Symbol'].to_numpy() if 'Symbol' in data_set.columns else None
                close_arr = data_set['Close'].to_numpy() if 'Close' in data_set.columns else np.asarray([])
                date_arr = pd.to_datetime(data_set['Date']).to_numpy() if 'Date' in data_set.columns else None
                num_symbols = int(np.unique(symbol_arr).size) if symbol_arr is not None else 1
            else:
                # Fallback: treat dataset as array-like or structured array
                try:
                    symbol_arr = np.asarray(data_set['Symbol'])
                except Exception:
                    symbol_arr = None
                try:
                    close_arr = np.asarray(data_set['Close'])
                except Exception:
                    close_arr = np.asarray(data_set)
                date_arr = None
                num_symbols = 1

            equity_deduplicated = equity_np[::num_symbols] if num_symbols > 1 else equity_np

            fig, axes = plt.subplots(4, 2, figsize=(20, 16))
            axes = axes.flatten()

            # Downsampling helper
            def downsample(arr):
                arr = np.asarray(arr)
                if len(arr) > max_points:
                    idx = np.linspace(0, len(arr) - 1, max_points, dtype=int)
                    return arr[idx]
                return arr

            # 1) Equity curve
            ax1 = axes[0]
            eq_plot = downsample(equity_deduplicated)
            ax1.plot(eq_plot, label='Equity Curve', color='tab:blue', rasterized=True)
            ax1.set_xlabel('Time (events)')
            ax1.set_ylabel('Portfolio Value')
            ax1.set_title('Equity Curve')
            ax1.legend()
            ax1.grid(False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            # 2) Underwater plot (shaded drawdown)
            if len(equity_deduplicated):
                eq_dd = downsample(equity_deduplicated)
                running_max_dedup = np.maximum.accumulate(eq_dd)
                drawdowns_dedup = (running_max_dedup - eq_dd) / running_max_dedup
                ax2 = axes[1]
                ax2.fill_between(np.arange(len(drawdowns_dedup)), drawdowns_dedup * -100.0, color='tab:red', alpha=0.5)
                ax2.plot(drawdowns_dedup * -100.0, color='tab:red', rasterized=True)
                ax2.set_xlabel('Time (events)')
                ax2.set_ylabel('Drawdown (%)')
                ax2.set_title('Underwater Plot (Drawdown)')
                ax2.grid(False)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)

            # 3) Returns distribution
            if len(returns_np):
                ax3 = axes[2]
                ret_plot = downsample(returns_np)
                ax3.hist(ret_plot, bins=50, color='tab:purple', alpha=0.8)
                ax3.set_xlabel('Per-step returns')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Returns Distribution')
                ax3.grid(False)
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)

            # 4) Monthly returns heatmap
            if len(equity_deduplicated) > 1:
                # Use the precomputed `date_arr` when available to avoid repeated
                # pandas indexing and conversions.
                if date_arr is not None:
                    dates = date_arr[::num_symbols] if num_symbols > 1 else date_arr
                    dates = pd.to_datetime(dates)
                    # Align lengths
                    min_len = min(len(dates), len(equity_deduplicated))
                    eq_df = pd.DataFrame({'Equity': equity_deduplicated[:min_len]}, index=dates[:min_len])
                    monthly_returns = eq_df['Equity'].resample('ME').last().pct_change()
                    # Directly build pivot table for heatmap
                    pivot_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack(fill_value=np.nan)
                    ax4 = axes[3]
                    sns.heatmap(pivot_table, annot=False, cmap='RdYlGn', ax=ax4, cbar=True)
                    ax4.set_title('Monthly Returns Heatmap')
                    ax4.set_xlabel('Month')
                    ax4.set_ylabel('Year')

            # 5) Trade PnL distribution
            if trade_pairs:
                net_pnls = np.array([tp['net_pnl'] for tp in trade_pairs], dtype=float)
                ax5 = axes[4]
                ax5.hist(net_pnls, bins=40, color='tab:green', alpha=0.8)
                ax5.set_xlabel('Trade PnL (net)')
                ax5.set_ylabel('Frequency')
                ax5.set_title('Trade PnL Distribution (Net)')
                ax5.grid(False)
                ax5.spines['top'].set_visible(False)
                ax5.spines['right'].set_visible(False)

                # 6) Per-symbol PnL - optimized aggregation
                sym_pnl = {}
                for tp in trade_pairs:
                    sym = tp['entry'].get('symbol')
                    if sym is not None:
                        sym_pnl[sym] = sym_pnl.get(sym, 0.0) + float(tp['net_pnl'])
                if sym_pnl:
                    ax6 = axes[5]
                    symbols = list(sym_pnl.keys())
                    pnls = np.array([sym_pnl[s] for s in symbols], dtype=float)
                    ax6.bar(range(len(symbols)), pnls, color='tab:blue', alpha=0.8)
                    ax6.set_xticks(range(len(symbols)))
                    ax6.set_xticklabels(symbols, rotation=45, ha='right')
                    ax6.set_ylabel('PnL (net)')
                    ax6.set_title('PnL by Symbol (Net)')
                    ax6.grid(False)
                    ax6.spines['top'].set_visible(False)
                    ax6.spines['right'].set_visible(False)

            # 7) Price vs Equity (dual axis) - work with precomputed arrays
            if symbol_arr is not None and symbol_arr.size > 0:
                first_symbol = symbol_arr[0]
                mask = (symbol_arr == first_symbol)
                close_series = close_arr[mask]
                title_suffix = f" ({first_symbol})"
            else:
                close_series = close_arr
                title_suffix = ""
            # Resample close_series to match equity_deduplicated length
            if len(close_series) != len(equity_deduplicated):
                indices_orig = np.linspace(0, len(close_series) - 1, len(close_series))
                indices_new = np.linspace(0, len(close_series) - 1, len(equity_deduplicated))
                close_series = np.interp(indices_new, indices_orig, close_series)
            eq7_plot = downsample(equity_deduplicated)
            close_plot = downsample(close_series)
            ax7 = axes[6]
            ax7_twin = ax7.twinx()
            ax7.plot(eq7_plot, color='tab:blue', label='Equity Curve', rasterized=True)
            ax7.set_xlabel('Time (events)')
            ax7.set_ylabel('Portfolio Value', color='tab:blue')
            ax7.tick_params(axis='y', labelcolor='tab:blue')
            ax7_twin.plot(close_plot, color='tab:orange', label='Price (Close)', rasterized=True)
            ax7_twin.set_ylabel('Price', color='tab:orange')
            ax7_twin.tick_params(axis='y', labelcolor='tab:orange')
            ax7.set_title('Equity and Price Curve' + title_suffix)
            ax7.grid(False)
            ax7.spines['top'].set_visible(False)
            ax7.spines['right'].set_visible(False)

            # 8) Rolling Sharpe Ratio
            if len(returns_np) > 20:
                window = min(60, len(returns_np))
                rolling_mean = pd.Series(returns_np).rolling(window=window).mean()
                rolling_std = pd.Series(returns_np).rolling(window=window).std()
                rolling_sharpe = (rolling_mean - risk_free_rate) / (rolling_std + 1e-8) * np.sqrt(annualization)
                ax8 = axes[7]
                sharpe_plot = downsample(rolling_sharpe.dropna())
                ax8.plot(sharpe_plot, color='tab:cyan', label=f'Rolling Sharpe ({window})')
                ax8.set_xlabel('Time (events)')
                ax8.set_ylabel('Sharpe Ratio')
                ax8.set_title('Rolling Sharpe Ratio')
                ax8.legend()
                ax8.grid(False)
                ax8.spines['top'].set_visible(False)
                ax8.spines['right'].set_visible(False)

            plt.tight_layout()
            output_path = './backtest_results.png'
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"Chart saved to {output_path}")
            plt.close()

        if save:
            # Only convert to DataFrame when saving to CSV
            if trade_log:
                trades_df = pd.DataFrame(trade_log)
                trades_df.to_csv("./trade_log.csv", index=False)
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
        
        # Prepare stats dictionary for testability
        stats = {
            'FinalValue': final_value,
            'TotalPnL': pnl,
            'MaxDrawdown': max_drawdown,
            'SharpeRatio': sharpe,
            'SortinoRatio': sortino,
            'TotalTrades': len(trade_pairs) if trade_pairs else total_trades,
            'TotalCommission': total_commission,
            'WinRate': win_rate/100 if trade_pairs else float('nan'),
            'ProfitFactor': profit_factor if trade_pairs else float('nan'),
            'AvgWin': avg_win if trade_pairs else float('nan'),
            'AvgLoss': avg_loss if trade_pairs else float('nan'),
            'Expectancy': expectancy if trade_pairs else float('nan'),
            'MaxConsecWins': max_consec_wins if trade_pairs else 0,
            'MaxConsecLosses': max_consec_losses if trade_pairs else 0
        }
        return stats