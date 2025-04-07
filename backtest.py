# backtest.py
import backtrader as bt
import datetime
import yfinance as yf
import pandas as pd
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Strategy Definition (Using internal indicators optimized by Cerebro) ---
class CombinedStrategy(bt.Strategy):
    """
    Combines SMA Crossover, RSI, and MACD confirmation for signals.
    Parameters are optimized via cerebro.optstrategy.
    """
    # Define parameters with defaults (will be overridden by optstrategy)
    params = (
        ('short_window', 12),
        ('long_window', 26),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('macd_short', 12),
        ('macd_long', 26),
        ('macd_signal', 9),
        ('printlog', False), # Disable per-run logging during optimization
    )

    def __init__(self):
        """Initialize indicators and strategy state."""
        self.dataclose = self.datas[0].close
        self.order = None

        # Indicators - Ensure they use self.p to get optimized params
        self.sma_short = bt.indicators.SimpleMovingAverage(period=self.p.short_window)
        self.sma_long = bt.indicators.SimpleMovingAverage(period=self.p.long_window)
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(
            self.datas[0],
            period_me1=self.p.macd_short,
            period_me2=self.p.macd_long,
            period_signal=self.p.macd_signal
        )
        self.sma_crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        # Log parameters only once at init if needed (can be verbose)
        # self.log(f"Init Params: SW={self.p.short_window}, LW={self.p.long_window}, ... ")

    def log(self, txt, dt=None, doprint=False):
        """Logging function (typically disabled during optimization)"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """Handles order notifications."""
        if order.status in [order.Submitted, order.Accepted]: return
        # Log completions/rejections only if needed for debugging opt runs
        # if order.status in [order.Completed]: self.log(...)
        # elif order.status in [order.Canceled, order.Margin, order.Rejected]: self.log(...)
        self.order = None

    def notify_trade(self, trade):
        """Handles trade notifications."""
        if not trade.isclosed: return
        # Log P/L only if needed for debugging opt runs
        # self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        """Define the core strategy logic using combined indicators."""
        if self.order: return # Check if an order is pending

        # Define Signal Conditions
        sma_golden_cross = self.sma_crossover > 0
        sma_death_cross = self.sma_crossover < 0
        macd_bullish_cross = self.macd_crossover > 0
        macd_bearish_cross = self.macd_crossover < 0
        rsi_not_overbought = self.rsi < self.p.rsi_overbought
        rsi_not_oversold = self.rsi > self.p.rsi_oversold

        if not self.position: # No position held
            # BUY condition
            if sma_golden_cross and macd_bullish_cross and rsi_not_overbought:
                 self.order = self.buy()
                 # self.log(f'BUY CREATED, Price: {self.dataclose[0]:.2f}')
        else: # Position held
            # SELL condition
            if sma_death_cross and macd_bearish_cross and rsi_not_oversold:
                self.order = self.sell()
                # self.log(f'SELL CREATED, Price: {self.dataclose[0]:.2f}')

    # No stop method needed, results collected by Cerebro analyzers

# --- Data Fetching (Keep the improved version) ---
def fetch_data(symbol, start_date, end_date):
    """Fetch historical data using yfinance and clean columns for backtrader."""
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
    logger.info(f"Raw data columns from yfinance: {df.columns}")
    if df.empty: logger.error(f"No data found for {symbol}..."); return None
    if isinstance(df.columns, pd.MultiIndex):
        logger.info("DataFrame has MultiIndex columns, flattening...")
        df.columns = df.columns.get_level_values(0)
        logger.info(f"Columns after flattening MultiIndex: {list(df.columns)}")
    df.columns = [str(col).lower() for col in df.columns]
    logger.info(f"Columns after forcing lowercase: {list(df.columns)}")
    expected_cols = {'open', 'high', 'low', 'close', 'adj close', 'volume'}
    final_cols = [col for col in expected_cols if col in df.columns]
    if 'close' not in final_cols: logger.error("CRITICAL: 'close' column missing..."); return None
    df = df[final_cols]
    logger.info(f"Processed {len(df)} rows. Final columns for backtrader: {list(df.columns)}")
    return df

# --- Main Execution ---
if __name__ == '__main__':
    cerebro = bt.Cerebro(optreturn=False) # optreturn=False returns analyzer results directly

    # --- Configure Cerebro for Optimization ---
    # Define parameter ranges - Adjust as needed, keep combinations reasonable
    logger.info("Defining optimization parameter ranges...")
    cerebro.optstrategy(
        CombinedStrategy,
        short_window=range(10, 21, 5),       # Test 10, 15, 20
        long_window=range(25, 46, 10),      # Test 25, 35, 45
        rsi_period=range(10, 19, 4),         # Test 10, 14, 18
        macd_short=range(10, 15, 4),         # Test 10, 14
        macd_long=range(24, 31, 6),          # Test 24, 30
        macd_signal=range(8, 11, 2)          # Test 8, 10
        # Keep RSI levels fixed or optimize:
        # rsi_overbought=[70, 75],
        # rsi_oversold=[30, 25]
    )
    logger.info("Optimization strategy added.")

    # --- Data Loading ---
    symbol_to_test = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    data_df = fetch_data(symbol_to_test, start_date, end_date)

    if data_df is not None:
        data_feed = bt.feeds.PandasData(dataname=data_df)
        cerebro.adddata(data_feed)
        logger.info("Data feed added.")

        # --- Broker Settings ---
        initial_cash = 100000.0
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addsizer(bt.sizers.FixedSize, stake=10)
        logger.info("Broker and sizer configured.")

        # --- Analyzers ---
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Years)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        logger.info("Analyzers added.")

        # --- Run Optimization ---
        logger.info(f'Starting optimization for {symbol_to_test}... This may take some time.')
        # Use maxcpus=None to utilize more cores if available, but start with 1 for debugging
        opt_results = cerebro.run(maxcpus=1)
        logger.info('Optimization complete.')

        # --- Process and Print Optimization Results ---
        logger.info("--- Processing Optimization Results ---")

        results_list = []
        run_count = 0
        error_count = 0
        for run_result in opt_results:
            run_count += 1
            for strategy_instance in run_result:
                try:
                    params = strategy_instance.params
                    sharpe = strategy_instance.analyzers.sharpe.get_analysis().get('sharperatio', 0)
                    # Handle potential None value from SharpeRatio
                    sharpe = sharpe if sharpe is not None else 0.0
                    drawdown = strategy_instance.analyzers.drawdown.get_analysis().max.drawdown
                    returns_analyzer = strategy_instance.analyzers.returns.get_analysis()
                    total_return = returns_analyzer.get('rtot', 0) * 100
                    final_value = initial_cash * (1 + returns_analyzer.get('rtot', 0))

                    trade_info = strategy_instance.analyzers.tradeanalyzer.get_analysis()
                    total_trades = trade_info.total.total
                    pnl_net = trade_info.pnl.net.total if total_trades > 0 else 0

                    results_list.append({
                        'sw': params.short_window, 'lw': params.long_window, 'rp': params.rsi_period,
                        'ms': params.macd_short, 'ml': params.macd_long, 'msig': params.macd_signal,
                        'final_value': final_value,
                        'return_pct': total_return,
                        'sharpe': sharpe,
                        'max_dd_pct': drawdown,
                        'trades': total_trades,
                        'pnl_net': pnl_net
                    })
                except AttributeError as ae:
                    # Handle cases where analyzers might not have data (e.g., no trades)
                    logger.warning(f"AttributeError processing run results (likely no trades/metrics): {ae}")
                    error_count += 1
                    # Add placeholder or skip? For now, add placeholder with minimal info
                    results_list.append({
                        'sw': params.short_window, 'lw': params.long_window, 'rp': params.rsi_period,
                        'ms': params.macd_short, 'ml': params.macd_long, 'msig': params.macd_signal,
                        'final_value': initial_cash, 'return_pct': 0, 'sharpe': 0,
                        'max_dd_pct': 0, 'trades': 0, 'pnl_net': 0
                    })

                except Exception as e:
                    logger.error(f"Error processing results for one run: {e}")
                    error_count += 1
                    # Optionally log parameters here

        logger.info(f"Processed {run_count} optimization runs. Encountered {error_count} errors during result processing.")

        if not results_list:
             logger.error("No valid results collected from optimization runs.")
        else:
            # Convert to DataFrame and display results
            results_df = pd.DataFrame(results_list)
            # Ensure numeric types for sorting
            results_df[['final_value', 'return_pct', 'sharpe', 'max_dd_pct', 'trades', 'pnl_net']] = results_df[['final_value', 'return_pct', 'sharpe', 'max_dd_pct', 'trades', 'pnl_net']].apply(pd.to_numeric)

            # Sort by final value
            results_df_sorted_value = results_df.sort_values(by='final_value', ascending=False)
            print("\n--- Top 10 Parameter Sets (Sorted by Final Value) ---")
            print(results_df_sorted_value.head(10).to_string()) # Use to_string for better console formatting

            # Sort by Sharpe ratio
            results_df_sorted_sharpe = results_df.sort_values(by='sharpe', ascending=False)
            print("\n--- Top 10 Parameter Sets (Sorted by Sharpe Ratio) ---")
            print(results_df_sorted_sharpe.head(10).to_string())

            # Find overall best Sharpe
            best_sharpe_run = results_df_sorted_sharpe.iloc[0]
            print("\n--- Best Overall Parameter Set (by Sharpe Ratio) ---")
            print(best_sharpe_run)

        # Note: Plotting is not directly available for optimization results this way

    else:
        logger.error("Cannot run optimization due to data loading failure.")
