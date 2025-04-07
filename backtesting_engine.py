# backtesting_engine.py
import backtrader as bt
import datetime
import yfinance as yf
import pandas as pd
import logging
import os
import io # For handling plot in memory (alternative)

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__) # Use Flask's logger later if needed

# --- Strategy Definition (Using internal indicators optimized by Cerebro) ---
class CombinedStrategy(bt.Strategy):
    """
    Combines SMA Crossover, RSI, and MACD confirmation for signals.
    Indicators can be toggled on/off via parameters.
    """
    params = (
        # Indicator Toggles
        ('use_sma', True),
        ('use_rsi', True),
        ('use_macd', True),
        # SMA Params
        ('short_window', 12), ('long_window', 26),
        # RSI Params
        ('rsi_period', 14), ('rsi_overbought', 70), ('rsi_oversold', 30),
        # MACD Params
        ('macd_short', 12), ('macd_long', 26), ('macd_signal', 9),
        # NEW Exit Parameters
        ('use_stop_loss', True),
        ('use_take_profit', True),
        # NEW ATR Exit Parameters
        ('atr_period', 14),        # Period for ATR calculation
        ('stop_atr_multiplier', 2.0), # ATR Multiplier for Stop Loss
        ('take_profit_atr_multiplier', 4.0), # ATR Multiplier for Take Profit
        # Logging
        ('printlog', False),
    )

    def __init__(self):
        """Initialize indicators and strategy state based on params."""
        self.dataclose = self.datas[0].close
        self.order = None
        self.entry_price = None # To store price when position is opened

        # Conditionally initialize indicators
        if self.p.use_sma:
            self.sma_short = bt.indicators.SimpleMovingAverage(period=self.p.short_window)
            self.sma_long = bt.indicators.SimpleMovingAverage(period=self.p.long_window)
            self.sma_crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        else:
            self.sma_short = None
            self.sma_long = None
            self.sma_crossover = None

        if self.p.use_rsi:
            self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        else:
            self.rsi = None

        if self.p.use_macd:
            self.macd = bt.indicators.MACD(
                period_me1=self.p.macd_short,
                period_me2=self.p.macd_long,
                period_signal=self.p.macd_signal
            )
            self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        else:
            self.macd = None
            self.macd_crossover = None

        # Initialize ATR indicator (always needed if exits might be used)
        self.atr = bt.indicators.AverageTrueRange(period=self.p.atr_period)

    def log(self, txt, dt=None, doprint=False):
        # Disabled for optimization runs usually
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')
        # pass # No need for pass if there's an if block

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return

        if order.status in [order.Completed]:
            if order.isbuy():
                if self.params.printlog:
                    self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.entry_price = order.executed.price # Record entry price
            elif order.issell():
                if self.params.printlog:
                    self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.entry_price = None # Clear entry price on sell
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
             if self.params.printlog:
                  self.log('Order Canceled/Margin/Rejected')
             self.entry_price = None # Also clear if order fails

        self.order = None # Reset pending order status

    def notify_trade(self, trade):
        # Minimal implementation for optimization
        if not trade.isclosed:
            return
        if self.params.printlog: # Only log if enabled
            self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
        # pass # Not needed as there's logic

    def next(self):
        """Define the core strategy logic checking active indicators and ATR exits."""
        current_close = self.dataclose[0]
        current_atr = self.atr[0]
        if self.order: return

        # Get current time (handle potential errors on daily data)
        try:
            current_dt = self.datas[0].datetime.datetime(0)
            current_time = current_dt.time()
            is_intraday = True
        except AttributeError:
            current_time = None # No time available for daily data
            is_intraday = False
            current_dt = self.datas[0].datetime.date(0) # Fallback to date for logging

        # Define trading hours (adjust as needed, e.g., for different markets/timezones)
        trading_start = datetime.time(9, 30)
        trading_end = datetime.time(16, 0) 
        market_close_time = datetime.time(15, 55)

        # Check if within trading hours
        in_trading_hours = is_intraday and (trading_start <= current_time < trading_end)

        if not self.position:
            # --- Check for BUY signal ONLY during trading hours --- 
            if in_trading_hours:
                sma_golden_cross = self.sma_crossover > 0 if self.p.use_sma else True
                macd_bullish_cross = self.macd_crossover > 0 if self.p.use_macd else True
                rsi_not_overbought = self.rsi < self.p.rsi_overbought if self.p.use_rsi else True
                buy_signal = sma_golden_cross and macd_bullish_cross and rsi_not_overbought
                if not self.p.use_sma and not self.p.use_rsi and not self.p.use_macd: buy_signal = False
                
                if buy_signal:
                     self.order = self.buy()
                     if self.params.printlog: self.log(f'BUY CREATE @ {current_time}, Price: {current_close:.2f}')
            # else: # Optional: Log if outside hours
            #    if self.params.printlog: self.log(f'Outside trading hours ({current_time}), no buy check.')
        else: # We are in the market
            exit_signal = False
            exit_reason = ""

            # Check for End-of-Day exit first (if intraday)
            if is_intraday and current_time >= market_close_time:
                exit_signal = True
                exit_reason = "End-of-Day Exit"
                if self.params.printlog: self.log(f'EOD EXIT TRIGGERED at {current_time}')
            
            # 1. Check ATR Stop Loss (conditionally, only if not EOD exit)
            if not exit_signal and self.p.use_stop_loss and self.entry_price is not None and current_atr > 0:
                 stop_price = self.entry_price - self.p.stop_atr_multiplier * current_atr
                 if current_close <= stop_price:
                     exit_signal = True
                     exit_reason = f"ATR STOP ({self.p.stop_atr_multiplier:.1f}xATR={current_atr:.2f}) Hit"
            
            # 2. Check ATR Take Profit (conditionally & only if not already EOD/stopped)
            if not exit_signal and self.p.use_take_profit and self.entry_price is not None and current_atr > 0:
                profit_target = self.entry_price + self.p.take_profit_atr_multiplier * current_atr
                if current_close >= profit_target:
                    exit_signal = True
                    exit_reason = f"ATR PROFIT ({self.p.take_profit_atr_multiplier:.1f}xATR={current_atr:.2f}) Hit"

            # 3. Check Original Strategy SELL Signal (only if not already exited)
            if not exit_signal:
                sma_death_cross = self.sma_crossover < 0 if self.p.use_sma else True
                macd_bearish_cross = self.macd_crossover < 0 if self.p.use_macd else True
                rsi_not_oversold = self.rsi > self.p.rsi_oversold if self.p.use_rsi else True
                strategy_sell_signal = sma_death_cross and macd_bearish_cross and rsi_not_oversold
                if not self.p.use_sma and not self.p.use_rsi and not self.p.use_macd: strategy_sell_signal = False
                
                if strategy_sell_signal:
                    exit_signal = True
                    exit_reason = "Strategy SELL Signal"

            # --- Execute Exit --- 
            if exit_signal:
                self.order = self.sell()
                if self.params.printlog: self.log(f'SELL/CLOSE CREATE ({exit_reason}), Price: {current_close:.2f}')

    # No stop method needed

# --- Data Fetching ---
def fetch_data(symbol, start_date, end_date, interval='1d'):
    """Fetch historical data using yfinance and clean columns for backtrader."""
    # Map common intervals to yfinance intervals if needed, handle validation
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    if interval not in valid_intervals:
        logger.warning(f"Invalid interval '{interval}' requested. Defaulting to '1d'.")
        interval = '1d'

    # Note yfinance constraints on intraday data periods
    # (e.g., 1m data limited to last 7 days, hourly limited to 730 days)
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date} with interval {interval}")
    try:
        # Pass interval to yf.download
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
        logger.info(f"Raw data columns from yfinance: {df.columns}")
        if df.empty: logger.error(f"No data found for {symbol}..."); return None

        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Flattening MultiIndex columns...")
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).lower().replace(' ', '_') for col in df.columns] # Force lowercase and replace spaces

        expected_cols = {'open', 'high', 'low', 'close', 'adj_close', 'volume'} # Use underscore
        # Rename 'adj close' if needed
        if 'adj close' in df.columns:
            df.rename(columns={'adj close': 'adj_close'}, inplace=True)

        final_cols = [col for col in expected_cols if col in df.columns]
        if 'close' not in final_cols: logger.error("CRITICAL: 'close' column missing..."); return None

        # Add 'datetime' column required by PandasData feed if index is not named 'datetime'
        if df.index.name != 'datetime':
             df['datetime'] = pd.to_datetime(df.index)
             df.set_index('datetime', inplace=True)

        # Ensure required columns exist with expected names for PandasData
        df_final = df.rename(columns={
             'adj_close': 'adjusted_close' # Example: If you want to map it
             # Ensure open, high, low, close, volume exist
        })
        # Select only standard OHLCV columns for simplicity if needed
        std_cols = ['open', 'high', 'low', 'close', 'volume'] # Adjusted close can be added if needed
        final_cols_std = [col for col in std_cols if col in df_final.columns]
        df_final = df_final[final_cols_std]


        logger.info(f"Processed {len(df_final)} rows for backtrader.")
        return df_final
    except Exception as e:
        logger.error(f"Error fetching/processing data: {e}", exc_info=True)
        return None


# --- Single Backtest Run Function ---
def run_single_backtest(symbol, start_date, end_date, strategy_params, interval='1d',
                       initial_cash=100000.0, commission=0.001, stake=10):
    """Runs a single backtest instance with parameters passed via dict."""
    logger.info(f"Running single backtest for {symbol} ({interval}) with params: {strategy_params}")
    cerebro = bt.Cerebro()
    data_df = fetch_data(symbol, start_date, end_date, interval=interval)
    if data_df is None: return None, "Data fetching failed."
    if not isinstance(data_df.index, pd.DatetimeIndex): # Ensure index is datetime
        try: data_df.index = pd.to_datetime(data_df.index)
        except Exception as e: return None, f"Invalid data index: {e}"

    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)

    # Pass the whole dictionary
    cerebro.addstrategy(CombinedStrategy, **strategy_params)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(perc=0.0001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Years)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    try:
        results = cerebro.run()
        strat = results[0]
        final_value = cerebro.broker.getvalue()
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        returns_analysis = strat.analyzers.returns.get_analysis()
        trade_analysis = strat.analyzers.tradeanalyzer.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        winning_trades = trade_analysis.get('won', {}).get('total', 0)
        losing_trades = trade_analysis.get('lost', {}).get('total', 0)
        pnl_net = trade_analysis.get('pnl', {}).get('net', {}).get('total', 0.0)
        total_trades = total_trades if total_trades is not None else 0
        winning_trades = winning_trades if winning_trades is not None else 0
        losing_trades = losing_trades if losing_trades is not None else 0
        pnl_net = pnl_net if pnl_net is not None else 0.0
        sharpe = sharpe_analysis.get('sharperatio', 0) if sharpe_analysis else 0.0
        sharpe = sharpe if sharpe is not None else 0.0
        max_dd = drawdown_analysis.max.drawdown if drawdown_analysis and drawdown_analysis.max else 0.0
        total_return = returns_analysis.get('rtot', 0) * 100 if returns_analysis else 0.0

        analysis_results = {
            'initial_cash': initial_cash, 'final_value': final_value, 'total_return_pct': total_return,
            'sharpe_ratio': sharpe, 'max_drawdown_pct': max_dd, 'total_trades': total_trades,
            'winning_trades': winning_trades, 'losing_trades': losing_trades, 'pnl_net': pnl_net
        }
        logger.info(f"Single run complete. Final Value: {final_value:.2f}")
        return analysis_results, None
    except Exception as e:
        logger.error(f"Error during single backtest run for {symbol}: {e}", exc_info=True)
        return None, f"Backtest execution error: {e}"


# --- Optimization Run Function ---
def run_optimization(symbol, start_date, end_date, opt_ranges, opt_params_fixed, interval='1d',
                    initial_cash=100000.0, commission=0.001, stake=10, maxcpus=1):
    """Runs strategy optimization, considering active indicators/exits."""
    logger.info(f"Running optimization for {symbol} ({interval}) using {maxcpus} CPUs. Ranges: {opt_ranges}, Fixed: {opt_params_fixed}")
    cerebro = bt.Cerebro(optreturn=False)

    # Combine optimizable ranges and fixed boolean toggles
    opt_params_combined = {**opt_ranges, **opt_params_fixed}
    logger.info(f"Combined optimization parameters: {opt_params_combined}")
    cerebro.optstrategy(CombinedStrategy, **opt_params_combined)

    data_df = fetch_data(symbol, start_date, end_date, interval=interval)
    if data_df is None: return None, "Data fetching failed."
    if not isinstance(data_df.index, pd.DatetimeIndex): # Ensure index is datetime
        try: data_df.index = pd.to_datetime(data_df.index)
        except Exception as e: return None, f"Invalid data index: {e}"
    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(perc=0.0001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Years)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    try:
        logger.info("Starting Cerebro optimization run...")
        opt_results = cerebro.run(maxcpus=maxcpus)
        logger.info("Cerebro optimization run finished.")

        results_list = []
        error_count = 0
        for run_result in opt_results:
            for strategy_instance in run_result:
                 try:
                    params_obj = strategy_instance.params
                    # Extract metrics safely...
                    sharpe_analysis = strategy_instance.analyzers.sharpe.get_analysis()
                    sharpe = sharpe_analysis.get('sharperatio', 0) if sharpe_analysis else 0.0
                    sharpe = sharpe if sharpe is not None else 0.0
                    drawdown_analysis = strategy_instance.analyzers.drawdown.get_analysis()
                    drawdown = drawdown_analysis.max.drawdown if drawdown_analysis and drawdown_analysis.max else 0.0
                    returns_analysis = strategy_instance.analyzers.returns.get_analysis()
                    returns = returns_analysis.get('rtot', 0) * 100 if returns_analysis else 0.0
                    final_val = initial_cash * (1 + returns/100.0)
                    trade_analysis = strategy_instance.analyzers.tradeanalyzer.get_analysis()
                    trades = trade_analysis.get('total', {}).get('total', 0) or 0
                    pnl = trade_analysis.get('pnl', {}).get('net', {}).get('total', 0.0) or 0.0
                    trades = trades if trades is not None else 0
                    pnl = pnl if pnl is not None else 0.0

                    # Store results including toggle status
                    result_entry = {
                        # Toggles
                        'use_sma': params_obj.use_sma, 'use_rsi': params_obj.use_rsi, 'use_macd': params_obj.use_macd,
                        'use_stop_loss': params_obj.use_stop_loss, 'use_take_profit': params_obj.use_take_profit,
                        # Performance Metrics
                        'final_value': final_val, 'return_pct': returns, 'sharpe': sharpe,
                        'max_dd_pct': drawdown, 'trades': trades, 'pnl_net': pnl,
                        # Base ATR Param
                        'atr_p': params_obj.atr_period 
                    }
                    # Add indicator params only if they were used
                    if params_obj.use_sma: result_entry.update({'sw': params_obj.short_window, 'lw': params_obj.long_window})
                    if params_obj.use_rsi: result_entry.update({'rp': params_obj.rsi_period, 'rsi_ob': params_obj.rsi_overbought, 'rsi_os': params_obj.rsi_oversold})
                    if params_obj.use_macd: result_entry.update({'ms': params_obj.macd_short, 'ml': params_obj.macd_long, 'msig': params_obj.macd_signal})
                    # Conditionally add ATR multiplier results, using N/A if not used
                    result_entry['sl_atr'] = params_obj.stop_atr_multiplier if params_obj.use_stop_loss else 'N/A'
                    result_entry['tp_atr'] = params_obj.take_profit_atr_multiplier if params_obj.use_take_profit else 'N/A'
                    
                    results_list.append(result_entry)

                 except Exception as e:
                     params_dict = strategy_instance.params.__dict__ if hasattr(strategy_instance, 'params') else {}
                     logger.error(f"Error processing results for one optimization run (Params: {params_dict}): {e}", exc_info=True)
                     error_count += 1

        if not results_list: return None, "No valid results collected."
        results_df = pd.DataFrame(results_list)
        logger.info(f"Optimization processing complete. Found {len(results_df)} valid results. Encountered {error_count} processing errors.")
        return results_df, None
    except Exception as e:
        logger.error(f"Error during optimization run for {symbol}: {e}", exc_info=True)
        return None, f"Optimization execution error: {e}"


# --- Plot Generation Function ---
def generate_plot_for_params(symbol, start_date, end_date, strategy_params, plot_filename, interval='1d'):
    """Runs a backtest with specific params and saves a plot."""
    logger.info(f"Generating plot for {symbol} ({interval}) with params: {strategy_params}")
    cerebro = bt.Cerebro(stdstats=False) # Disable standard stats printing

    data_df = fetch_data(symbol, start_date, end_date, interval=interval)
    if data_df is None:
        logger.error("Data fetching failed for plotting.")
        return False
    if not isinstance(data_df.index, pd.DatetimeIndex):
        try: data_df.index = pd.to_datetime(data_df.index)
        except Exception as e:
             logger.error(f"Invalid data index for plotting: {e}")
             return False

    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)

    # Add strategy with exact parameters
    cerebro.addstrategy(CombinedStrategy, **strategy_params)

    try:
        # Run quietly
        cerebro.run()

        # Ensure the plot directory exists
        plot_dir = os.path.dirname(plot_filename)
        os.makedirs(plot_dir, exist_ok=True)

        # Generate and save the plot
        fig = cerebro.plot(
            style='candlestick',
            barup='green',
            bardown='red',
            volup='#2ca02c', # Green volume bars
            voldown='#d62728', # Red volume bars
            savefig=True,
            figfilename=plot_filename,
            figscale=1.5 # Increase overall scale
        )
        if fig:
             plt.close(fig[0][0]) 

        logger.info(f"Plot saved successfully to {plot_filename}")
        return True
    except Exception as e:
        logger.error(f"Error generating plot for {symbol}: {e}", exc_info=True)
        try:
            if 'fig' in locals() and fig:
                 plt.close(fig[0][0])
        except Exception:
            pass 
        return False
