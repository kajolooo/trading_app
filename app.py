# app.py
# FORCE MATPLOTLIB BACKEND EARLY
import matplotlib
matplotlib.use('Agg')

import logging
import os # For plot directory
import time # For unique filenames
from flask import Flask, render_template, request, redirect, url_for, send_from_directory # Added send_from_directory
from backtesting_engine import (
    run_single_backtest, 
    run_optimization, 
    generate_plot_for_params, 
    CombinedStrategy # <--- IMPORT THE CLASS
)
import pandas as pd
import numpy as np # For linspace

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.logger.setLevel(logging.INFO) # Use Flask's logger

# Define plot directory relative to app.py location
PLOT_FOLDER = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_FOLDER, exist_ok=True)
app.config['PLOT_FOLDER'] = PLOT_FOLDER

# --- Helper Function for Optimization Ranges --- 
def generate_range(min_val, max_val, steps, is_pct=False, is_float=False):
    """Generates a list of values for optimization from min, max, steps."""
    if steps <= 0:
        steps = 1 # Ensure at least one step
    if steps == 1:
        val = float(min_val) if is_float or is_pct else int(min_val)
        return [val / 100.0] if is_pct else [val]
    
    # Use linspace for even distribution, including endpoints
    values = np.linspace(float(min_val), float(max_val), int(steps))
    
    if is_pct:
        # Convert percentages to decimals
        return [round(v / 100.0, 4) for v in values] # Round decimals
    elif is_float:
        return [round(v, 4) for v in values]
    else:
        # Ensure integers for periods, windows etc.
        # Round and convert to int, remove duplicates if rounding causes them
        return sorted(list(set(int(round(v)) for v in values)))

@app.route('/', methods=['GET'])
def index():
    """Renders the main input form with default values."""
    app.logger.info("Rendering index page.")
    # Provide default values for the initial form load
    default_form_data = {
        'symbols_str': 'AAPL',
        'start_date': '2022-01-01',
        'end_date': '2023-12-31',
        'timeframe': '1d', # Default timeframe
        'initial_cash': '100000', # Add default cash
        'optimize': False,
        'opt_steps': '4',
        'max_cpus': '1', # Default max CPUs
        'generate_plots': True, # Default plot generation to True
        'use_sma': True,
        'sma_short_min': '10', 'sma_short_max': '15',
        'sma_long_min': '25', 'sma_long_max': '35',
        'use_rsi': True,
        'rsi_period_min': '12', 'rsi_period_max': '16',
        'rsi_ob_min': '65', 'rsi_ob_max': '75',
        'rsi_os_min': '25', 'rsi_os_max': '35',
        'use_macd': True,
        'macd_short_min': '10', 'macd_short_max': '14',
        'macd_long_min': '24', 'macd_long_max': '30',
        'macd_signal_min': '8', 'macd_signal_max': '10',
        'use_stop_loss': True,
        'use_take_profit': True,
        'atr_period_min': '10', 'atr_period_max': '20',
        'stop_atr_multiplier_min': '1.5', 'stop_atr_multiplier_max': '3.0',
        'take_profit_atr_multiplier_min': '3.0', 'take_profit_atr_multiplier_max': '6.0',
    }
    return render_template('index.html', form_data=default_form_data)

@app.route('/run_backtest', methods=['POST'])
def run_backtest_route():
    """Handles form submission and runs backtest/optimization."""
    app.logger.info("Received POST request to /run_backtest")
    error = None
    single_results = None
    opt_results_html = None
    symbol = None
    plot_info = {}
    run_summary = None # Initialize run summary

    try:
        # --- Parse General Inputs --- 
        symbols_str = request.form.get('symbols', 'AAPL')
        symbol = symbols_str.split(',')[0].strip().upper()
        if not symbol: raise ValueError("Symbol cannot be empty.")
        start_date = request.form.get('start_date', '2022-01-01')
        end_date = request.form.get('end_date', '2023-12-31')
        timeframe = request.form.get('timeframe', '1d') # Parse timeframe
        initial_cash = float(request.form.get('initial_cash', '100000')) # Parse cash
        optimize = request.form.get('optimize') == 'yes'
        generate_plots = request.form.get('generate_plots') == 'yes' # Parse plot toggle
        opt_steps = int(request.form.get('opt_steps', '4'))
        max_cpus = int(request.form.get('max_cpus', '1')) # Parse max_cpus
        # Ensure at least 1 CPU
        if max_cpus < 1:
            max_cpus = 1 
        use_sma = request.form.get('use_sma') == 'yes'
        use_rsi = request.form.get('use_rsi') == 'yes'
        use_macd = request.form.get('use_macd') == 'yes'
        use_stop_loss = request.form.get('use_stop_loss') == 'yes'
        use_take_profit = request.form.get('use_take_profit') == 'yes'

        app.logger.info(f"Symbol: {symbol}, Start: {start_date}, End: {end_date}, Timeframe: {timeframe}, Cash: {initial_cash}, Optimize: {optimize}, OptSteps: {opt_steps}, MaxCPUs: {max_cpus}, GeneratePlots: {generate_plots}")
        app.logger.info(f"Indicators Active: SMA={use_sma}, RSI={use_rsi}, MACD={use_macd}")
        app.logger.info(f"Exits Active: StopLoss={use_stop_loss}, TakeProfit={use_take_profit}")

        # --- Parse Parameter Min/Max --- 
        params_in = {}
        param_keys = [
            'sma_short', 'sma_long', 'rsi_period', 'rsi_ob', 'rsi_os', 
            'macd_short', 'macd_long', 'macd_signal', 
            'atr_period', 'stop_atr_multiplier', 'take_profit_atr_multiplier'
        ]
        for key in param_keys:
            is_float = 'multiplier' in key
            params_in[key] = {
                'min_val': request.form.get(f'{key}_min', '0'),
                'max_val': request.form.get(f'{key}_max', '0'),
                'is_pct': False,
                'is_float': is_float
            }

        # We need the form_data dict created BEFORE the run for saving later
        # --- Prepare form data (used for saving and repopulation) --- 
        current_form_data = {
            'symbols_str': request.form.get('symbols', 'AAPL'),
            'start_date': request.form.get('start_date', '2022-01-01'),
            'end_date': request.form.get('end_date', '2023-12-31'),
            'timeframe': request.form.get('timeframe', '1d'),
            'initial_cash': request.form.get('initial_cash', '100000'),
            'optimize': request.form.get('optimize') == 'yes',
            'generate_plots': request.form.get('generate_plots') == 'yes',
            'opt_steps': request.form.get('opt_steps', '4'),
            'max_cpus': request.form.get('max_cpus', '1'),
            'use_sma': request.form.get('use_sma') == 'yes',
            'use_rsi': request.form.get('use_rsi') == 'yes',
            'use_macd': request.form.get('use_macd') == 'yes',
            'use_stop_loss': request.form.get('use_stop_loss') == 'yes',
            'use_take_profit': request.form.get('use_take_profit') == 'yes',
        }
        for key in param_keys:
            current_form_data[f'{key}_min'] = request.form.get(f'{key}_min', '0')
            current_form_data[f'{key}_max'] = request.form.get(f'{key}_max', '0')

        if optimize:
            # --- Run Optimization --- 
            app.logger.info("Starting optimization run...")
            opt_ranges = {}
            # Build ranges dynamically using the single opt_steps value
            if use_sma:
                opt_ranges['short_window'] = generate_range(**params_in['sma_short'], steps=opt_steps)
                opt_ranges['long_window'] = generate_range(**params_in['sma_long'], steps=opt_steps)
            if use_rsi:
                opt_ranges['rsi_period'] = generate_range(**params_in['rsi_period'], steps=opt_steps)
                opt_ranges['rsi_overbought'] = generate_range(**params_in['rsi_ob'], steps=opt_steps)
                opt_ranges['rsi_oversold'] = generate_range(**params_in['rsi_os'], steps=opt_steps)
            if use_macd:
                opt_ranges['macd_short'] = generate_range(**params_in['macd_short'], steps=opt_steps)
                opt_ranges['macd_long'] = generate_range(**params_in['macd_long'], steps=opt_steps)
                opt_ranges['macd_signal'] = generate_range(**params_in['macd_signal'], steps=opt_steps)
            
            # Build ATR ranges (always added? or based on exit toggles? Let's add always for now)
            opt_ranges['atr_period'] = generate_range(**params_in['atr_period'], steps=opt_steps)
            if use_stop_loss:
                opt_ranges['stop_atr_multiplier'] = generate_range(**params_in['stop_atr_multiplier'], steps=opt_steps)
            if use_take_profit:
                opt_ranges['take_profit_atr_multiplier'] = generate_range(**params_in['take_profit_atr_multiplier'], steps=opt_steps)

            # Add fixed boolean toggles (including exits)
            opt_params_fixed = {
                'use_sma': [use_sma],
                'use_rsi': [use_rsi],
                'use_macd': [use_macd],
                'use_stop_loss': [use_stop_loss],
                'use_take_profit': [use_take_profit]
            }

            app.logger.info(f"Generated Opt Ranges: {opt_ranges}")
            app.logger.info(f"Fixed Opt Params: {opt_params_fixed}")
            
            results_df, error = run_optimization(symbol, start_date, end_date, 
                                               opt_ranges, opt_params_fixed, 
                                               interval=timeframe, 
                                               initial_cash=initial_cash,
                                               maxcpus=max_cpus) # Pass max_cpus
            if error:
                 app.logger.error(f"Optimization failed: {error}")
            elif results_df is not None and not results_df.empty:
                 app.logger.info("Optimization successful, preparing results table and plots.")
                 # Sort results
                 df_sorted_val = results_df.sort_values(by='final_value', ascending=False)
                 df_sorted_sharpe = results_df.sort_values(by='sharpe', ascending=False)
                 # Get top 5 for display
                 df_display_val = df_sorted_val.head(5)
                 df_display_sharpe = df_sorted_sharpe.head(5)
                 opt_results_html = "<h4>Top 5 by Final Value:</h4>" + df_display_val.to_html(classes='table table-striped', index=False, float_format='%.2f')
                 opt_results_html += "<br><h4>Top 5 by Sharpe Ratio:</h4>" + df_display_sharpe.to_html(classes='table table-striped', index=False, float_format='%.3f')

                 # --- Generate Plots ONLY if Optimize and Generate Plots are checked --- 
                 if generate_plots:
                     app.logger.info("Plot generation enabled.")
                     try:
                         # Get full parameter set for top by value
                         top_val_params = df_sorted_val.iloc[0].to_dict()
                         # Get full parameter set for top by sharpe
                         top_sharpe_params = df_sorted_sharpe.iloc[0].to_dict()
        
                         # Manually define the expected strategy parameter keys
                         valid_strategy_param_keys = [
                             'use_sma', 'use_rsi', 'use_macd', 'use_stop_loss', 'use_take_profit',
                             'short_window', 'long_window', 'rsi_period', 'rsi_overbought', 'rsi_oversold',
                             'macd_short', 'macd_long', 'macd_signal', 'atr_period',
                             'stop_atr_multiplier', 'take_profit_atr_multiplier', 'printlog'
                         ]
        
                         top_val_plot_params = {k: v for k, v in top_val_params.items() if k in valid_strategy_param_keys}
                         top_sharpe_plot_params = {k: v for k, v in top_sharpe_params.items() if k in valid_strategy_param_keys}
        
                         # Create unique filenames
                         timestamp = int(time.time())
                         plot_filename_val = f"plot_val_{symbol}_{timestamp}.png"
                         plot_filename_sharpe = f"plot_sharpe_{symbol}_{timestamp}.png"
                         full_plot_path_val = os.path.join(app.config['PLOT_FOLDER'], plot_filename_val)
                         full_plot_path_sharpe = os.path.join(app.config['PLOT_FOLDER'], plot_filename_sharpe)
                         
                         # Generate plots
                         plot_success_val = generate_plot_for_params(symbol, start_date, end_date, top_val_plot_params, full_plot_path_val, interval=timeframe)
                         plot_success_sharpe = generate_plot_for_params(symbol, start_date, end_date, top_sharpe_plot_params, full_plot_path_sharpe, interval=timeframe)
        
                         # Store relative paths for the template
                         if plot_success_val: plot_info['top_value_plot'] = plot_filename_val
                         if plot_success_sharpe: plot_info['top_sharpe_plot'] = plot_filename_sharpe
                     except Exception as plot_e:
                         app.logger.error(f"Error during plot generation: {plot_e}", exc_info=True)
                         # Optionally inform user about plot failure
                         error = (error + " | Plot generation failed.") if error else "Plot generation failed."
                 else:
                     app.logger.info("Plot generation disabled.")
                 
                 # Prepare summary for history (use top result by value)
                 top_val_result = df_sorted_val.iloc[0].to_dict()
                 run_summary = {
                     'type': 'Optimization',
                     'timestamp': int(time.time() * 1000), # JS uses milliseconds
                     'settings': current_form_data, # Save the settings used for this run
                     'results': { # Store key results 
                         'return_pct': top_val_result.get('return_pct'),
                         'sharpe': top_val_result.get('sharpe'),
                         'trades': top_val_result.get('trades')
                     }
                 }
            else:
                 error = "Optimization ran but produced no valid results."

        else:
            # --- Run Single Backtest --- 
            app.logger.info("Starting single backtest run...")
            # Rebuild strategy_params from current_form_data min values
            strategy_params = {}
            if current_form_data['use_sma']:
                strategy_params['short_window'] = int(current_form_data['sma_short_min'])
                strategy_params['long_window'] = int(current_form_data['sma_long_min'])
            if current_form_data['use_rsi']:
                strategy_params['rsi_period'] = int(current_form_data['rsi_period_min'])
                strategy_params['rsi_overbought'] = int(current_form_data['rsi_ob_min'])
                strategy_params['rsi_oversold'] = int(current_form_data['rsi_os_min'])
            if current_form_data['use_macd']:
                strategy_params['macd_short'] = int(current_form_data['macd_short_min'])
                strategy_params['macd_long'] = int(current_form_data['macd_long_min'])
                strategy_params['macd_signal'] = int(current_form_data['macd_signal_min'])
            strategy_params['atr_period'] = int(current_form_data['atr_period_min'])
            if current_form_data['use_stop_loss']:
                strategy_params['stop_atr_multiplier'] = float(current_form_data['stop_atr_multiplier_min'])
            if current_form_data['use_take_profit']:
                strategy_params['take_profit_atr_multiplier'] = float(current_form_data['take_profit_atr_multiplier_min'])
            # Add boolean toggles
            strategy_params['use_sma'] = current_form_data['use_sma']
            strategy_params['use_rsi'] = current_form_data['use_rsi']
            strategy_params['use_macd'] = current_form_data['use_macd']
            strategy_params['use_stop_loss'] = current_form_data['use_stop_loss']
            strategy_params['use_take_profit'] = current_form_data['use_take_profit']
                
            single_results, error = run_single_backtest(symbol, start_date, end_date, strategy_params, 
                                                        interval=timeframe,
                                                        initial_cash=initial_cash) # Pass cash
            if not error and single_results:
                 # Prepare summary for history
                run_summary = {
                    'type': 'Single Run',
                    'timestamp': int(time.time() * 1000),
                    'settings': current_form_data, # Save the settings used
                    'results': { # Store key results from single_results dict
                        'total_return_pct': single_results.get('total_return_pct'),
                        'sharpe_ratio': single_results.get('sharpe_ratio'),
                        'total_trades': single_results.get('total_trades')
                    }
                 }
            # Ensure correct indentation for following code if any was removed
            elif error:
                 app.logger.error(f"Single backtest failed: {error}")
            else:
                 error = "Backtest ran but produced no results."

    except ValueError as ve:
         error = f"Invalid input: {ve}"
         app.logger.error(error)
    except Exception as e:
        error = f"An unexpected error occurred: {e}"
        app.logger.error(error, exc_info=True)

    # --- Prepare form data to pass back FOR REPOPULATION (Can be same as current_form_data) --- 
    form_data_repop = current_form_data 

    # --- Render Template with Results --- 
    return render_template('index.html',
                           symbol=symbol,
                           single_results=single_results,
                           opt_results_html=opt_results_html,
                           error=error,
                           form_data=form_data_repop, # Use the correct dict
                           plot_info=plot_info,
                           run_summary=run_summary # Pass summary for JS to save
                           )

# --- Route to Serve Plots --- 
@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serves generated plot files."""
    app.logger.info(f"Serving plot: {filename}")
    try:
        return send_from_directory(app.config['PLOT_FOLDER'], filename)
    except FileNotFoundError:
        app.logger.error(f"Plot file not found: {filename}")
        return "Plot not found", 404

if __name__ == '__main__':
    app.run(debug=True)