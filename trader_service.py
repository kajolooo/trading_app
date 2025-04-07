# trader_service.py
import logging
import time
import datetime
import os
from dotenv import load_dotenv

# --- Centralized Logging Setup ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Load .env file ---
load_dotenv(override=True)
logger.info(".env file loading attempted.")

# --- Import Broker and Strategy ---
try:
    from broker import get_api, place_paper_market_order, get_position, APIError
    from strategy import (
        calculate_indicators,
        SHORT_WINDOW, LONG_WINDOW, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, # Import needed params
        ATR_PERIOD, STOP_ATR_MULTIPLIER, TAKE_PROFIT_ATR_MULTIPLIER # Import ATR params
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    exit(1) # Exit if core components are missing

# --- Configuration ---
SYMBOL = "TSLA"
QTY_TO_TRADE = 5 # Example quantity, adjust as needed
LOOP_INTERVAL_SECONDS = 60 # Check every minute
TRADING_START_HOUR = 9
TRADING_START_MINUTE = 30
TRADING_END_HOUR = 16 # Trades won't enter at/after this hour
MARKET_CLOSE_HOUR = 15 # Force exit check before this hour
MARKET_CLOSE_MINUTE = 55 # Force exit check before this minute

# --- State Variables (In-Memory) ---
# These reflect the script's *intended* state. We still verify with the broker.
position_held = False
entry_price = None

def run_trading_loop():
    """Main trading loop."""
    global position_held, entry_price
    logger.info(f"Starting trading loop for {SYMBOL}...")

    trading_start_time = datetime.time(TRADING_START_HOUR, TRADING_START_MINUTE)
    trading_end_time = datetime.time(TRADING_END_HOUR, 0)
    market_exit_time = datetime.time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)

    while True:
        try:
            # --- Get API Client ---
            try:
                api = get_api()
                logger.debug("Alpaca API client obtained.")
            except ValueError as e:
                logger.error(f"Failed to get API client: {e}. Retrying in {LOOP_INTERVAL_SECONDS} seconds.")
                time.sleep(LOOP_INTERVAL_SECONDS)
                continue

            # --- Check Market Hours ---
            clock = api.get_clock()
            current_dt = clock.timestamp.to_pydatetime() # Use broker timestamp
            current_time = current_dt.time()
            is_market_open = clock.is_open
            logger.debug(f"Current Broker Time: {current_dt}, Market Open: {is_market_open}")

            if not is_market_open and not position_held:
                 logger.info("Market closed and no position held. Waiting...")
                 # Sleep longer if market is closed
                 sleep_time = LOOP_INTERVAL_SECONDS * 5
                 time.sleep(sleep_time)
                 continue

            # --- Get Data and Indicators ---
            logger.debug(f"Calculating indicators for {SYMBOL}...")
            data_df = calculate_indicators(SYMBOL, api)

            if data_df is None or data_df.empty:
                logger.warning("Could not get data or calculate indicators. Skipping this cycle.")
                time.sleep(LOOP_INTERVAL_SECONDS)
                continue

            latest_data = data_df.iloc[-1]
            previous_data = data_df.iloc[-2]
            latest_close = latest_data['close']
            latest_atr = latest_data['atr'] if 'atr' in latest_data else 0

            logger.debug(f"Latest Close: {latest_close:.2f}, Latest ATR: {latest_atr:.3f}")

            # --- Verify Position with Broker ---
            current_position_qty = 0
            try:
                position = get_position(SYMBOL, api)
                if position:
                    current_position_qty = float(position.qty)
                    logger.info(f"Verified position with broker: {current_position_qty} shares of {SYMBOL}.")
                    if not position_held: # Correct internal state if mismatched
                         logger.warning("Internal state mismatch: Broker shows position, but script thought flat. Updating state.")
                         position_held = True
                         # Cannot know the entry price if state was lost! Using current close as rough estimate.
                         entry_price = latest_close
                         logger.warning(f"Entry price unknown due to state mismatch, using current close {latest_close} as estimate.")

                else:
                     logger.info(f"Verified position with broker: No position held for {SYMBOL}.")
                     if position_held:
                         logger.warning("Internal state mismatch: Broker shows flat, but script thought position held. Updating state.")
                         position_held = False
                         entry_price = None

            except APIError as e:
                 logger.error(f"API error getting position for {SYMBOL}: {e}. Cannot proceed reliably.")
                 time.sleep(LOOP_INTERVAL_SECONDS)
                 continue
            except Exception as e:
                 logger.error(f"Unexpected error getting position for {SYMBOL}: {e}. Cannot proceed reliably.", exc_info=True)
                 time.sleep(LOOP_INTERVAL_SECONDS)
                 continue


            # --- Decision Logic ---
            exit_triggered = False
            if position_held:
                # --- Exit Logic ---
                exit_reason = "None"

                # 1. EOD Exit Check
                if current_time >= market_exit_time:
                    logger.info(f"EOD exit condition met at {current_time}.")
                    exit_triggered = True
                    exit_reason = "EOD"

                # 2. ATR Stop Loss Check
                elif entry_price is not None and latest_atr > 0:
                    stop_price = entry_price - (STOP_ATR_MULTIPLIER * latest_atr)
                    if latest_close <= stop_price:
                        logger.info(f"ATR Stop Loss triggered. Close={latest_close:.2f} <= Stop={stop_price:.2f} (Entry={entry_price:.2f}, ATR={latest_atr:.3f})")
                        exit_triggered = True
                        exit_reason = "ATR Stop"

                # 3. ATR Take Profit Check
                elif entry_price is not None and latest_atr > 0:
                     take_profit_price = entry_price + (TAKE_PROFIT_ATR_MULTIPLIER * latest_atr)
                     if latest_close >= take_profit_price:
                         logger.info(f"ATR Take Profit triggered. Close={latest_close:.2f} >= Target={take_profit_price:.2f} (Entry={entry_price:.2f}, ATR={latest_atr:.3f})")
                         exit_triggered = True
                         exit_reason = "ATR Profit"

                # 4. Strategy Sell Signal Check (SMA Death Cross + RSI)
                elif 'short_sma' in latest_data and 'long_sma' in latest_data and 'rsi' in latest_data: # Check indicators exist
                     sma_death_cross = latest_data['short_sma'] < latest_data['long_sma'] and previous_data['short_sma'] >= previous_data['long_sma']
                     rsi_not_oversold = latest_data['rsi'] > RSI_OVERSOLD
                     if sma_death_cross and rsi_not_oversold:
                         logger.info("Strategy SELL signal triggered (SMA Death Cross + RSI > Oversold).")
                         exit_triggered = True
                         exit_reason = "Strategy Signal"

                # --- Execute Exit Order ---
                if exit_triggered:
                    logger.info(f"Attempting to place SELL order for {QTY_TO_TRADE} {SYMBOL} due to: {exit_reason}")
                    try:
                        # Ensure we sell the correct quantity, ideally what we hold
                        qty_to_sell = abs(current_position_qty) if current_position_qty != 0 else QTY_TO_TRADE # Sell what broker says we have
                        if qty_to_sell > 0:
                            order = place_paper_market_order(symbol=SYMBOL, qty=qty_to_sell, side='sell', api_client=api)
                            if order:
                                logger.info(f"SELL order placed successfully: {order.id}")
                                position_held = False # Update internal state
                                entry_price = None    # Reset entry price
                            else:
                                logger.error(f"SELL order placement failed (returned None) for {SYMBOL}.")
                        else:
                             logger.warning(f"Exit triggered but broker shows qty {current_position_qty}. Not placing sell order.")
                             position_held = False # Correct state if broker shows flat
                             entry_price = None

                    except APIError as e:
                        logger.error(f"API Error placing SELL order for {SYMBOL}: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error placing SELL order for {SYMBOL}: {e}", exc_info=True)

            else: # Not in position
                # --- Entry Logic ---
                if trading_start_time <= current_time < trading_end_time:
                    # Strategy Buy Signal Check (SMA Golden Cross + RSI)
                    if 'short_sma' in latest_data and 'long_sma' in latest_data and 'rsi' in latest_data: # Check indicators exist
                         sma_golden_cross = latest_data['short_sma'] > latest_data['long_sma'] and previous_data['short_sma'] <= previous_data['long_sma']
                         rsi_not_overbought = latest_data['rsi'] < RSI_OVERBOUGHT
                         if sma_golden_cross and rsi_not_overbought:
                            logger.info(f"Strategy BUY signal triggered (SMA Golden Cross + RSI < Overbought).")
                            logger.info(f"Attempting to place BUY order for {QTY_TO_TRADE} {SYMBOL}")
                            try:
                                order = place_paper_market_order(symbol=SYMBOL, qty=QTY_TO_TRADE, side='buy', api_client=api)
                                if order:
                                    logger.info(f"BUY order placed successfully: {order.id}")
                                    position_held = True # Update internal state
                                    entry_price = latest_close # Approx entry price
                                    logger.info(f"Updated state: Position Held, Estimated Entry Price: {entry_price:.2f}")
                                else:
                                    logger.error(f"BUY order placement failed (returned None) for {SYMBOL}.")
                            except APIError as e:
                                logger.error(f"API Error placing BUY order for {SYMBOL}: {e}")
                            except Exception as e:
                                logger.error(f"Unexpected error placing BUY order for {SYMBOL}: {e}", exc_info=True)
                else:
                     logger.debug(f"Outside trading hours ({current_time}). No entry check.")

            # --- End of Cycle ---
            logger.debug("End of trading cycle.")

        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user (KeyboardInterrupt). Exiting.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main trading loop: {e}", exc_info=True)
            # Avoid rapid-fire loops on persistent errors
            logger.info("Waiting longer due to error...")
            time.sleep(LOOP_INTERVAL_SECONDS * 2)

        # --- Wait for next interval ---
        # Calculate sleep time more precisely if needed, accounting for execution time
        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    logger.info(f"Trader Service starting for {SYMBOL}...")
    run_trading_loop()
    logger.info("Trader Service finished.")