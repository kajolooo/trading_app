# test_get_bars.py
import os
import logging
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError, TimeFrame
import pandas as pd # Need pandas to check the result type

# --- Logging Setup ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv(override=True)
logger.info(".env file loaded (if exists) and override is enabled.")

# --- Function to Get API Client (Copied essentials from broker.py) ---
def get_test_api():
    """Initializes and returns an Alpaca REST API client instance for testing."""
    logger.debug("Attempting to get Alpaca API client for test...")
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    # Ensure it uses the paper trading URL
    raw_base_url_from_env = os.getenv("APCA_API_BASE_URL") # Get raw value first
    logger.info(f"Value of APCA_API_BASE_URL from os.getenv: {raw_base_url_from_env}") # Print raw value
    base_url = raw_base_url_from_env or "https://paper-api.alpaca.markets" # Apply default if None
    logger.info(f"Base URL to be used for client init: {base_url}") # Print value before init

    if not api_key or not api_secret:
        logger.error("Alpaca API Key ID or Secret Key not found in environment variables.")
        raise ValueError("Alpaca API Key ID and Secret Key must be set.")

    try:
        # Use v2 for the latest API version
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        # Print the ACTUAL base_url the client object is configured with
        # logger.info(f"Client object initialized. api.base_url is now: {api.base_url}") # Removed this line - causes AttributeError
        
        # Optionally test connection with get_account to be sure
        try:
             account = api.get_account()
             logger.info(f"Connection test successful. Account Status: {account.status}")
        except APIError as e:
             logger.error(f"Connection test failed (get_account): {e}")
             raise ValueError(f"Failed to connect with API keys: {e}")

        return api
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca API client: {e}", exc_info=True)
        raise ValueError(f"Failed to initialize Alpaca API client: {e}")

# --- Test Parameters ---
test_symbol = 'AAPL'  # The symbol you were having trouble with
# Use the same limit as in strategy.py
# Increased limit slightly to be safe, but 200 should be fine
test_limit = 250 # DATA_LIMIT was 200 in strategy.py

# --- Main Test Execution ---
if __name__ == "__main__":
    logger.info(f"--- Starting get_bars test for {test_symbol} ---")
    api_client = None
    try:
        api_client = get_test_api()
    except ValueError as e:
        logger.error(f"Could not get API client: {e}")
        # Exit if we can't get the client
        exit()

    if api_client:
        try:
            # Try changing TimeFrame.Day to TimeFrame.Hour
            target_timeframe = TimeFrame.Hour 
            logger.info(f"Attempting api.get_bars('{test_symbol}', {target_timeframe}, limit={test_limit}).df")

            # THE ACTUAL CALL WE ARE TESTING
            bars = api_client.get_bars(test_symbol, target_timeframe, limit=test_limit).df

            logger.info("--- RESULTS ---")
            print(f"Type of result: {type(bars)}") # Should be <class 'pandas.core.frame.DataFrame'>
            if isinstance(bars, pd.DataFrame):
                print(f"Is DataFrame empty? {bars.empty}") # <<< THIS IS THE KEY CHECK
                print(f"Number of rows: {len(bars)}")
                if not bars.empty:
                    print("First 5 rows:")
                    print(bars.head())
                    print("\nLast 5 rows:")
                    print(bars.tail())
                else:
                     print("DataFrame is empty, cannot show head() or tail().")
            else:
                 print(f"Result is NOT a Pandas DataFrame.")


        except APIError as e:
            logger.error(f"Alpaca API Error during get_bars: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected Error during get_bars: {e}", exc_info=True)

    logger.info(f"--- Finished get_bars test for {test_symbol} ---")
