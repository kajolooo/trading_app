# broker.py
import os
import logging
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

# Get logger instance for this module
logger = logging.getLogger(__name__)

# Load environment variables from .env file (REMOVED - Load in main.py now)
# load_dotenv(override=True) 
# logger.info("broker.py: .env file loaded (if exists) and override is enabled.") 

def get_api():
    """Initializes and returns an Alpaca REST API client instance.

    Reads API credentials and base URL from environment variables:
    - APCA_API_KEY_ID
    - APCA_API_SECRET_KEY
    - APCA_API_BASE_URL (defaults to Alpaca paper trading URL if not set)

    Raises:
        ValueError: If required API keys (ID and Secret) are not found in environment variables.

    Returns:
        tradeapi.REST: An initialized Alpaca REST API client.
    """
    logger.debug("Attempting to get Alpaca API client...")
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not api_secret:
        logger.error("Alpaca API Key ID or Secret Key not found in environment variables.")
        raise ValueError("Alpaca API Key ID and Secret Key must be set in environment variables.")

    try:
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        logger.info(f"Successfully initialized Alpaca API client for base URL: {base_url}")
        return api
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca API client: {e}", exc_info=True)
        # Re-raise the exception or handle it as appropriate
        # For now, re-raising to make the failure explicit upstream
        raise ValueError(f"Failed to initialize Alpaca API client: {e}")

def get_account_info(api):
    """Fetches relevant account information from Alpaca.

    Args:
        api: An initialized Alpaca REST API client instance.

    Returns:
        dict: A dictionary containing account details like buying power, cash, 
              and portfolio value, or None if an error occurs.
    """
    logger.debug("Attempting to fetch Alpaca account information...")
    try:
        account = api.get_account()
        account_info = {
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "currency": account.currency,
            "status": account.status,
            "account_number": account.account_number
            # Add other relevant fields as needed
        }
        logger.info(f"Successfully fetched account info for {account.account_number}: Status={account.status}, Equity={account_info['equity']}")
        return account_info
    except APIError as e:
        logger.error(f"Alpaca API error fetching account info: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching account info: {e}", exc_info=True)
        return None

def get_all_positions(api):
    """Fetches all open positions from Alpaca.

    Args:
        api: An initialized Alpaca REST API client instance.

    Returns:
        dict: A dictionary mapping symbols to quantities (as floats), 
              or None if an error occurs.
              Returns an empty dict if there are no positions.
    """
    logger.debug("Attempting to fetch all Alpaca positions...")
    positions_dict = {}
    try:
        positions = api.list_positions()
        for position in positions:
            try:
                 # Store quantity as float for consistency
                 positions_dict[position.symbol] = float(position.qty)
            except Exception as e:
                 logger.warning(f"Could not process position for {position.symbol}: {e}")
                 positions_dict[position.symbol] = 'Error' # Mark problematic position
                 
        logger.info(f"Successfully fetched {len(positions_dict)} positions.")
        return positions_dict
    except APIError as e:
        logger.error(f"Alpaca API error fetching positions: {e}", exc_info=True)
        return None # Indicate failure to fetch
    except Exception as e:
        logger.error(f"Unexpected error fetching positions: {e}", exc_info=True)
        return None # Indicate failure to fetch

def get_position(symbol: str, api):
    """Fetches the open position for a specific symbol from Alpaca.

    Args:
        symbol: The stock ticker symbol to check.
        api: An initialized Alpaca REST API client instance.

    Returns:
        Position object if a position exists for the symbol,
        None if no position exists or an error occurs.
        Raises APIError specifically if Alpaca returns an error.
    """
    logger.debug(f"Attempting to fetch position for symbol: {symbol}...")
    try:
        position = api.get_position(symbol)
        logger.info(f"Successfully fetched position for {symbol}: Qty={position.qty}, Side={position.side}")
        return position
    except APIError as e:
        # Alpaca API specifically returns 404 Not Found if position doesn't exist
        if e.status_code == 404:
            logger.info(f"No position found for symbol: {symbol}")
            return None
        else:
            # Re-raise other API errors
            logger.error(f"Alpaca API error fetching position for {symbol}: {e}", exc_info=True)
            raise e # Re-raise the APIError to be handled by the caller
    except Exception as e:
        logger.error(f"Unexpected error fetching position for {symbol}: {e}", exc_info=True)
        # For unexpected errors, we might return None or raise depending on desired caller behavior
        # Returning None for now to prevent crashing the loop, but caller should be aware.
        return None

def place_paper_market_order(symbol: str, qty: int, side: str, api_client=None):
    """Places a paper trading market order on Alpaca.

    Args:
        symbol: The stock ticker symbol.
        qty: The number of shares to trade. Must be positive.
        side: The order side ('buy' or 'sell').
        api_client: Optional. An existing initialized Alpaca REST API client.
                    If None, a new client will be obtained.

    Returns:
        Order object if successful, None otherwise.
    """
    if qty <= 0:
        logger.error(f"Order quantity must be positive. Got: {qty}")
        return None
    if side not in ['buy', 'sell']:
        logger.error(f"Invalid order side: {side}. Must be 'buy' or 'sell'.")
        return None

    logger.info(f"Attempting to place paper market order: {side} {abs(qty)} {symbol}")
    api = api_client # Use provided client if available
    try:
        if api is None:
            logger.debug("API client not provided, obtaining new one...")
            api = get_api() # Get new client if not provided
            logger.debug(f"Got API client for order: {side} {abs(qty)} {symbol}")

        # Ensure quantity is positive for the API call
        order_qty = abs(qty)

        order = api.submit_order(
            symbol=symbol,
            qty=order_qty, # Use positive quantity
            side=side,
            type='market',
            time_in_force='day' # Good Till Canceled ('gtc') or Day ('day') are common
        )
        # Log the actual order details returned by Alpaca
        logger.info(f"Paper market order submitted successfully via API: ID={order.id}, Symbol={order.symbol}, Qty={order.qty}, Side={order.side}, Status={order.status}")
        return order
    except APIError as e:
        logger.error(f"Alpaca API error submitting order for {symbol}: {e}", exc_info=True)
        return None
    except ValueError as e:
        # Error likely came from get_api() failing
        logger.error(f"Failed to place order due to API client initialization error: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing order for {symbol}: {e}", exc_info=True)
        return None 