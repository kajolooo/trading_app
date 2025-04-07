# main.py
import logging
import os
from dotenv import load_dotenv

# --- Centralized Logging Setup ---
# Load environment variables first to potentially configure logging based on env
# load_dotenv() # Removed old load

# Configure logging (do this before other imports that might use logging)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__) # Get logger instance for this module

# --- Load .env file (Override existing env vars) ---
load_dotenv(override=True)
logger.info(".env file loading attempted (override=True).")

# --- Other Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import Literal
from broker import get_api, place_paper_market_order, get_account_info, get_all_positions
from alpaca_trade_api.rest import APIError
from strategy import generate_signal

# --- Pydantic Models ---
class TradeRequest(BaseModel):
    symbol: str = Field(..., description="The stock ticker symbol to trade.")
    # Use Literal to restrict signal values and provide validation
    signal: Literal['BUY', 'SELL'] = Field(..., description="The trading signal ('BUY' or 'SELL').")
    qty: int = Field(..., gt=0, description="The quantity of shares to trade (must be positive).")

# Create the FastAPI app instance
logger.info("Creating FastAPI app instance...")
app = FastAPI()
logger.info("FastAPI app instance created.")

# --- API Endpoints ---

@app.get("/api/account")
async def account_details():
    """Fetches and returns Alpaca account details.

    Returns:
        JSON response with account details (buying power, cash, equity, etc.)

    Raises:
        HTTPException: 500 if API client cannot be initialized or account info cannot be fetched.
    """
    logger.info("Received request for account details.")
    try:
        api = get_api() # Handles missing keys error via ValueError
        account_info = get_account_info(api)
        if account_info:
            logger.info(f"Successfully returned account details for {account_info.get('account_number', 'N/A')}")
            return account_info
        else:
            logger.error("Failed to get account info (get_account_info returned None).")
            raise HTTPException(status_code=500, detail="Failed to fetch account information from broker.")
    except ValueError as e:
        # Error from get_api (missing keys)
        logger.error(f"API client initialization failed for account details request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Broker API client initialization failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching account details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred while fetching account details.")

# API endpoint to get the latest price for a symbol
@app.get("/api/price/{symbol}")
async def get_price(symbol: str):
    logger.info(f"Received request for price of {symbol}")
    """Fetches the latest ask price for a given stock symbol from Alpaca.
    
    Args:
        symbol: The stock ticker symbol.
        
    Returns:
        JSON response with symbol and latest ask price.
        
    Raises:
        HTTPException: 404 if the symbol is not found or API error occurs.
        HTTPException: 500 for other server errors.
    """
    try:
        api = get_api()
        logger.debug(f"Got Alpaca API client for price fetch: {symbol}")
        quote = api.get_latest_quote(symbol)
        logger.info(f"Successfully fetched price for {symbol}: {quote.ap}")
        return {"symbol": symbol, "price": quote.ap}
    except APIError as e:
        logger.warning(f"Alpaca API error fetching price for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"Could not fetch price for {symbol}: {e}")
    except ValueError as e:
        logger.error(f"Configuration error fetching price for {symbol} (e.g., missing API keys): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /api/price/{symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# API endpoint to get the trading signal for a symbol
@app.get("/api/signal/{symbol}")
async def get_signal(symbol: str):
    logger.info(f"Received request for signal of {symbol}")
    """Generates a trading signal for a given stock symbol using the defined strategy.

    Args:
        symbol: The stock ticker symbol.

    Returns:
        JSON response with symbol and calculated signal ('BUY', 'SELL', 'HOLD').

    Raises:
        HTTPException: 500 if an error occurs during signal generation.
    """
    try:
        # generate_signal already includes internal error handling and logging,
        # and returns 'HOLD' on failure. We call it directly.
        # If get_api itself fails due to missing keys, generate_signal handles it.
        signal = generate_signal(symbol)
        logger.info(f"Generated signal for {symbol}: {signal}")
        return {"symbol": symbol, "signal": signal}
    except Exception as e:
        # Catch any unexpected errors not handled within generate_signal
        # Although generate_signal aims to return 'HOLD' on errors, 
        # we add a safeguard here.
        # Also log the error for backend visibility
        logger.error(f"Unexpected error in /api/signal/{symbol} endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating the signal for {symbol}.")

# API endpoint to execute a paper trade
@app.post("/api/execute_trade")
async def execute_trade(trade_request: TradeRequest):
    logger.info(f"Received trade execution request for {trade_request.symbol}")
    """Executes a paper market order based on the provided signal.

    Args:
        trade_request (TradeRequest): JSON body containing symbol, signal ('BUY' or 'SELL'), and qty.

    Returns:
        JSON response indicating success or failure of the order placement.
        
    Raises:
        HTTPException: 400 for invalid signal (should be caught by Pydantic), 
                       500 if order placement fails.
    """
    try:
        # Pydantic validation already ensures signal is 'BUY' or 'SELL' and qty > 0
        symbol = trade_request.symbol
        signal = trade_request.signal
        qty = trade_request.qty

        # Determine order side based on signal
        side = signal.lower() # Convert 'BUY' -> 'buy', 'SELL' -> 'sell'

        logger.info(f"Received trade execution request: {side} {qty} {symbol}")

        # Place the paper order using the broker function
        order = place_paper_market_order(symbol=symbol, qty=qty, side=side)

        if order:
            logger.info(f"Paper trade successful for {symbol}: Order ID {order.id}, Status {order.status}")
            # Successfully submitted order
            # You might want to return relevant order details
            return {
                "message": "Paper trade executed successfully",
                "order_id": order.id,
                "symbol": order.symbol,
                "qty": order.qty,
                "side": order.side,
                "status": order.status
            }
        else:
            logger.error(f"Order placement failed for {symbol} (place_paper_market_order returned None). Request: {trade_request}")
            # place_paper_market_order returned None, indicating an error (already logged)
            raise HTTPException(status_code=500, detail=f"Failed to place paper order for {symbol}. Check server logs.")

    except HTTPException as http_exc:
         logger.warning(f"HTTP exception during trade execution for {trade_request.symbol}: {http_exc.status_code} - {http_exc.detail}")
         raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"Unexpected error in /api/execute_trade for {trade_request.symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during trade execution: {e}")

@app.get("/api/positions")
async def get_positions():
    """Fetches and returns all open positions from Alpaca.

    Returns:
        JSON response with a dictionary mapping symbols to quantities.
        Example: {"AAPL": 10.5, "TSLA": 5.0}

    Raises:
        HTTPException: 500 if positions cannot be fetched.
    """
    logger.info("Received request for all positions.")
    try:
        api = get_api()
        positions = get_all_positions(api)
        if positions is not None: # Check for None (indicates fetch error)
            logger.info(f"Successfully returned {len(positions)} positions.")
            return positions
        else:
            logger.error("Failed to get positions (get_all_positions returned None).")
            raise HTTPException(status_code=500, detail="Failed to fetch positions from broker.")
    except ValueError as e:
        logger.error(f"API client initialization failed for positions request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Broker API client initialization failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred while fetching positions.")

# --- Frontend Serving ---

# Root GET endpoint to serve the frontend's index.html
@app.get("/")
async def read_index():
    logger.debug("Serving index.html")
    """
    Serves the main index.html file.
    """
    return FileResponse('frontend/index.html', media_type='text/html')

# Mount static files (e.g., CSS, JS) from the 'frontend' directory
# This needs to be defined AFTER the root endpoint
logger.info("Mounting static files from 'frontend' directory...")
app.mount("/", StaticFiles(directory="frontend"), name="static")
logger.info("Static files mounted.")

# Example command to run this app:
# uvicorn main:app --reload 

logger.info("Application setup complete. Ready to run with uvicorn.") 