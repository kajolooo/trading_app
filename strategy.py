# strategy.py
import pandas as pd
import numpy as np
import logging
from broker import get_api
from alpaca_trade_api.rest import APIError, TimeFrame

# Get logger instance
logger = logging.getLogger(__name__)

# Parameters (Updated based on optimization)
SHORT_WINDOW = 10 # Was 12
LONG_WINDOW = 20  # Was 26
RSI_PERIOD = 30   # Was 14
RSI_OVERBOUGHT = 70 # Unchanged
RSI_OVERSOLD = 30   # Unchanged
# MACD_SHORT = 12   # MACD Not Used
# MACD_LONG = 26    # MACD Not Used
# MACD_SIGNAL = 9   # MACD Not Used
DATA_LIMIT = 1000 # Increased limit for minute data

# ATR Parameters (Needed for future exit logic - add constants now)
ATR_PERIOD = 5
STOP_ATR_MULTIPLIER = 6.0
TAKE_PROFIT_ATR_MULTIPLIER = 9.0

def calculate_indicators(symbol: str, api):
    """
    Fetches historical data and calculates SMA and RSI indicators.
    MACD calculation removed.
    ATR calculation added (but not used in signals yet).
    Uses Minute data.
    """
    logger.debug(f"Calculating indicators for {symbol} using {DATA_LIMIT} data points (Minute timeframe).")
    try:
        logger.debug(f"Fetching bars for {symbol}...")
        bars = api.get_bars(symbol, TimeFrame.Minute, limit=DATA_LIMIT).df
        logger.debug(f"Fetched {len(bars)} bars for {symbol}.")

        if not isinstance(bars, pd.DataFrame) or bars.empty:
            logger.error(f"Could not fetch valid bar data for {symbol}. Received: {type(bars)}")
            return None

        # --- Data Cleaning & Preparation ---
        if not isinstance(bars.index, pd.DatetimeIndex):
            bars.index = pd.to_datetime(bars.index)
        if bars.index.tz is None:
             bars.index = bars.index.tz_localize('UTC')
        else:
             bars.index = bars.index.tz_convert('UTC')

        df = bars.copy()
        if 'close' not in df.columns:
            logger.error(f"'close' column not found in data for {symbol}. Columns: {df.columns}")
            return None

        # Adjust min_data_needed based on active indicators
        min_data_needed = max(LONG_WINDOW, RSI_PERIOD, ATR_PERIOD)
        if len(df) < min_data_needed:
            logger.warning(f"Insufficient data for {symbol} (need {min_data_needed}, got {len(df)}).")
            return None

        logger.debug(f"Calculating SMAs, RSI, and ATR for {symbol}...")

        # --- Calculate SMAs ---
        df['short_sma'] = df['close'].rolling(window=SHORT_WINDOW).mean()
        df['long_sma'] = df['close'].rolling(window=LONG_WINDOW).mean()

        # --- Calculate RSI ---
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        # Use Simple Moving Average for Avg Gain/Loss in Wilder's RSI (closer to backtrader's default)
        # Use alpha = 1 / period for Wilder's smoothing
        avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
        avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
        # Handle potential division by zero for RS
        rs = avg_gain / avg_loss.replace(0, 1e-9) # Replace 0 loss with small number
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df['rsi'] = df['rsi'].fillna(50) # Fill initial NaNs

        # --- Calculate ATR (Added) ---
        # ATR calculation needs high, low, close
        if not all(col in df.columns for col in ['high', 'low', 'close']):
             logger.error("ATR calculation requires 'high', 'low', 'close' columns.")
             return None # Cannot calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()

        logger.info(f"Successfully calculated indicators for {symbol}.")
        return df

    except APIError as e:
        logger.error(f"Alpaca API error fetching bars for {symbol}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error calculating indicators for {symbol}: {e}", exc_info=True)
        return None

def generate_signal(symbol: str, data_df: pd.DataFrame = None):
    """
    Generates a trading signal (BUY, SELL, HOLD) based on SMA crossover
    and RSI confirmation. MACD removed.
    ATR is calculated but not used for signals here (requires state).
    """
    try:
        df = data_df
        if df is None:
            logger.info(f"No pre-calculated data passed for {symbol}, fetching fresh data...")
            api = get_api()
            df = calculate_indicators(symbol, api)
        else:
            logger.info(f"Using pre-calculated data for {symbol} signal generation.")

        if df is None or len(df) < 2:
            logger.warning(f"Could not generate signal for {symbol} due to insufficient data or calculation error.")
            return 'HOLD'

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # Update required indicators
        required_indicators = ['short_sma', 'long_sma', 'rsi'] # Removed macd
        if pd.isna(latest[required_indicators]).any() or pd.isna(previous[required_indicators]).any():
            logger.warning(f"NaN values detected in indicators for {symbol}. Defaulting to HOLD.")
            nan_check_latest = latest[required_indicators].isna()
            nan_check_prev = previous[required_indicators].isna()
            if nan_check_latest.any(): logger.warning(f"NaN in latest indicators: {nan_check_latest[nan_check_latest].index.tolist()}")
            if nan_check_prev.any(): logger.warning(f"NaN in previous indicators: {nan_check_prev[nan_check_prev].index.tolist()}")
            return 'HOLD'

        # --- Define Signal Conditions (SMA + RSI only) ---
        signal = 'HOLD'
        sma_golden_cross = latest['short_sma'] > latest['long_sma'] and previous['short_sma'] <= previous['long_sma']
        sma_death_cross = latest['short_sma'] < latest['long_sma'] and previous['short_sma'] >= previous['long_sma']
        rsi_not_overbought = latest['rsi'] < RSI_OVERBOUGHT
        rsi_not_oversold = latest['rsi'] > RSI_OVERSOLD

        # BUY condition (SMA Golden Cross + RSI Confirmation)
        if sma_golden_cross and rsi_not_overbought:
            signal = 'BUY'
        # SELL condition (SMA Death Cross + RSI Confirmation)
        elif sma_death_cross and rsi_not_oversold:
            signal = 'SELL'

        # --- Corrected Log Message (Removed MACD) --- 
        logger.info(
            f"Signal for {symbol}: {signal}. "
            f"Close={latest['close']:.2f}, "
            f"SMA({SHORT_WINDOW})={latest['short_sma']:.2f}, "
            f"SMA({LONG_WINDOW})={latest['long_sma']:.2f}, "
            f"RSI({RSI_PERIOD})={latest['rsi']:.2f}"
            # MACD log removed
        )
        return signal

    except ValueError as e:
        logger.error(f"Configuration error generating signal ({symbol}): {e}", exc_info=True)
        return 'HOLD'
    except Exception as e:
        logger.error(f"Unexpected error generating signal for {symbol}: {e}", exc_info=True)
        return 'HOLD' 