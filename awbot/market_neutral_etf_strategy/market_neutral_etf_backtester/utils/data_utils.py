from loguru import logger
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


def load_market_data(
    symbols: List[str],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    data_source: str = "yahoo",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Load market data for multiple symbols.

    Parameters:
        symbols: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        data_source: Source of data (currently supports 'yahoo')
        interval: Data frequency ('1d', '1h', etc.)

    Returns:
        Dictionary of DataFrames with market data for each symbol
    """
    logger.info(f"Loading market data for {len(symbols)} symbols")

    if data_source.lower() == "yahoo":
        try:
            # Download data for all symbols at once for efficiency
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
            )

            # Reorganize multi-level columns into separate dataframes
            market_data = {}
            for symbol in symbols:
                if len(symbols) > 1:
                    symbol_data = data[symbol].copy()
                else:
                    symbol_data = data.copy()

                symbol_data.columns = symbol_data.columns.str.lower()
                market_data[symbol] = _validate_and_clean_data(symbol_data, symbol)

            return market_data

        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise

    else:
        raise ValueError(f"Unsupported data source: {data_source}")


def prepare_backtest_data(
    market_data: Dict[str, pd.DataFrame], pairs_config: List[Tuple[str, str, float]]
) -> Dict[Tuple[str, str, float], Dict[str, pd.DataFrame]]:
    """
    Prepare market data for backtesting by pairing instruments.

    Parameters:
        market_data: Dictionary of market data by symbol
        pairs_config: List of tuples (base_symbol, inverse_symbol, leverage)

    Returns:
        Dictionary of paired market data
    """
    pairs_data = {}

    for base_symbol, inverse_symbol, leverage in pairs_config:
        if base_symbol not in market_data or inverse_symbol not in market_data:
            raise KeyError(f"Missing data for pair {base_symbol}-{inverse_symbol}")

        # Get data for each symbol
        base_data = market_data[base_symbol].copy()
        inverse_data = market_data[inverse_symbol].copy()

        # Align dates
        common_dates = base_data.index.intersection(inverse_data.index)
        if len(common_dates) == 0:
            raise DataValidationError(
                f"No overlapping dates for pair {base_symbol}-{inverse_symbol}"
            )

        base_data = base_data.loc[common_dates]
        inverse_data = inverse_data.loc[common_dates]

        # Store paired data
        pairs_data[(base_symbol, inverse_symbol, leverage)] = {
            base_symbol: base_data,
            inverse_symbol: inverse_data,
        }

        logger.info(f"Prepared pair {base_symbol}-{inverse_symbol} with {len(common_dates)} dates")

    return pairs_data


def calculate_returns(
    df: pd.DataFrame, method: str = "log", column: str = "adj close"
) -> pd.Series:
    """
    Calculate returns from price data.

    Parameters:
        df: DataFrame containing price data
        method: Return calculation method ('log' or 'simple')
        column: Column to use for price data

    Returns:
        Series of returns
    """
    if method.lower() == "log":
        returns = np.log(df[column] / df[column].shift(1))
    elif method.lower() == "simple":
        returns = df[column].pct_change()
    else:
        raise ValueError(f"Unsupported return calculation method: {method}")

    return returns


def calculate_volatility(
    returns: pd.Series, window: int = 21, annualize: bool = True, trading_days: int = 252
) -> pd.Series:
    """
    Calculate rolling volatility of returns.

    Parameters:
        returns: Series of returns
        window: Rolling window size
        annualize: Whether to annualize the volatility
        trading_days: Number of trading days per year

    Returns:
        Series of volatility values
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol


def _validate_and_clean_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate and clean market data.

    Parameters:
        df: Raw market data DataFrame
        symbol: Symbol for the data

    Returns:
        Cleaned DataFrame
    """
    required_columns = ["open", "high", "low", "close", "volume"]

    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns for {symbol}: {missing_cols}")

    # Check for missing values
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values in {symbol}:\n{missing_values[missing_values > 0]}")

        # Forward fill prices, zero fill volume
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
        df["volume"] = df["volume"].fillna(0)

    # Validate price relationships
    invalid_prices = (
        (df["high"] < df["low"])
        | (df["close"] < df["low"])
        | (df["close"] > df["high"])
        | (df["open"] < df["low"])
        | (df["open"] > df["high"])
    )

    if invalid_prices.any():
        logger.warning(f"Found {invalid_prices.sum()} invalid price relationships in {symbol}")
        df.loc[invalid_prices, ["open", "high", "low", "close"]] = np.nan
        df.ffill(inplace=True)

    # Remove zero prices
    zero_prices = (df[["open", "high", "low", "close"]] == 0).any(axis=1)
    if zero_prices.any():
        logger.warning(f"Found {zero_prices.sum()} zero prices in {symbol}")
        df.loc[:, ["open", "high", "low", "close"]] = (
            df[["open", "high", "low", "close"]].replace(0, np.nan).ffill()
        )

    # Add adjusted prices if missing
    if "adj close" not in df.columns:
        df["adj close"] = df["close"]
        logger.info(f"Added adj close column for {symbol}")

    return df


def calculate_market_impact(
    price: float,
    quantity: float,
    adv: float,
    participation_rate: float = 0.1,
    nonlinear_factor: float = 0.5,
) -> float:
    """
    Calculate market impact for a trade.

    Parameters:
        price: Current price
        quantity: Trade quantity
        adv: Average daily volume
        participation_rate: Target participation rate
        nonlinear_factor: Nonlinearity of impact

    Returns:
        Estimated market impact cost
    """
    trade_value = abs(price * quantity)
    trade_participation = trade_value / adv

    # Scale impact by participation rate and nonlinearity
    if trade_participation > participation_rate:
        excess_participation = (trade_participation / participation_rate) ** nonlinear_factor
        impact = trade_value * 0.0001 * excess_participation
    else:
        impact = trade_value * 0.0001

    # Cap impact at reasonable level
    max_impact = trade_value * 0.01  # 1% cap
    return min(impact, max_impact)


def get_market_depths(symbols: List[str], lookback_days: int = 21) -> Dict[str, float]:
    """
    Calculate market depths for symbols.

    Parameters:
        symbols: List of symbols
        lookback_days: Days to use for calculation

    Returns:
        Dictionary of market depths by symbol
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)

    market_data = load_market_data(symbols, start_date, end_date)
    depths = {}

    for symbol, data in market_data.items():
        # Calculate average daily dollar volume
        daily_volume = data["adj close"] * data["volume"]
        depths[symbol] = daily_volume.mean()

    return depths


def resample_data(
    df: pd.DataFrame, freq: str, price_col: str = "adj close", volume_col: str = "volume"
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different frequency.

    Parameters:
        df: OHLCV DataFrame
        freq: Target frequency ('1H', '1D', etc.)
        price_col: Column name for price data
        volume_col: Column name for volume data

    Returns:
        Resampled DataFrame
    """
    resampler = df.resample(freq)

    result = pd.DataFrame(
        {
            "open": resampler[price_col].first(),
            "high": resampler[price_col].max(),
            "low": resampler[price_col].min(),
            "close": resampler[price_col].last(),
            "volume": resampler[volume_col].sum(),
        }
    )

    return result


def align_data(dfs: List[pd.DataFrame], method: str = "inner") -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to common dates.

    Parameters:
        dfs: List of DataFrames to align
        method: Join method ('inner' or 'outer')

    Returns:
        List of aligned DataFrames
    """
    if not dfs:
        return []

    # Get common index
    common_idx = dfs[0].index
    for df in dfs[1:]:
        if method == "inner":
            common_idx = common_idx.intersection(df.index)
        else:  # outer
            common_idx = common_idx.union(df.index)

    # Align all dataframes
    aligned_dfs = [df.reindex(common_idx) for df in dfs]
    return aligned_dfs
