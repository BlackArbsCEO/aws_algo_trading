from decimal import Decimal, ROUND_HALF_DOWN

import numpy as np
import pandas as pd
import yfinance as yf

symbols = ['BTC-USD', 'SPY', 'QQQ', 'IWM', 'TLT', 'GLD']


def convert_to_decimal(
    number: float, precision: int = 2, rounding: str = ROUND_HALF_DOWN
):
    """
    Quantizes a number to a given precision and returns a float.

    Args:
        number (float): The number to quantize.
        precision (float, optional): The precision to quantize to. Defaults to 2.
        rounding (str, optional): The rounding mode to use. Defaults to ROUND_HALF_DOWN.

    Returns:
        Decimal: The quantized number.
    """
    try:
        number_decimal = Decimal(str(number))
    except Exception as e:
        raise Exception(f'unable to convert {number} to decimal: {e}')

    # Set the precision to the desired number of decimal places
    precision_decimal = Decimal("0." + "0" * precision)

    # Quantize the number using ROUND_HALF_DOWN rounding mode
    quantized_decimal = number_decimal.quantize(precision_decimal, rounding=rounding)
    return quantized_decimal


def get_price_history(
    symbols: list,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    period: str = None,
) -> pd.DataFrame:
    """
    Downloads the price history for a given list of symbols from Yahoo Finance.

    Parameters
    ----------
    symbols : list
        A list of symbols to download the price history for.
    start_date : pd.Timestamp, optional
        The start date for the price history. If not provided, the entire price
        history will be downloaded.
    end_date : pd.Timestamp, optional
        The end date for the price history. If not provided, the price history
        will be downloaded until the current date.
    period : str, optional
        The period for which to download the price history. If not provided, the
        entire price history will be downloaded. Valid values are "1mo", "3mo",
        "6mo", "1y", "2y", "5y", "10y", and "ytd".

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the price history for the given symbols. The
        index is a datetime index, and the columns are "close", "high", "low",
        and "open". The values are Decimals, which are suitable for use with
        DynamoDB.
    """
    drop_cols = ['capital gains', 'stock splits']
    if period is not None:
        prices = (
            yf.Tickers(" ".join(symbols))
            .history(period=period)
            .drop(drop_cols, axis=1)
            .rename(columns=str.lower)
            .dropna()
            .stack(level=1, future_stack=True)
            .rename_axis(["datetime", "ticker"])
            .reset_index(level=1)
        )
    elif start_date is None and period is None:
        prices = (
            yf.Tickers(" ".join(symbols))
            .history(period="max")
            .rename(columns=str.lower)
            .drop(drop_cols, axis=1)
            .dropna()
            .stack(level=1, future_stack=True)
            .rename_axis(["datetime", "ticker"])
            .reset_index(level=1)
        )
    elif start_date is not None and end_date is not None:
        prices = (
            yf.Tickers(" ".join(symbols))
            .history(start=start_date, end=end_date)
            .rename(columns=str.lower)
            .drop(drop_cols, axis=1)
            .dropna()
            .stack(level=1, future_stack=True)
            .rename_axis(["datetime", "ticker"])
            .reset_index(level=1)
        )
    else:
        raise ValueError(f'{period=} {start_date=} {end_date=} can not all be None or invalid')

    # convert floats to Decimals for dynamodb
    for col in prices.select_dtypes(include=[np.number]).columns:
        prices[col] = prices[col].apply(
            lambda x: convert_to_decimal(x, 5)
        )
    return prices