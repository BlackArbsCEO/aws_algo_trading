from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import boto3
import numpy as np
import pandas as pd
import yfinance as yf
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from loguru import logger

from awbot.data_utils import convert_to_decimal, quantize_number


@dataclass
class DBOps:
    @staticmethod
    def put_price_items_with_condition(items: list[dict[str, str | float]]) -> None:
        """
        Puts a list of price items into the DynamoDB table, with a condition check
        to prevent overwriting existing data.

        Parameters
        ----------
        items : list[dict[str, str | float]]
            A list of dictionaries, each representing a price data item. The
            dictionaries must have the following keys:
            - ticker: str
            - timestamp: Decimal
            - open: Decimal
            - high: Decimal
            - low: Decimal
            - close: Decimal
            - volume: Decimal
            - dividend: Decimal

        Returns
        -------
        None

        Notes
        -----
        The items are added to the DynamoDB table one by one, with a condition check
        to prevent overwriting existing data. If the item already exists, a
        ConditionalCheckFailedException is raised and logged as a warning. Any other
        ClientError is logged as an exception.
        """
        for item in items:
            try:
                # Conditional put item
                response = price_table.put_item(
                    Item=item,
                    # ConditionExpression ensures that the item is only added if it does not already exist
                    ConditionExpression="attribute_not_exists(#ts)",
                    ExpressionAttributeNames={"#ts": "timestamp"},
                )
                logger.info(
                    f"PutItem succeeded for ticker '{item['ticker']}' and timestamp '{item['timestamp']}':"
                )
                logger.info(response)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                    logger.warning(
                        f"Item with ticker-timestamp '{item['ticker']}'-'{item['timestamp']}' already exists. No overwrite occurred."
                    )
                else:
                    logger.exception(f"Unexpected error: {e}")
                    raise
        return

    @staticmethod
    def put_price_data_in_table(
        df: pd.DataFrame, bulk_insert: bool = False, overwrite: bool = False
    ) -> None:
        """
        Inserts price data from a DataFrame into a DynamoDB table.

        This function converts the DataFrame rows into a list of dictionaries, each
        representing a price data item, and writes them to a DynamoDB table. If
        `bulk_insert` or `overwrite` is set to True, the data is inserted in bulk
        using a batch writer. Otherwise, individual items are inserted with a
        conditional check to prevent overwriting existing data.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing price data with columns 'ticker', 'timestamp',
            'open', 'high', 'low', and 'close'.
        bulk_insert : bool, optional
            If True, the items are inserted in bulk. Defaults to False.
        overwrite : bool, optional
            If True, existing items are overwritten during bulk insert. Defaults to False.

        Returns
        -------
        None
        """
        items = [
            {
                "ticker": row["ticker"],
                "timestamp": pd.to_datetime(timestamp).strftime("%Y-%m-%dT%H:%M:%S"),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "dividends": row["dividends"],
            }
            for timestamp, row in df.iterrows()
        ]
        if bulk_insert or overwrite:
            # Use batch writer to insert items in bulk
            with price_table.batch_writer() as batch:
                for item in items:
                    batch.put_item(Item=item)
            logger.info(f"Inserted {len(items)} items into DynamoDB")
        else:
            DBOps.put_price_items_with_condition(items)
        return

    @staticmethod
    def query_last_n_prices(ticker: str, n: int) -> List[Dict[str, Any]]:
        """
        Queries the DynamoDB price table for the last N periods of prices
        for the given ticker. It will return a list of dictionaries, where
        each dictionary is a record returned from DynamoDB.

        Parameters
        ----------
        ticker : str
            The ticker symbol to query.
        n : int
            The number of records to return.

        Returns
        -------
        list
            A list of dictionaries, where each dictionary is a record
            returned from DynamoDB.
        """
        response = price_table.query(
            KeyConditionExpression=Key("ticker").eq(ticker.lower()),
            ScanIndexForward=False,  # This orders results in descending order
            Limit=n,  # Limit the results to the last N records
        )
        return response["Items"]

    @staticmethod
    def get_last_n_prices(symbols: list, n: int, column: str = "close") -> pd.DataFrame:
        """
        Retrieves the last N periods of prices for the given list of symbols from the DynamoDB price table.

        Parameters
        ----------
        symbols : list
            The list of symbols to query prices for.
        n : int
            The number of periods to retrieve.
        column : str, optional
            The column to retrieve from the DynamoDB price table (default is 'close').

        Returns
        -------
        pd.DataFrame
            A DataFrame with the specified column for each symbol, indexed by timestamp.
        """
        records = pd.DataFrame()
        for symbol in symbols:
            symbol = symbol.lower()
            record = DBOps.query_last_n_prices(symbol, n)
            record_series = (
                pd.DataFrame.from_records(record)
                .assign(timestamp=lambda df: pd.DatetimeIndex(df["timestamp"]))
                .set_index("timestamp")[column]
                .rename(symbol)
                .map(quantize_number)
            )
            records = pd.concat([records, record_series], axis=1).sort_index()
            records.index.name = "timestamp"
        return records


@dataclass
class YahooFinanceAPI:
    @staticmethod
    def get_price_history(
        symbols: list,
        start_date: Optional[Union[pd.Timestamp, datetime.date]] = None,
        end_date: Optional[Union[pd.Timestamp, datetime.date]] = None,
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
        drop_cols = ["capital gains", "stock splits"]
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
            raise ValueError(
                f"{period=} {start_date=} {end_date=} can not all be None or invalid"
            )

        # convert floats to Decimals for dynamodb
        for col in prices.select_dtypes(include=[np.number]).columns:
            prices[col] = prices[col].apply(lambda x: convert_to_decimal(x, 5))
        return prices


def update_price_table(symbols: list):
    """
    Updates the DynamoDB table by getting the most recent close prices for the
    given list of symbols, and then comparing the most recent date in the table
    with today's date. If the dates are different, it will get the new prices
    between the last date in the table and today, and then update the table with
    the new prices.

    Parameters
    ----------
    symbols : list
        A list of stock symbols to update the table with.

    Returns
    -------
    None
    """
    most_recent_closes = DBOps.get_last_n_prices(symbols, 1, column="close")
    today = pd.to_datetime("today").date()
    last_date = most_recent_closes.index[-1].date()
    if today != last_date:
        if (today - last_date).days == 1:
            new_prices = YahooFinanceAPI.get_price_history(symbols, period="1d")
        else:
            new_prices = YahooFinanceAPI.get_price_history(
                symbols,
                start_date=last_date + pd.Timedelta(days=1),
                end_date=today,
            )
        DBOps.put_price_data_in_table(new_prices)

    return


def warmup_asset_data(symbols: list):
    """
    Warm up the DynamoDB table by querying the last price for the first symbol in the list.
    If the item does not exist, it will be added to the table. This is useful for avoiding
    cold start issues in AWS Lambda.

    Parameters
    ----------
    symbols : list
        A list of stock symbols to query prices for.

    Returns
    -------
    None
    """
    symbol = symbols[0]
    item = None
    try:
        item = DBOps.query_last_n_prices(symbol, n=1)
    except Exception as ClientError:
        logger.error(ClientError)
    finally:
        if item is not None:
            logger.info(f"Last record for {symbol}:\n{item}")
        else:
            logger.info(
                f"No records found for {symbol} bulk inserting all available data..."
            )
            # initialize prices
            prices = YahooFinanceAPI.get_price_history(symbols)
            DBOps.put_price_data_in_table(prices, bulk_insert=True, overwrite=True)
            logger.info(f"price data bulk loaded for {symbols}...[DONE]")

    return


def init_dynamodb():
    """
    Initializes the DynamoDB table by logging into the "aws_algo_trader" profile and
    setting the region to "us-east-1". The function returns a DynamoDB Table object
    for the "aws_price_table" table.

    Returns
    -------
    boto3.resources.factory.dynamodb.Table
    """
    session = boto3.Session(profile_name="aws_algo_trader")
    dynamodb = session.resource("dynamodb", region_name="us-east-1")
    return dynamodb.Table("aws_price_table")


price_table = init_dynamodb()


if __name__ == "__main__":
    tickers = ["BTC-USD", "SPY", "QQQ", "IWM", "TLT", "GLD"]
    warmup_asset_data(tickers)
    update_price_table(tickers)
