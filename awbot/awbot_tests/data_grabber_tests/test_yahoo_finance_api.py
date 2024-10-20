import pytest
from pytest_mock import MockerFixture
import pandas as pd
import numpy as np
from decimal import Decimal
from awbot.data_grabber import YahooFinanceAPI, convert_to_decimal

@pytest.fixture
def mock_yf_tickers(mocker: MockerFixture):
    mock_tickers = mocker.patch('yfinance.Tickers')
    mock_history = mocker.MagicMock()
    mock_tickers.return_value.history = mock_history
    return mock_history


def create_mock_df():
    dates = pd.date_range(start='2023-01-01', end='2023-01-05')

    # Define the data
    data = {
        ('AAPL', 'capital gains'): [0.0, 0.0, 0.0, 0.0, 0.0],
        ('AAPL', 'stock splits'): [np.nan, np.nan, np.nan, np.nan, np.nan],
        ('AAPL', 'close'): [150.0, 151.0, 152.0, 153.0, 154.0],
        ('AAPL', 'open'): [148.0, 149.0, 150.0, 151.0, 152.0],
        ('AAPL', 'high'): [151.0, 152.0, 153.0, 154.0, 155.0],
        ('AAPL', 'low'): [147.0, 148.0, 149.0, 150.0, 151.0],
        ('AAPL', 'dividends'): [0.0, 0.0, 0.0, 0.0, 0.0],
        ('AAPL', 'volume'): [1000000, 1100000, 1200000, 1300000, 1400000],
        ('GOOGL', 'capital gains'): [0.0, 0.0, 0.0, 0.0, 0.0],
        ('GOOGL', 'stock splits'): [np.nan, np.nan, np.nan, np.nan, np.nan],
        ('GOOGL', 'close'): [2500.0, 2510.0, 2520.0, 2530.0, 2540.0],
        ('GOOGL', 'open'): [2495.0, 2500.0, 2505.0, 2510.0, 2515.0],
        ('GOOGL', 'high'): [2515.0, 2525.0, 2535.0, 2545.0, 2555.0],
        ('GOOGL', 'low'): [2490.0, 2495.0, 2500.0, 2505.0, 2510.0],
        ('GOOGL', 'dividends'): [0.0, 0.0, 0.0, 0.0, 0.0],
        ('GOOGL', 'volume'): [1500000, 1600000, 1700000, 1800000, 1900000],
    }
    # Create the DataFrame
    df = pd.DataFrame(data, index=dates)
    df.columns = df.columns.swaplevel()
    df.columns.name = 'Symbols'
    df.index.name = 'Date'
    return df

def test_get_price_history_with_period(mock_yf_tickers):
    mock_df = create_mock_df()
    mock_yf_tickers.return_value = mock_df

    result = YahooFinanceAPI.get_price_history(['AAPL', 'GOOGL'], period='1mo')

    mock_yf_tickers.assert_called_once_with(period='1mo')
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'datetime'
    assert 'ticker' in result.columns
    assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert all(isinstance(val, Decimal) for val in result['close'])

def test_get_price_history_without_dates(mock_yf_tickers):
    mock_df = create_mock_df()
    mock_yf_tickers.return_value = mock_df

    result = YahooFinanceAPI.get_price_history(['AAPL', 'GOOGL'])

    mock_yf_tickers.assert_called_once_with(period='max')
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'datetime'
    assert 'ticker' in result.columns

def test_get_price_history_with_dates(mock_yf_tickers):
    mock_df = create_mock_df()
    mock_yf_tickers.return_value = mock_df

    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-01-05')
    result = YahooFinanceAPI.get_price_history(['AAPL', 'GOOGL'], start_date=start_date, end_date=end_date)

    mock_yf_tickers.assert_called_once_with(start=start_date, end=end_date)
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'datetime'
    assert 'ticker' in result.columns

def test_get_price_history_invalid_params():
    with pytest.raises(ValueError):
        YahooFinanceAPI.get_price_history(['AAPL', 'GOOGL'], start_date=pd.Timestamp('2023-01-01'))

def test_convert_to_decimal():
    assert convert_to_decimal(150.12345, 5) == Decimal('150.12345')
    assert convert_to_decimal(150.123456789, 5) == Decimal('150.12346')

def test_get_price_history_decimal_conversion(mock_yf_tickers):
    mock_df = create_mock_df()
    mock_yf_tickers.return_value = mock_df

    result = YahooFinanceAPI.get_price_history(['AAPL', 'GOOGL'], period='1mo')

    assert all(isinstance(val, Decimal) for val in result['close'])
    assert all(isinstance(val, Decimal) for val in result['open'])
    assert all(isinstance(val, Decimal) for val in result['high'])
    assert all(isinstance(val, Decimal) for val in result['low'])
    assert all(isinstance(val, Decimal) for val in result['volume'])

def test_get_price_history_drop_columns(mock_yf_tickers):
    mock_df = create_mock_df()
    mock_df['capital gains'] = 0
    mock_df['stock splits'] = 0
    mock_yf_tickers.return_value = mock_df

    result = YahooFinanceAPI.get_price_history(['AAPL', 'GOOGL'], period='1mo')

    assert 'capital gains' not in result.columns
    assert 'stock splits' not in result.columns

def test_get_price_history_column_names(mock_yf_tickers):
    mock_df = create_mock_df()
    mock_yf_tickers.return_value = mock_df

    result = YahooFinanceAPI.get_price_history(['AAPL', 'GOOGL'], period='1mo')

    assert all(col.lower() == col for col in result.columns)

def test_get_price_history_no_na_values(mock_yf_tickers):
    mock_df = create_mock_df()
    mock_df.loc['2023-01-03', 'AAPL'] = np.nan
    mock_yf_tickers.return_value = mock_df

    result = YahooFinanceAPI.get_price_history(['AAPL', 'GOOGL'], period='1mo')

    assert not result.isna().any().any()