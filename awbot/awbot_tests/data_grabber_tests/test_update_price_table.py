import pytest
from pytest_mock import MockerFixture
import pandas as pd
from decimal import Decimal
from datetime import date
from awbot.data_grabber import update_price_table


@pytest.fixture
def mock_dbops(mocker: MockerFixture):
    return mocker.patch('awbot.data_grabber.DBOps')


@pytest.fixture
def mock_yahoo_finance_api(mocker: MockerFixture):
    return mocker.patch('awbot.data_grabber.YahooFinanceAPI')


def create_mock_prices(date):
    return pd.DataFrame({
        'close': [Decimal('100.00'), Decimal('200.00')],
        'ticker': ['AAPL', 'GOOGL']
    }, index=pd.DatetimeIndex([date, date], name='datetime'))


def test_update_price_table_same_day(mock_dbops, mock_yahoo_finance_api, mocker):
    symbols = ['AAPL', 'GOOGL']
    today = pd.Timestamp('2023-05-01')
    mocker.patch('pandas.to_datetime', return_value=today)

    mock_dbops.get_last_n_prices.return_value = create_mock_prices(today)

    update_price_table(symbols)

    mock_dbops.get_last_n_prices.assert_called_once_with(symbols, 1, column="close")
    mock_yahoo_finance_api.get_price_history.assert_not_called()
    mock_dbops.put_price_data_in_table.assert_not_called()


def test_update_price_table_one_day_difference(mock_dbops, mock_yahoo_finance_api, mocker):
    symbols = ['AAPL', 'GOOGL']
    today = pd.Timestamp('2023-05-02')
    yesterday = pd.Timestamp('2023-05-01')
    mocker.patch('pandas.to_datetime', return_value=today)

    mock_dbops.get_last_n_prices.return_value = create_mock_prices(yesterday)
    mock_yahoo_finance_api.get_price_history.return_value = create_mock_prices(today)

    update_price_table(symbols)

    mock_dbops.get_last_n_prices.assert_called_once_with(symbols, 1, column="close")
    mock_yahoo_finance_api.get_price_history.assert_called_once_with(symbols, period="1d")
    mock_dbops.put_price_data_in_table.assert_called_once()


def test_update_price_table_multiple_days_difference(mock_dbops, mock_yahoo_finance_api, mocker):
    symbols = ['AAPL', 'GOOGL']
    today = pd.Timestamp('2023-05-05')
    last_date = pd.Timestamp('2023-05-01')
    mocker.patch('pandas.to_datetime', return_value=today)

    mock_dbops.get_last_n_prices.return_value = create_mock_prices(last_date)
    mock_yahoo_finance_api.get_price_history.return_value = create_mock_prices(today)

    update_price_table(symbols)

    mock_dbops.get_last_n_prices.assert_called_once_with(symbols, 1, column="close")
    mock_yahoo_finance_api.get_price_history.assert_called_once_with(
        symbols,
        start_date=date(2023, 5, 2),  # This is now a datetime.date object
        end_date=date(2023, 5, 5)  # This is also a datetime.date object
    )
    mock_dbops.put_price_data_in_table.assert_called_once()


def test_update_price_table_empty_result(mock_dbops, mock_yahoo_finance_api, mocker):
    symbols = ['AAPL', 'GOOGL']
    today = pd.Timestamp('2023-05-01')
    mocker.patch('pandas.to_datetime', return_value=today)

    mock_dbops.get_last_n_prices.return_value = pd.DataFrame()

    with pytest.raises(IndexError):
        update_price_table(symbols)

    mock_dbops.get_last_n_prices.assert_called_once_with(symbols, 1, column="close")
    mock_yahoo_finance_api.get_price_history.assert_not_called()
    mock_dbops.put_price_data_in_table.assert_not_called()


def test_update_price_table_error_handling(mock_dbops, mock_yahoo_finance_api, mocker):
    symbols = ['AAPL', 'GOOGL']
    today = pd.Timestamp('2023-05-02')
    yesterday = pd.Timestamp('2023-05-01')
    mocker.patch('pandas.to_datetime', return_value=today)

    mock_dbops.get_last_n_prices.return_value = create_mock_prices(yesterday)
    mock_yahoo_finance_api.get_price_history.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        update_price_table(symbols)

    mock_dbops.get_last_n_prices.assert_called_once_with(symbols, 1, column="close")
    mock_yahoo_finance_api.get_price_history.assert_called_once()
    mock_dbops.put_price_data_in_table.assert_not_called()