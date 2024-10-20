import pytest
from pytest_mock import MockerFixture
from awbot.data_grabber import warmup_asset_data, DBOps, YahooFinanceAPI, logger

@pytest.fixture
def mock_logger(mocker: MockerFixture):
    return mocker.patch('awbot.data_grabber.logger')

@pytest.fixture
def mock_dbops(mocker: MockerFixture):
    return mocker.patch('awbot.data_grabber.DBOps')

@pytest.fixture
def mock_yahoo_finance_api(mocker: MockerFixture):
    return mocker.patch('awbot.data_grabber.YahooFinanceAPI')

def test_warmup_asset_data_existing_item(mock_logger, mock_dbops):
    symbols = ['AAPL']
    mock_item = {'symbol': 'AAPL', 'price': 150.0}
    mock_dbops.query_last_n_prices.return_value = mock_item

    warmup_asset_data(symbols)

    mock_dbops.query_last_n_prices.assert_called_once_with('AAPL', n=1)
    mock_logger.info.assert_called_once_with(f"Last record for AAPL:\n{mock_item}")
    mock_dbops.put_price_data_in_table.assert_not_called()

def test_warmup_asset_data_no_existing_item(mock_logger, mock_dbops, mock_yahoo_finance_api):
    symbols = ['GOOGL']
    mock_dbops.query_last_n_prices.return_value = None
    mock_prices = {'GOOGL': [{'date': '2023-04-20', 'price': 105.0}]}
    mock_yahoo_finance_api.get_price_history.return_value = mock_prices

    warmup_asset_data(symbols)

    mock_dbops.query_last_n_prices.assert_called_once_with('GOOGL', n=1)
    # Check if the first log message is for "No records found"
    mock_logger.info.assert_any_call("No records found for GOOGL bulk inserting all available data...")
    # Check if the second log message is for bulk load completion
    mock_logger.info.assert_any_call("price data bulk loaded for ['GOOGL']...[DONE]")

    mock_yahoo_finance_api.get_price_history.assert_called_once_with(symbols)
    mock_dbops.put_price_data_in_table.assert_called_once_with(mock_prices, bulk_insert=True, overwrite=True)

def test_warmup_asset_data_query_exception(mock_logger, mock_dbops, mock_yahoo_finance_api):
    symbols = ['TSLA']
    mock_dbops.query_last_n_prices.side_effect = Exception("Connection error")
    mock_prices = {'TSLA': [{'date': '2023-04-20', 'price': 200.0}]}
    mock_yahoo_finance_api.get_price_history.return_value = mock_prices

    warmup_asset_data(symbols)

    mock_dbops.query_last_n_prices.assert_called_once_with('TSLA', n=1)
    mock_logger.error.assert_called_once()
    mock_yahoo_finance_api.get_price_history.assert_called_once_with(symbols)
    mock_dbops.put_price_data_in_table.assert_called_once_with(mock_prices, bulk_insert=True, overwrite=True)

def test_warmup_asset_data_multiple_symbols(mock_logger, mock_dbops):
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    mock_item = {'symbol': 'AAPL', 'price': 150.0}
    mock_dbops.query_last_n_prices.return_value = mock_item

    warmup_asset_data(symbols)

    mock_dbops.query_last_n_prices.assert_called_once_with('AAPL', n=1)
    mock_logger.info.assert_called_once_with(f"Last record for AAPL:\n{mock_item}")
    mock_dbops.put_price_data_in_table.assert_not_called()