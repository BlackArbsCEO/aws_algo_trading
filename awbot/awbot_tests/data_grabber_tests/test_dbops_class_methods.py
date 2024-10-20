import pytest
from pytest_mock import MockerFixture
import pandas as pd
from decimal import Decimal
from botocore.exceptions import ClientError
from awbot.data_grabber import DBOps, init_dynamodb  # Update this import to match your module structure


@pytest.fixture
def mock_price_table(mocker: MockerFixture):
    mock_table = mocker.MagicMock()
    mocker.patch('awbot.data_grabber.init_dynamodb', return_value=mock_table)
    mocker.patch('awbot.data_grabber.price_table', mock_table)
    return mock_table


def test_put_price_items_with_condition(mock_price_table):
    items = [
        {'ticker': 'AAPL', 'timestamp': '2023-01-01T00:00:00', 'open': Decimal('150.0'), 'high': Decimal('155.0'),
         'low': Decimal('149.0'), 'close': Decimal('153.0'), 'volume': Decimal('1000000'), 'dividend': Decimal('0.0')},
        {'ticker': 'GOOGL', 'timestamp': '2023-01-01T00:00:00', 'open': Decimal('2500.0'), 'high': Decimal('2550.0'),
         'low': Decimal('2490.0'), 'close': Decimal('2540.0'), 'volume': Decimal('500000'), 'dividend': Decimal('0.0')}
    ]

    # Test successful put_item
    mock_price_table.put_item.return_value = {'ResponseMetadata': {'HTTPStatusCode': 200}}
    DBOps.put_price_items_with_condition(items)
    assert mock_price_table.put_item.call_count == 2

    # Test ConditionalCheckFailedException
    mock_price_table.put_item.side_effect = ClientError({'Error': {'Code': 'ConditionalCheckFailedException'}},
                                                        'put_item')
    DBOps.put_price_items_with_condition(items)
    assert mock_price_table.put_item.call_count == 4  # 2 more calls

    # Test other ClientError
    mock_price_table.put_item.side_effect = ClientError({'Error': {'Code': 'OtherError'}}, 'put_item')
    with pytest.raises(ClientError):
        DBOps.put_price_items_with_condition(items)


def test_put_price_data_in_table(mock_price_table, mocker: MockerFixture):
    df = pd.DataFrame({
        'ticker': ['AAPL', 'AAPL'],
        'open': [150.0, 151.0],
        'high': [155.0, 156.0],
        'low': [149.0, 150.0],
        'close': [153.0, 154.0],
        'volume': [1000000, 1100000],
        'dividends': [0.0, 0.0]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))

    # Test without bulk_insert and overwrite
    mocker.patch('awbot.data_grabber.DBOps.put_price_items_with_condition')
    DBOps.put_price_data_in_table(df)
    DBOps.put_price_items_with_condition.assert_called_once()

    # Test with bulk_insert
    mock_batch_writer = mocker.MagicMock()
    mock_price_table.batch_writer.return_value.__enter__.return_value = mock_batch_writer
    DBOps.put_price_data_in_table(df, bulk_insert=True)
    assert mock_batch_writer.put_item.call_count == 2

    # Test with overwrite
    DBOps.put_price_data_in_table(df, overwrite=True)
    assert mock_batch_writer.put_item.call_count == 4  # 2 more calls


def test_query_last_n_prices(mock_price_table, mocker:MockerFixture):
    mock_price_table.query.return_value = {
        'Items': [
            {'ticker': 'AAPL', 'timestamp': '2023-01-02T00:00:00', 'close': Decimal('154.0')},
            {'ticker': 'AAPL', 'timestamp': '2023-01-01T00:00:00', 'close': Decimal('153.0')}
        ]
    }

    result = DBOps.query_last_n_prices('AAPL', 2)

    mock_price_table.query.assert_called_once_with(
        KeyConditionExpression=mocker.ANY,
        ScanIndexForward=False,
        Limit=2
    )
    assert len(result) == 2
    assert result[0]['ticker'] == 'AAPL'
    assert result[0]['close'] == Decimal('154.0')


def test_get_last_n_prices(mocker: MockerFixture):
    mock_query_last_n_prices = mocker.patch('awbot.data_grabber.DBOps.query_last_n_prices')
    mock_query_last_n_prices.side_effect = [
        [
            {'ticker': 'AAPL', 'timestamp': '2023-01-02T00:00:00', 'close': Decimal('154.0')},
            {'ticker': 'AAPL', 'timestamp': '2023-01-01T00:00:00', 'close': Decimal('153.0')}
        ],
        [
            {'ticker': 'GOOGL', 'timestamp': '2023-01-02T00:00:00', 'close': Decimal('2540.0')},
            {'ticker': 'GOOGL', 'timestamp': '2023-01-01T00:00:00', 'close': Decimal('2530.0')}
        ]
    ]

    result = DBOps.get_last_n_prices(['AAPL', 'GOOGL'], 2)

    assert mock_query_last_n_prices.call_count == 2
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['aapl', 'googl']
    assert len(result) == 2
    assert result.index.name == 'timestamp'