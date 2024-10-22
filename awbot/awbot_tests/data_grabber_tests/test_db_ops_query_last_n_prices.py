import pytest
from boto3.dynamodb.conditions import Key

# Import the actual DBOps class from your module
from awbot.data_grabber import DBOps  # Adjust this import path to match your project structure


class TestDBOps:
    @pytest.fixture(autouse=True)
    def setup_mocks(self, mocker):
        """Setup all mocks needed for testing"""
        # Mock the price_table in the module where DBOps is defined
        self.mock_price_table = mocker.patch(
            "awbot.data_grabber.price_table"
        )  # Adjust this path to match your project structure
        # Ensure the query method exists and returns a default value
        self.mock_price_table.query.return_value = {"Items": []}
        return self.mock_price_table

    def test_query_single_item_returns_list(self):
        """Test querying a single item returns a list even when DynamoDB returns a dict"""
        # Setup
        mock_item = {"ticker": "aapl", "timestamp": "2024-01-01", "price": 100.00}
        # Important: When n=1 and single item, DynamoDB returns the item directly, not in a list
        self.mock_price_table.query.return_value = {"Items": mock_item}

        # Execute
        result = DBOps.query_last_n_prices(ticker="AAPL", n=1)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == mock_item
        self.mock_price_table.query.assert_called_once_with(
            KeyConditionExpression=Key("ticker").eq("aapl"), ScanIndexForward=False, Limit=1
        )

    def test_query_single_nonexistent_item_returns_empty_list(self):
        """Test querying a single nonexistent item returns an empty list"""
        # Setup
        self.mock_price_table.query.return_value = {"Items": []}

        # Execute
        result = DBOps.query_last_n_prices(ticker="NONEXISTENT", n=1)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0
        self.mock_price_table.query.assert_called_once_with(
            KeyConditionExpression=Key("ticker").eq("nonexistent"), ScanIndexForward=False, Limit=1
        )

    def test_query_multiple_items_returns_list(self):
        """Test querying multiple items returns a list"""
        # Setup
        mock_items = [
            {"ticker": "aapl", "timestamp": "2024-01-01", "price": 100.00},
            {"ticker": "aapl", "timestamp": "2024-01-02", "price": 101.00},
            {"ticker": "aapl", "timestamp": "2024-01-03", "price": 102.00},
        ]
        self.mock_price_table.query.return_value = {"Items": mock_items}

        # Execute
        result = DBOps.query_last_n_prices(ticker="AAPL", n=3)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == mock_items
        self.mock_price_table.query.assert_called_once_with(
            KeyConditionExpression=Key("ticker").eq("aapl"), ScanIndexForward=False, Limit=3
        )

    def test_query_with_invalid_response_raises_type_error(self):
        """Test that an invalid response type from DynamoDB raises TypeError"""
        # Setup
        self.mock_price_table.query.return_value = {"Items": "invalid_type"}

        # Execute and Assert
        with pytest.raises(TypeError, match="DynamoDB query result is not a list"):
            DBOps.query_last_n_prices(ticker="AAPL", n=2)

    def test_ticker_exact_match(self):
        """Test that the ticker must match exactly in lowercase form"""
        # Setup
        expected_ticker = "aapl"
        mock_items = [{"ticker": expected_ticker, "timestamp": "2024-01-01", "price": 100.00}]

        # Test exact match
        self.mock_price_table.query.return_value = {"Items": mock_items}
        result = DBOps.query_last_n_prices(ticker="AAPL", n=1)
        assert len(result) == 1
        assert result[0]["ticker"] == expected_ticker

        # Test similar but incorrect tickers
        similar_tickers = ["aap", "aapll", "appl", "apl"]
        self.mock_price_table.query.return_value = {"Items": []}

        for incorrect_ticker in similar_tickers:
            self.mock_price_table.query.reset_mock()
            result = DBOps.query_last_n_prices(ticker=incorrect_ticker, n=1)
            assert (
                len(result) == 0
            ), f"Ticker '{incorrect_ticker}' should not match '{expected_ticker}'"
            self.mock_price_table.query.assert_called_once_with(
                KeyConditionExpression=Key("ticker").eq(incorrect_ticker),
                ScanIndexForward=False,
                Limit=1,
            )
