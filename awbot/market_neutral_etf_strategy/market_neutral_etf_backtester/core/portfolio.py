from typing import Dict

import pandas as pd

from ..config.backtest_config import PairConfig


class PairSymbols:
    """
    Manages a pair of symbols (base and inverse) for market neutral trading.
    Handles position tracking, trade recording, and performance calculations.
    """

    def __init__(self, config: PairConfig):
        self.config = config
        self.base_symbol = config.base_symbol
        self.inverse_symbol = config.inverse_symbol
        self.inverse_leverage = config.inverse_leverage
        self.base_data = config.base_data
        self.inverse_data = config.inverse_data
        self.pair_allocation = config.pair_allocation

        # Initialize tracking structures
        self.positions = {
            self.base_symbol: pd.Series(dtype=float),
            self.inverse_symbol: pd.Series(dtype=float),
        }
        self.trades = pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "size",
                "price",
                "cost",
                "current_position",
                "direction",
                "notional_value",
            ]
        )
        self.position_values = {
            self.base_symbol: pd.Series(dtype=float),
            self.inverse_symbol: pd.Series(dtype=float),
        }
        self._validate_data()

    def _validate_data(self):
        """Validate that required columns exist in price data"""
        required_columns = ["open", "high", "low", "close", "volume"]
        for data in [self.base_data, self.inverse_data]:
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Data must contain columns: {required_columns}")

    def add_trade(self, trade: Dict):
        """Add a trade to the trades DataFrame"""
        self.trades = pd.concat([self.trades, pd.DataFrame([trade])], ignore_index=True)

    def calculate_market_neutral_weights(self) -> Dict[str, float]:
        """Calculate market neutral weights accounting for leverage"""
        inverse_weight = 1 / (1 + self.inverse_leverage)
        base_weight = 1 - inverse_weight

        weights = {self.base_symbol: base_weight, self.inverse_symbol: inverse_weight}
        return weights

    def get_current_positions(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Get current positions for both symbols at given timestamp"""
        positions = {}
        for symbol in [self.base_symbol, self.inverse_symbol]:
            mask = self.positions[symbol].index <= timestamp
            if mask.any():
                positions[symbol] = self.positions[symbol][mask].iloc[-1]
            else:
                positions[symbol] = 0.0
        return positions

    def get_position_values(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Get current position values for both symbols at given timestamp"""
        values = {}
        positions = self.get_current_positions(timestamp)

        for symbol in [self.base_symbol, self.inverse_symbol]:
            data = self.base_data if symbol == self.base_symbol else self.inverse_data
            try:
                price = data.loc[timestamp, "close"]
                values[symbol] = positions[symbol] * price
            except KeyError:
                values[symbol] = 0.0

        return values

    def calculate_daily_pnl(self, timestamp: pd.Timestamp) -> tuple[float, float]:
        """Calculate daily P&L for the pair"""
        gross_pnl = 0.0

        # Get today's trades and costs
        daily_trades = self.trades[
            pd.to_datetime(self.trades["timestamp"]).dt.date == timestamp.date()
        ]
        daily_costs = daily_trades["cost"].sum() if not daily_trades.empty else 0.0

        for symbol in [self.base_symbol, self.inverse_symbol]:
            data = self.base_data if symbol == self.base_symbol else self.inverse_data
            try:
                current_price = data.loc[timestamp, "close"]
                prev_idx = data.index.get_loc(timestamp) - 1

                if prev_idx >= 0:
                    prev_price = data.iloc[prev_idx]["close"]
                    position = self.get_current_positions(timestamp)[symbol]

                    if position != 0:
                        pnl = position * (current_price - prev_price)
                        gross_pnl += pnl

            except (KeyError, IndexError):
                continue

        net_pnl = gross_pnl - daily_costs
        return gross_pnl, net_pnl
