from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from ..config.backtest_config import (
    BacktestConfig,
    PairConfig,
    RebalanceFrequency,
)
from .calendar import TradingCalendar
from .portfolio import PairSymbols
from ..utils.trading_utils import calculate_slippage


class MarketNeutralBacktester:
    """
    Core backtesting engine for market neutral strategies.
    Handles position management, trade execution, and rebalancing.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.pairs: Dict[str, PairSymbols] = {}
        self.portfolio_value = []
        self.trades = []
        self.results = None
        self.daily_gross_returns = pd.Series(dtype=float)
        self.daily_net_returns = pd.Series(dtype=float)
        self.daily_exposures = pd.Series(dtype=float)
        self.calendar = TradingCalendar()

    def add_pair(self, pair_config: PairConfig, data: Dict[str, pd.DataFrame]):
        """Add a trading pair to the backtester"""
        pair_key = f"{pair_config.base_symbol}-{pair_config.inverse_symbol}"

        # Extract data for each symbol
        base_data = data[pair_config.base_symbol]
        inverse_data = data[pair_config.inverse_symbol]

        # Create pair configuration with data
        full_config = PairConfig(
            base_symbol=pair_config.base_symbol,
            inverse_symbol=pair_config.inverse_symbol,
            inverse_leverage=pair_config.inverse_leverage,
            base_data=base_data,
            inverse_data=inverse_data,
            pair_allocation=pair_config.pair_allocation,
        )

        self.pairs[pair_key] = PairSymbols(full_config)

    def backtest(self) -> pd.DataFrame:
        """Run the backtest"""
        results = []
        gross_portfolio_value = self.config.initial_capital
        net_portfolio_value = self.config.initial_capital

        # Get common timestamp index across all pairs
        all_timestamps = set()
        for pair in self.pairs.values():
            all_timestamps.update(pair.base_data.index)
            all_timestamps.update(pair.inverse_data.index)
        timestamps = pd.DatetimeIndex(sorted(all_timestamps))

        # Initialize positions at first valid business day
        first_timestamp = timestamps[0]
        while not self.calendar.is_business_day(first_timestamp):
            first_timestamp = self.calendar.next_business_day(first_timestamp)

        # Initial rebalance
        initial_weights = self._calculate_target_weights(gross_portfolio_value)
        self._execute_selected_rebalance(
            first_timestamp, gross_portfolio_value, initial_weights, list(self.pairs.keys())
        )
        last_rebalance_date = first_timestamp
        next_rebalance_date = self._find_next_rebalance_date(last_rebalance_date)

        # Track rebalance counts per pair
        rebalance_counts = {pair_key: 1 for pair_key in self.pairs.keys()}

        # Initialize costs tracking
        self.cumulative_costs = pd.Series(np.nan, index=timestamps)

        # Main backtest loop
        for timestamp in timestamps:
            if not self.calendar.is_business_day(timestamp):
                continue

            # Calculate daily metrics
            gross_pnl, net_pnl, gross_return, net_return = self._calculate_daily_metrics(
                timestamp, gross_portfolio_value, net_portfolio_value
            )

            # Update portfolio values
            gross_portfolio_value += gross_pnl
            net_portfolio_value += net_pnl

            # Store returns
            self.daily_gross_returns[timestamp] = gross_return
            self.daily_net_returns[timestamp] = net_return

            # Check if rebalance is needed
            if timestamp.date() >= next_rebalance_date.date():
                current_weights = self._calculate_current_weights(timestamp, gross_portfolio_value)
                target_weights = self._calculate_target_weights(gross_portfolio_value)

                # Find pairs that need rebalancing
                pairs_to_rebalance = []
                for pair_key in self.pairs:
                    if self._check_pair_weight_deviation(
                        pair_key, current_weights, target_weights
                    ):
                        pairs_to_rebalance.append(pair_key)
                        rebalance_counts[pair_key] += 1

                if pairs_to_rebalance:
                    self._execute_selected_rebalance(
                        timestamp, gross_portfolio_value, target_weights, pairs_to_rebalance
                    )

                last_rebalance_date = timestamp
                next_rebalance_date = self._find_next_rebalance_date(last_rebalance_date)

            # Calculate exposures
            exposures = self._calculate_position_exposures(timestamp)

            # Calculate daily costs
            daily_costs = self._calculate_daily_costs(timestamp)
            # Update cumulative costs
            self.cumulative_costs[timestamp] = (
                self.cumulative_costs.get(timestamp - pd.Timedelta(days=1), 0) + daily_costs
            )

            # Record results
            result = {
                "timestamp": timestamp,
                "gross_portfolio_value": gross_portfolio_value,
                "net_portfolio_value": net_portfolio_value,
                "gross_daily_pnl": gross_pnl,
                "net_daily_pnl": net_pnl,
                "gross_daily_return": gross_return,
                "net_daily_return": net_return,
                "daily_costs": daily_costs,
                "cost_drag": (gross_return - net_return) if gross_return is not None else 0,
                "total_exposure": sum(abs(exp) for exp in exposures.values()),
            }

            # Add pair-specific metrics
            # Add individual pair positions and metrics
            for pair_key, pair in self.pairs.items():
                for symbol in [pair.base_symbol, pair.inverse_symbol]:
                    result[f"{symbol}_position"] = pair.positions[symbol].get(timestamp, 0)
                    result[f"{symbol}_value"] = pair.position_values[symbol].get(timestamp, 0)

                # Store both gross and net metrics for pairs
                result[f"{pair_key}_gross_pnl"] = getattr(
                    pair, "daily_gross_pnl", pd.Series()
                ).get(timestamp, 0)
                result[f"{pair_key}_net_pnl"] = getattr(pair, "daily_net_pnl", pd.Series()).get(
                    timestamp, 0
                )
                result[f"{pair_key}_gross_return"] = getattr(
                    pair, "daily_gross_returns", pd.Series()
                ).get(timestamp, 0)
                result[f"{pair_key}_net_return"] = getattr(
                    pair, "daily_net_returns", pd.Series()
                ).get(timestamp, 0)
                result[f"{pair_key}_exposure"] = exposures[pair_key]
                result[f"{pair_key}_rebalance_count"] = rebalance_counts[pair_key]

            results.append(result)

        self.results = pd.DataFrame(results).set_index("timestamp")
        return self.results

    def _calculate_target_weights(self, portfolio_value: float) -> Dict[str, Dict[str, float]]:
        """Calculate target weights for each pair"""
        target_weights = {}

        for pair_key, pair in self.pairs.items():
            pair_allocation = pair.pair_allocation
            intra_pair_weights = pair.calculate_market_neutral_weights()

            pair_targets = {}
            for symbol, intra_weight in intra_pair_weights.items():
                final_weight = intra_weight * pair_allocation
                pair_targets[symbol] = final_weight

            target_weights[pair_key] = pair_targets

        return target_weights

    def _execute_trade(
        self,
        pair: PairSymbols,
        symbol: str,
        timestamp: pd.Timestamp,
        current_position: float,
        target_position: float,
    ):
        """Execute a single trade with slippage and transaction costs"""
        data = pair.base_data if symbol == pair.base_symbol else pair.inverse_data
        current_price = data.loc[timestamp, "close"]
        trade_size = target_position - current_position

        if abs(trade_size) < 1:
            return

        # Apply slippage and calculate costs
        direction = np.sign(trade_size)
        execution_price, transaction_cost = self._apply_slippage_and_costs(
            trade_size, current_price, direction
        )
        trade_value = trade_size * execution_price

        # Initialize tracking structures if needed
        if not hasattr(pair, "first_trade_date"):
            pair.first_trade_date = {}
        if symbol not in pair.first_trade_date:
            pair.first_trade_date[symbol] = timestamp.date()

        is_first_trade = (
            symbol not in pair.first_trade_date
            or pair.first_trade_date[symbol] == timestamp.date()
        )

        # Initialize position tracking series
        if symbol not in pair.positions:
            pair.positions[symbol] = pd.Series(dtype=float)
        if symbol not in pair.position_values:
            pair.position_values[symbol] = pd.Series(dtype=float)

        # Calculate PnL components
        try:
            if is_first_trade:
                # First trade: No P&L, just initialization
                existing_position_pnl = 0
                trade_pnl = 0
                prev_price = execution_price
            else:
                # Get previous price for P&L calculation
                prev_idx = data.index.get_loc(timestamp) - 1
                if prev_idx >= 0:
                    prev_price = data.iloc[prev_idx]["close"]

                    # Calculate existing position P&L
                    if current_position != 0:
                        existing_position_pnl = current_position * (current_price - prev_price)
                    else:
                        existing_position_pnl = 0

                    # Calculate trade P&L
                    if trade_size != 0:
                        if (current_position > 0 > trade_size) or (
                            current_position < 0 < trade_size
                        ):
                            # Closing or reducing position
                            closed_size = min(abs(trade_size), abs(current_position)) * np.sign(
                                current_position
                            )
                            trade_pnl = closed_size * (current_price - prev_price)
                        else:
                            # Opening or increasing position
                            trade_pnl = trade_size * (current_price - execution_price)
                else:
                    existing_position_pnl = 0
                    trade_pnl = 0
                    prev_price = execution_price

        except (KeyError, IndexError) as e:
            print(f"Error calculating P&L for {symbol}: {e}")
            existing_position_pnl = 0
            trade_pnl = 0
            prev_price = execution_price

        # Calculate final P&L
        gross_pnl = existing_position_pnl + trade_pnl
        net_pnl = gross_pnl - transaction_cost
        assert (
            net_pnl <= gross_pnl
        ), "net pnl should be less than or equal to gross pnl always in this backtest"

        # Create trade record
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "size": trade_size,
            "price": execution_price,
            "prev_price": prev_price,
            "cost": transaction_cost,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "current_position": target_position,
            "direction": direction,
            "prev_position": current_position,
            "notional_value": abs(trade_value),
            "is_first_trade": is_first_trade,
        }

        # Store trade
        if not isinstance(pair.trades, pd.DataFrame):
            pair.trades = pd.DataFrame(columns=list(trade.keys()))
        pair.trades = pd.concat([pair.trades, pd.DataFrame([trade])], ignore_index=True)

        # Update master trade list
        self.trades.append(trade)

        # Update position and value tracking
        pair.positions[symbol][timestamp] = target_position
        pair.position_values[symbol][timestamp] = target_position * current_price

    def _apply_slippage_and_costs(
        self, trade_size: float, price: float, direction: int
    ) -> tuple[float, float]:
        """Apply slippage to price and calculate transaction costs"""
        slipped_price, _ = calculate_slippage(
            price,
            trade_size,
            direction,
            market_depth=10_000_000_000,
            max_slippage_percentage=0.001,
        )

        # Calculate costs
        slippage_cost = abs(price - slipped_price) * abs(trade_size)
        commission_costs = abs(trade_size * price * self.config.commission_costs)
        total_costs = slippage_cost + commission_costs

        return slipped_price, total_costs

    def get_backtest_data(self) -> Dict[str, pd.DataFrame]:
        """Get comprehensive backtest data for analysis"""
        # Combine all trades into a single DataFrame
        all_trades = []
        for pair_key, pair in self.pairs.items():
            if not pair.trades.empty:
                pair_trades = pair.trades.copy()
                pair_trades["pair"] = pair_key
                all_trades.append(pair_trades)

        trades_df = pd.concat(all_trades) if all_trades else pd.DataFrame()

        # Create pair metrics
        pair_metrics = {}
        for pair_key, pair in self.pairs.items():
            pair_metrics[pair_key] = {
                "positions": pd.DataFrame(pair.positions),
                "position_values": pd.DataFrame(pair.position_values),
                "trades": pair.trades,
                "daily_gross_returns": getattr(pair, "daily_gross_returns", pd.Series()),
                "daily_net_returns": getattr(pair, "daily_net_returns", pd.Series()),
                "daily_gross_pnl": getattr(pair, "daily_gross_pnl", pd.Series()),
                "daily_net_pnl": getattr(pair, "daily_net_pnl", pd.Series()),
            }

        return {
            "portfolio_metrics": self.results,
            "daily_gross_returns": self.daily_gross_returns,
            "daily_net_returns": self.daily_net_returns,
            "trade_history": trades_df,
            "pair_metrics": pair_metrics,
            "costs": {
                "daily": self.results["daily_costs"],
                "cumulative": self.cumulative_costs,
            },
        }

    # Include the following helper methods (implementation details removed for brevity):
    def _find_next_rebalance_date(self, current_date: pd.Timestamp) -> pd.Timestamp:
        """
        Find the next valid rebalance date based on frequency

        Parameters:
        -----------
        current_date: pd.Timestamp
            Current date

        Returns:
        --------
        pd.Timestamp: Next valid rebalance date
        """
        if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            next_date = self.calendar.next_business_day(current_date)
            print(f"Next daily rebalance: {next_date}")
            return next_date

        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            # Start from next day
            next_date = current_date + pd.Timedelta(days=1)

            # Find next Monday
            while next_date.weekday() != 0:  # 0 = Monday
                next_date += pd.Timedelta(days=1)

            # Ensure it's a business day
            while not self.calendar.is_business_day(next_date):
                next_date += pd.Timedelta(days=1)

            print(f"Next weekly rebalance: {next_date}")
            return next_date

        elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            next_date = self.calendar.first_business_day_of_next_month(current_date)
            print(f"Next monthly rebalance: {next_date}")
            return next_date

        return current_date

    def _validate_rebalance_date(
        self, date: pd.Timestamp, last_rebalance: Optional[pd.Timestamp]
    ) -> bool:
        """Validate if rebalance should occur on given date"""
        pass

    def _calculate_position_exposures(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Calculate current position exposures for risk monitoring"""
        exposures = {}
        for pair_key, pair in self.pairs.items():
            pair_exposure = 0
            for symbol in [pair.base_symbol, pair.inverse_symbol]:
                position_qty = pair.positions[symbol].get(timestamp, 0)
                data = pair.base_data if symbol == pair.base_symbol else pair.inverse_data
                price = data.loc[timestamp, "close"]

                exposure = position_qty * price

                pair_exposure += abs(exposure)
                exposures[symbol] = exposure

            exposures[pair_key] = pair_exposure

        return exposures

    def _calculate_current_weights(
        self, timestamp: pd.Timestamp, portfolio_value: float
    ) -> Dict[str, Dict[str, float]]:
        """Calculate current weights with consistent scaling using pair values"""
        weights = {}
        for pair_key, pair in self.pairs.items():
            pair_weights = {}

            # First calculate total pair value for proper scaling
            pair_value = 0
            symbol_values = {}  # Store individual symbol values for later use

            for symbol in [pair.base_symbol, pair.inverse_symbol]:
                positions = pair.positions[symbol]
                if len(positions) > 0:
                    mask = positions.index <= timestamp
                    if mask.any():
                        last_position_qty = positions[mask].iloc[-1]
                        data = pair.base_data if symbol == pair.base_symbol else pair.inverse_data
                        try:
                            price = data.loc[timestamp, "close"]
                            symbol_value = last_position_qty * price
                            pair_value += abs(symbol_value)
                            symbol_values[symbol] = symbol_value
                        except KeyError:
                            symbol_values[symbol] = 0
                    else:
                        symbol_values[symbol] = 0
                else:
                    symbol_values[symbol] = 0

            # Now calculate weights using pair value for proper scaling
            for symbol in [pair.base_symbol, pair.inverse_symbol]:
                if pair_value > 0:
                    # Calculate weight as proportion of total portfolio scaled by pair allocation
                    symbol_weight = (
                        (symbol_values[symbol] / portfolio_value) if portfolio_value > 0 else 0
                    )
                else:
                    symbol_weight = 0.0

                pair_weights[symbol] = symbol_weight

                # Debug output
                # print(f"\n{pair_key} - {symbol} weight calculation:")
                # print(f"Position value: ${symbol_values[symbol]:,.2f}")
                # print(f"Pair value: ${pair_value:,.2f}")
                # print(f"Portfolio value: ${portfolio_value:,.2f}")
                # print(f"Calculated weight: {symbol_weight:.4f}")

            weights[pair_key] = pair_weights

        return weights

    def _check_pair_weight_deviation(
        self,
        pair_key: str,
        current_weights: Dict[str, Dict[str, float]],
        target_weights: Dict[str, Dict[str, float]],
    ) -> bool:
        """Check weight deviation for a single pair with explicit debug output"""
        if pair_key not in current_weights or pair_key not in target_weights:
            return False

        pair_current = current_weights[pair_key]
        pair_target = target_weights[pair_key]

        max_deviation = 0.0
        deviations = {}

        for symbol in pair_current:
            current = pair_current[symbol]
            target = pair_target[symbol]
            deviation = abs(current - target)
            deviations[symbol] = deviation
            max_deviation = max(max_deviation, deviation)

        # Detailed debug output for all checks
        # print(f"\nWeight check for {pair_key}:")
        # for symbol in pair_current:
        #     print(f"{symbol}:")
        #     print(f"  Current weight: {pair_current[symbol]:.4f}")
        #     print(f"  Target weight:  {pair_target[symbol]:.4f}")
        #     print(f"  Deviation:      {deviations[symbol]:.4f}")
        # print(f"Max deviation: {max_deviation:.4f}")
        # print(f"Tolerance: {self.config.weight_tolerance:.4f}")
        # print(f"Need rebalance: {max_deviation > self.config.weight_tolerance}")

        return max_deviation > self.config.weight_tolerance

    def _calculate_daily_metrics(
        self, timestamp: pd.Timestamp, gross_portfolio_value: float, net_portfolio_value: float
    ) -> tuple:
        """Calculate daily PnL and returns for all pairs, properly handling first day initialization"""
        print(f"\n{'=' * 80}")
        print(f"Calculating daily metrics for {timestamp}")
        print(f"Starting gross portfolio value: ${gross_portfolio_value:,.2f}")
        print(f"Starting net portfolio value: ${net_portfolio_value:,.2f}")
        print(f"{'=' * 80}")

        total_gross_pnl = 0.0
        total_net_pnl = 0.0
        total_value = 0.0

        for pair_key, pair in self.pairs.items():
            print(f"\nProcessing pair: {pair_key}")
            pair_gross_pnl = 0.0
            pair_net_pnl = 0.0
            pair_value = 0.0

            # Check if this is the first day for this pair
            if isinstance(pair.trades, pd.DataFrame) and not pair.trades.empty:
                pair_trades = pair.trades.sort_values("timestamp")
                is_first_day = (
                    pd.to_datetime(pair_trades.iloc[0]["timestamp"]).date() == timestamp.date()
                )
                prev_day = (timestamp - pd.Timedelta(days=1)).date()
            else:
                pair_trades = pd.DataFrame()
                is_first_day = False
                prev_day = None

            print(f"Is first trading day: {is_first_day}")

            for symbol in [pair.base_symbol, pair.inverse_symbol]:
                positions = pair.positions[symbol]
                if len(positions) == 0:
                    print(f"No positions for {symbol}")
                    continue

                mask = positions.index <= timestamp  # Changed to include current timestamp
                if not mask.any():
                    print(f"No positions for {symbol}")
                    continue

                position_qty = positions[mask].iloc[-1]
                if position_qty == 0:
                    print(f"Zero position for {symbol}")
                    continue

                data = pair.base_data if symbol == pair.base_symbol else pair.inverse_data
                try:
                    current_price = data.loc[timestamp, "close"]

                    # Handle price reference and position value calculation
                    if is_first_day:
                        if not pair_trades.empty:
                            symbol_trades: pd.DataFrame = pair_trades[
                                pair_trades["symbol"] == symbol
                            ]
                            if not symbol_trades.empty:
                                first_trade: pd.Series = symbol_trades.iloc[0]
                                prev_price = first_trade["price"]  # Use execution price
                                print(
                                    f"{symbol} first day - using execution price: ${prev_price:.2f}"
                                )

                                # Calculate position value using execution price
                                position_value = abs(position_qty * prev_price)
                                position_pnl = position_qty * (current_price - prev_price)

                                # Get first day costs
                                day_costs = first_trade["cost"]
                            else:
                                continue
                        else:
                            continue
                    else:
                        prev_idx = data.index.get_loc(timestamp) - 1
                        if prev_idx >= 0:
                            prev_price = data.iloc[prev_idx]["close"]
                            print(f"{symbol} using previous close: ${prev_price:.2f}")

                            # Calculate standard position value and P&L
                            position_value = abs(position_qty * prev_price)
                            position_pnl = position_qty * (current_price - prev_price)

                            # Get previous day's costs
                            if not pair_trades.empty and prev_day is not None:
                                day_trades = pair_trades[
                                    (pair_trades["symbol"] == symbol)
                                    & (
                                        pd.to_datetime(pair_trades["timestamp"]).dt.date
                                        == prev_day
                                    )
                                ]
                                day_costs = day_trades["cost"].sum() if not day_trades.empty else 0
                            else:
                                day_costs = 0
                        else:
                            # Use execution price if no previous close available
                            if not pair_trades.empty:
                                symbol_trades = pair_trades[pair_trades["symbol"] == symbol]
                                if not symbol_trades.empty:
                                    latest_trade = symbol_trades.iloc[-1]
                                    prev_price = latest_trade["price"]
                                    position_value = abs(position_qty * prev_price)
                                    position_pnl = position_qty * (current_price - prev_price)
                                    day_costs = latest_trade["cost"]
                                else:
                                    continue
                            else:
                                continue

                    # Store current position value
                    if symbol not in pair.position_values:
                        pair.position_values[symbol] = pd.Series(dtype=float)
                    pair.position_values[symbol][timestamp] = position_qty * current_price

                    # Update pair metrics
                    pair_gross_pnl += position_pnl
                    pair_net_pnl += position_pnl - day_costs
                    pair_value += position_value

                    print(f"\n{symbol} calculations:")
                    print(f"Position: {position_qty:,.0f}")
                    print(f"Previous price: ${prev_price:.2f}")
                    print(f"Current price: ${current_price:.2f}")
                    print(f"Position value: ${position_value:,.2f}")
                    print(f"Position P&L: ${position_pnl:,.2f}")
                    print(f"Day's trading costs: ${day_costs:,.2f}")

                except (KeyError, IndexError) as error:
                    print(f"Error processing {symbol}: {error}")
                    continue

            # Calculate and store pair metrics
            if pair_value > 0:
                pair_gross_return = pair_gross_pnl / pair_value
                pair_net_return = pair_net_pnl / pair_value

                # Initialize series if needed
                if not hasattr(pair, "daily_gross_returns"):
                    pair.daily_gross_returns = pd.Series(dtype=float)
                if not hasattr(pair, "daily_net_returns"):
                    pair.daily_net_returns = pd.Series(dtype=float)
                if not hasattr(pair, "daily_gross_pnl"):
                    pair.daily_gross_pnl = pd.Series(dtype=float)
                if not hasattr(pair, "daily_net_pnl"):
                    pair.daily_net_pnl = pd.Series(dtype=float)

                # Store metrics
                pair.daily_gross_returns[timestamp] = pair_gross_return
                pair.daily_net_returns[timestamp] = pair_net_return
                pair.daily_gross_pnl[timestamp] = pair_gross_pnl
                pair.daily_net_pnl[timestamp] = pair_net_pnl

                total_gross_pnl += pair_gross_pnl
                total_net_pnl += pair_net_pnl
                total_value += pair_value

                print(f"\n{pair_key} summary:")
                print(f"Pair gross P&L: ${pair_gross_pnl:,.2f}")
                print(f"Pair net P&L: ${pair_net_pnl:,.2f}")
                print(f"Pair value: ${pair_value:,.2f}")
                print(f"Pair gross return: {pair_gross_return:.4%}")
                print(f"Pair net return: {pair_net_return:.4%}")

        # Calculate portfolio returns
        gross_return = (
            total_gross_pnl / gross_portfolio_value if gross_portfolio_value > 0 else 0.0
        )
        net_return = total_net_pnl / gross_portfolio_value if gross_portfolio_value > 0 else 0.0

        print(f"\nPortfolio summary for {timestamp}:")
        print(f"Total gross P&L: ${total_gross_pnl:,.2f}")
        print(f"Total net P&L: ${total_net_pnl:,.2f}")
        print(f"Total value: ${total_value:,.2f}")
        print(f"Portfolio gross return: {gross_return:.4%}")
        print(f"Portfolio net return: {net_return:.4%}")
        print(f"{'=' * 80}\n")

        return total_gross_pnl, total_net_pnl, gross_return, net_return

    def _execute_selected_rebalance(
        self,
        timestamp: pd.Timestamp,
        portfolio_value: float,
        target_weights: Dict[str, Dict[str, float]],
        pairs_to_rebalance: List[str],
    ):
        """Execute rebalancing trades with enhanced logging"""
        print(f"\nExecuting selective rebalance at {timestamp}")
        print(f"Portfolio value: ${portfolio_value:,.2f}")
        print(f"Rebalancing pairs: {pairs_to_rebalance}")

        for pair_key in pairs_to_rebalance:
            if pair_key not in self.pairs:
                continue

            pair = self.pairs[pair_key]
            pair_target = target_weights[pair_key]

            # Calculate current pair value
            pair_value = 0
            current_positions = {}

            for symbol in [pair.base_symbol, pair.inverse_symbol]:
                positions = pair.positions[symbol]
                current_position = 0
                if len(positions) > 0:
                    mask = positions.index < timestamp
                    if mask.any():
                        current_position = positions[mask].iloc[-1]

                current_positions[symbol] = current_position
                data = pair.base_data if symbol == pair.base_symbol else pair.inverse_data
                price = data.loc[timestamp, "close"]
                pair_value += abs(current_position * price)

            print(f"\nRebalancing {pair_key}")
            print(f"Current pair value: ${pair_value:,.2f}")
            print(f"Target pair allocation: {pair.pair_allocation:.4f}")

            for symbol in [pair.base_symbol, pair.inverse_symbol]:
                data = pair.base_data if symbol == pair.base_symbol else pair.inverse_data
                price = data.loc[timestamp, "close"]
                current_position = current_positions[symbol]

                # Calculate target position using portfolio value
                target_value = pair_target[symbol] * portfolio_value
                target_position = target_value / price

                print(f"\n{symbol}:")
                print(f"  Current position: {current_position:,.0f}")
                print(f"  Target position: {target_position:,.0f}")
                print(f"  Price: ${price:.2f}")
                print(f"  Current value: ${current_position * price:,.2f}")
                print(f"  Target value: ${target_value:,.2f}")

                if abs(current_position - target_position) > 1:  # Avoid tiny rebalances
                    self._execute_trade(pair, symbol, timestamp, current_position, target_position)
                    print(
                        f"  Executed rebalance trade: {current_position:,.0f} -> {target_position:,.0f}"
                    )

    def _calculate_daily_costs(self, timestamp: pd.Timestamp) -> float:
        """Calculate daily transaction costs"""
        daily_costs = 0
        for pair in self.pairs.values():
            day_trades = [
                t
                for i, t in pair.trades.iterrows()
                if pd.Timestamp(t["timestamp"]).date() == timestamp.date()
            ]
            daily_costs += sum(t["cost"] for t in day_trades)
        return daily_costs
