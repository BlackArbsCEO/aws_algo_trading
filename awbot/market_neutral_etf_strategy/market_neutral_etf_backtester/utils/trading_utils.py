from typing import Tuple

import numpy as np


def calculate_slippage(
    price: float,
    quantity: float,
    direction: int,  # 1 for buy, -1 for sell
    market_depth: float = 10_000_000,
    max_slippage_percentage: float = 0.001,
) -> Tuple[float, float]:
    """
    Calculate price slippage based on trade size and direction.
    Uses a simple square-root model for market impact.

    Parameters:
        price: Current market price
        quantity: Trade quantity (absolute value)
        direction: Trade direction (1 for buy, -1 for sell)
        market_depth: Market depth parameter for scaling impact
        max_slippage_percentage: Maximum allowed slippage as decimal

    Returns:
        Tuple[float, float]: (adjusted_price, slippage_percentage)
    """
    # Ensure valid direction
    if direction not in [-1, 1]:
        raise ValueError("Direction must be 1 (buy) or -1 (sell)")

    # Calculate base slippage using square root model
    dollar_volume = abs(quantity * price)
    base_slippage = np.sqrt(dollar_volume / market_depth)

    # Apply direction and cap at maximum
    slippage_percentage = min(base_slippage, max_slippage_percentage)
    adjusted_price = price * (1 + direction * slippage_percentage)

    return adjusted_price, slippage_percentage


def calculate_transaction_costs(
    price: float,
    quantity: float,
    commission_rate: float = 0.0,
    min_commission: float = 0.0,
    max_commission: float = float("inf"),
) -> float:
    """
    Calculate total transaction costs including commissions.

    Parameters:
        price: Execution price
        quantity: Trade quantity
        commission_rate: Commission rate as decimal
        min_commission: Minimum commission per trade
        max_commission: Maximum commission per trade

    Returns:
        float: Total transaction cost
    """
    notional_value = abs(price * quantity)
    base_commission = notional_value * commission_rate
    commission = np.clip(base_commission, min_commission, max_commission)

    return commission
