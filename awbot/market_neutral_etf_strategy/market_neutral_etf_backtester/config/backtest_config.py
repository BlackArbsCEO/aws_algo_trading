from dataclasses import dataclass
from enum import Enum

import pandas as pd


@dataclass
class PairConfig:
    base_symbol: str
    inverse_symbol: str
    inverse_leverage: float  # e.g., 3.0 for 3x leverage
    base_data: pd.DataFrame
    inverse_data: pd.DataFrame
    pair_allocation: float


class RebalanceFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000
    slippage: float = 0.0002
    commission_costs: float = 0.002
    margin_requirement: float = 0.2 # not used
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    weight_tolerance: float = 0.025
    pair_allocation: float = None
