from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd


# @dataclass
# class TradeMetrics:
#     """Detailed metrics for trade analysis with gross and net performance"""
#
#     total_trades: int
#     winning_trades: int
#     losing_trades: int
#     win_rate: float
#
#     # Gross performance metrics
#     gross_profit_factor: float
#     gross_avg_win: float
#     gross_avg_loss: float
#     gross_largest_win: float
#     gross_largest_loss: float
#
#     # Net performance metrics
#     net_profit_factor: float
#     net_avg_win: float
#     net_avg_loss: float
#     net_largest_win: float
#     net_largest_loss: float
#
#     # Trading costs and efficiency
#     avg_holding_period: float
#     total_fees: float
#     total_turnover: float
#     annualized_turnover: float
#     avg_trade_size: float
#     daily_turnover_rate: float
#     turnover_ratio: float
#     trade_volume_skew: float
#     rebalance_count: int
#     avg_rebalance_cost: float
#     max_trade_impact: float
#     cost_to_aum_ratio: float
#     cost_drag: float


@dataclass
class TradeMetrics:
    """Detailed metrics for trade analysis with gross and net performance"""

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Gross performance metrics
    gross_profit_factor: float
    gross_avg_win: float
    gross_avg_loss: float
    gross_largest_win: float
    gross_largest_loss: float

    # Net performance metrics
    net_profit_factor: float
    net_avg_win: float
    net_avg_loss: float
    net_largest_win: float
    net_largest_loss: float

    # Trading costs and efficiency
    total_fees: float
    total_turnover: float
    annualized_turnover: float
    avg_trade_size: float
    daily_turnover_rate: float
    turnover_ratio: float
    trade_volume_skew: float
    rebalance_count: int
    avg_rebalance_cost: float
    max_trade_impact: float
    cost_to_aum_ratio: float
    cost_drag: float

    # Portfolio-level metrics
    portfolio_turnover: float
    total_rebalances: int
    avg_rebalances_per_year: float
    weighted_avg_holding_period: float


@dataclass
class RiskMetrics:
    """Risk metrics with gross and net calculations"""

    # Volatility metrics
    gross_volatility: float
    net_volatility: float

    # Risk-adjusted returns
    gross_sharpe_ratio: float
    net_sharpe_ratio: float
    gross_sortino_ratio: float
    net_sortino_ratio: float
    gross_calmar_ratio: float
    net_calmar_ratio: float

    # Drawdown metrics
    gross_max_drawdown: float
    net_max_drawdown: float
    gross_max_drawdown_duration: int
    net_max_drawdown_duration: int

    # Value at Risk
    gross_var_95: float
    net_var_95: float
    gross_cvar_95: float
    net_cvar_95: float

    # Market metrics
    alpha: Optional[float]
    beta: Optional[float]
    correlation: Optional[float]

    # Cost impact
    cost_var_95: float


@dataclass
class ReturnMetrics:
    """Return metrics with gross and net calculations"""

    # Total returns
    gross_total_return: float
    net_total_return: float
    gross_annualized_return: float
    net_annualized_return: float

    # Monthly statistics
    gross_monthly_returns: pd.Series
    net_monthly_returns: pd.Series
    gross_best_month: float
    net_best_month: float
    gross_worst_month: float
    net_worst_month: float
    gross_positive_months: float
    net_positive_months: float
    gross_avg_monthly_return: float
    net_avg_monthly_return: float
    gross_monthly_volatility: float
    net_monthly_volatility: float

    # Cost impact
    cost_return_impact: float
    avg_monthly_costs: float
    cost_return_ratio: float


def calculate_trade_metrics(
    trades_df: pd.DataFrame, portfolio_df: pd.DataFrame, costs_df: pd.DataFrame
) -> TradeMetrics:
    """Calculate comprehensive trade metrics from trade history"""
    if trades_df.empty:
        return TradeMetrics(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )

    # Basic trade statistics
    gross_trades = trades_df[trades_df["gross_pnl"] != 0]
    winning_trades = len(trades_df[trades_df["net_pnl"] > 0])
    losing_trades = len(trades_df[trades_df["net_pnl"] < 0])

    # Calculate profit factors
    gross_winners = trades_df[trades_df["gross_pnl"] > 0]["gross_pnl"]
    gross_losers = trades_df[trades_df["gross_pnl"] < 0]["gross_pnl"]
    net_winners = trades_df[trades_df["net_pnl"] > 0]["net_pnl"]
    net_losers = trades_df[trades_df["net_pnl"] < 0]["net_pnl"]

    # Calculate turnover metrics
    turnover_metrics = calculate_turnover_metrics(trades_df, portfolio_df)

    # Calculate cost metrics
    cost_metrics = calculate_cost_metrics(costs_df, portfolio_df["gross_portfolio_value"])

    return TradeMetrics(
        total_trades=len(gross_trades),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=winning_trades / len(gross_trades) if len(gross_trades) > 0 else 0,
        gross_profit_factor=(
            abs(gross_winners.sum() / gross_losers.sum())
            if len(gross_losers) > 0
            else float("inf")
        ),
        net_profit_factor=(
            abs(net_winners.sum() / net_losers.sum()) if len(net_losers) > 0 else float("inf")
        ),
        gross_avg_win=gross_winners.mean() if len(gross_winners) > 0 else 0,
        gross_avg_loss=gross_losers.mean() if len(gross_losers) > 0 else 0,
        gross_largest_win=gross_winners.max() if len(gross_winners) > 0 else 0,
        gross_largest_loss=gross_losers.min() if len(gross_losers) > 0 else 0,
        net_avg_win=net_winners.mean() if len(net_winners) > 0 else 0,
        net_avg_loss=net_losers.mean() if len(net_losers) > 0 else 0,
        net_largest_win=net_winners.max() if len(net_winners) > 0 else 0,
        net_largest_loss=net_losers.min() if len(net_losers) > 0 else 0,
        trade_volume_skew=(
            trades_df["notional_value"].skew() if "notional_value" in trades_df.columns else 0
        ),
        rebalance_count=len(trades_df.groupby("timestamp")),
        **turnover_metrics["notional"],
        **turnover_metrics["portfolio"],
        **cost_metrics,
    )


def calculate_risk_metrics(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> RiskMetrics:
    """Calculate comprehensive risk metrics for both gross and net returns"""
    # Annualization factor
    annualization = np.sqrt(252)

    # Calculate volatilities
    gross_vol = gross_returns.std() * annualization
    net_vol = net_returns.std() * annualization

    # Calculate Sharpe ratios
    gross_sharpe = calculate_sharpe_ratio(gross_returns, risk_free_rate)
    net_sharpe = calculate_sharpe_ratio(net_returns, risk_free_rate)

    # Calculate Sortino ratios
    gross_sortino = calculate_sortino_ratio(gross_returns, risk_free_rate)
    net_sortino = calculate_sortino_ratio(net_returns, risk_free_rate)

    # Calculate drawdown metrics
    gross_dd = calculate_drawdown_metrics(gross_returns)
    net_dd = calculate_drawdown_metrics(net_returns)

    # Calculate VaR and CVaR
    gross_var, gross_cvar = calculate_var_cvar(gross_returns)
    net_var, net_cvar = calculate_var_cvar(net_returns)

    # Calculate market metrics if benchmark provided
    alpha, beta, correlation = (None, None, None)
    if benchmark_returns is not None:
        alpha, beta, correlation = calculate_market_metrics(net_returns, benchmark_returns)

    return RiskMetrics(
        gross_volatility=gross_vol,
        net_volatility=net_vol,
        gross_sharpe_ratio=gross_sharpe,
        net_sharpe_ratio=net_sharpe,
        gross_sortino_ratio=gross_sortino,
        net_sortino_ratio=net_sortino,
        gross_calmar_ratio=(
            abs(gross_returns.mean() * 252 / gross_dd["max_drawdown"])
            if gross_dd["max_drawdown"] != 0
            else float("inf")
        ),
        net_calmar_ratio=(
            abs(net_returns.mean() * 252 / net_dd["max_drawdown"])
            if net_dd["max_drawdown"] != 0
            else float("inf")
        ),
        gross_max_drawdown=gross_dd["max_drawdown"],
        net_max_drawdown=net_dd["max_drawdown"],
        gross_max_drawdown_duration=gross_dd["max_duration"],
        net_max_drawdown_duration=net_dd["max_duration"],
        gross_var_95=gross_var,
        net_var_95=net_var,
        gross_cvar_95=gross_cvar,
        net_cvar_95=net_cvar,
        alpha=alpha,
        beta=beta,
        correlation=correlation,
        cost_var_95=calculate_cost_var(gross_returns, net_returns),
    )


def calculate_return_metrics(
    gross_returns: pd.Series, net_returns: pd.Series, costs: pd.Series, trades_df: pd.DataFrame
) -> ReturnMetrics:
    """Calculate comprehensive return metrics"""
    # Calculate monthly returns
    gross_monthly = gross_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    net_monthly = net_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_costs = costs.resample("ME").sum()

    # Calculate total returns
    gross_total = (1 + gross_returns).prod() - 1
    net_total = (1 + net_returns).prod() - 1

    # Calculate annualized returns
    days = len(gross_returns)
    gross_annualized = (1 + gross_total) ** (252 / days) - 1
    net_annualized = (1 + net_total) ** (252 / days) - 1

    return ReturnMetrics(
        gross_total_return=gross_total,
        net_total_return=net_total,
        gross_annualized_return=gross_annualized,
        net_annualized_return=net_annualized,
        gross_monthly_returns=gross_monthly,
        net_monthly_returns=net_monthly,
        gross_best_month=gross_monthly.max(),
        net_best_month=net_monthly.max(),
        gross_worst_month=gross_monthly.min(),
        net_worst_month=net_monthly.min(),
        gross_positive_months=len(gross_monthly[gross_monthly > 0]) / len(gross_monthly),
        net_positive_months=len(net_monthly[net_monthly > 0]) / len(net_monthly),
        gross_avg_monthly_return=gross_monthly.mean(),
        net_avg_monthly_return=net_monthly.mean(),
        gross_monthly_volatility=gross_monthly.std(),
        net_monthly_volatility=net_monthly.std(),
        cost_return_impact=(gross_monthly - net_monthly).mean(),
        avg_monthly_costs=monthly_costs.mean(),
        cost_return_ratio=(
            costs.sum() / abs(trades_df["gross_pnl"].sum())
            if trades_df["gross_pnl"].sum() != 0
            else float("inf")
        ),
    )


# Helper functions for metric calculations
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    if returns.std() == 0:
        return 0.0
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio using downside deviation"""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float("inf")
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    return np.sqrt(252) * excess_returns.mean() / downside_std


def calculate_drawdown_metrics(returns: pd.Series) -> Dict[str, Union[float, int]]:
    """Calculate maximum drawdown and drawdown duration"""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - running_max) / running_max

    max_drawdown = drawdowns.min()
    max_duration = calculate_max_drawdown_duration(drawdowns)

    return {"max_drawdown": max_drawdown, "max_duration": max_duration}


def calculate_max_drawdown_duration(drawdown_series: pd.Series) -> int:
    """
    Calculate maximum drawdown duration in days

    Parameters:
    -----------
    drawdown_series: pd.Series
        Series of drawdown values

    Returns:
    --------
    int: Maximum drawdown duration in days
    """
    if drawdown_series.empty:
        return 0

    # Find drawdown periods
    drawdown_mask = drawdown_series != 0
    drawdown_periods = drawdown_mask.astype(int).diff()

    current_duration = 0
    max_duration = 0

    # Calculate durations
    for val in drawdown_periods:
        if val == 1:  # Start of drawdown
            current_duration = 1
        elif val == 0 and current_duration > 0:  # Continuing drawdown
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:  # End of drawdown
            current_duration = 0

    # Check if we're still in a drawdown at the end of the series
    if current_duration > 0:
        max_duration = max(max_duration, current_duration)

    return max_duration


def calculate_var_cvar(returns: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Value at Risk and Conditional Value at Risk"""
    var = returns.quantile(1 - confidence)
    cvar = returns[returns <= var].mean()
    return var, cvar


def calculate_market_metrics(
    returns: pd.Series, benchmark_returns: pd.Series
) -> Tuple[float, float, float]:
    """Calculate alpha, beta, and correlation to benchmark"""
    # Align series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) == 0:
        return 0.0, 0.0, 0.0

    # Calculate beta using covariance method
    covariance = aligned.cov().iloc[0, 1]
    benchmark_variance = aligned.iloc[:, 1].var()
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0

    # Calculate alpha (annualized)
    alpha = (aligned.iloc[:, 0].mean() - beta * aligned.iloc[:, 1].mean()) * 252

    # Calculate correlation
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

    return alpha, beta, correlation


def get_pairs_from_portfolio(portfolio_df: pd.DataFrame) -> list:
    """
    Extract trading pairs from portfolio dataframe column names
    """
    pairs = []
    for col in portfolio_df.columns:
        if col.endswith("_rebalance_count"):
            pairs.append(col.replace("_rebalance_count", ""))
    return pairs


def calculate_portfolio_weights(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate portfolio weights for each instrument pair

    Parameters:
    -----------
    portfolio_df: pd.DataFrame
        Portfolio dataframe with position values and gross portfolio value

    Returns:
    --------
    pd.DataFrame: Portfolio weights over time
    """
    weights_data = []
    pairs = get_pairs_from_portfolio(portfolio_df)

    for pair in pairs:
        # Extract long and short symbols from pair (e.g., 'QQQ-SQQQ' -> ['QQQ', 'SQQQ'])
        long_symbol, short_symbol = pair.split("-")

        # Calculate weights for both legs
        for symbol in [long_symbol, short_symbol]:
            weights_data.append(
                {
                    "timestamp": portfolio_df.index,
                    "symbol": symbol,
                    "pair": pair,
                    "weight": portfolio_df[f"{symbol}_value"]
                    / portfolio_df["gross_portfolio_value"],
                    "value": portfolio_df[f"{symbol}_value"],
                    "position": portfolio_df[f"{symbol}_position"],
                }
            )

    # Combine all weights into a single DataFrame
    weights_df = pd.DataFrame(weights_data).explode("timestamp")
    weights_df["timestamp"] = pd.to_datetime(weights_df["timestamp"])

    return weights_df


def get_rebalance_timestamps(portfolio_df: pd.DataFrame) -> Dict[str, pd.DatetimeIndex]:
    """
    Get rebalance timestamps for each pair from the portfolio dataframe
    """
    rebalance_times = {}
    pairs = get_pairs_from_portfolio(portfolio_df)

    for pair in pairs:
        # Get timestamps where rebalance count changes
        rebalance_series = portfolio_df[f"{pair}_rebalance_count"]
        rebalance_timestamps = portfolio_df.index[rebalance_series.diff() > 0]
        rebalance_times[pair] = rebalance_timestamps

    return rebalance_times


def calculate_pair_turnover(
    portfolio_df: pd.DataFrame, pair: str, rebalance_timestamps: pd.DatetimeIndex
) -> Dict[str, float]:
    """
    Calculate turnover metrics for a specific pair
    """
    # Extract symbols from pair
    long_symbol, short_symbol = pair.split("-")

    # Calculate weights at rebalance points
    turnover_by_rebalance = []
    holding_periods = []
    prev_timestamp = None

    for timestamp in sorted(rebalance_timestamps):
        if prev_timestamp is not None:
            # Calculate weight changes for both legs
            for symbol in [long_symbol, short_symbol]:
                curr_weight = (
                    portfolio_df.loc[timestamp, f"{symbol}_value"]
                    / portfolio_df.loc[timestamp, "gross_portfolio_value"]
                )
                prev_weight = (
                    portfolio_df.loc[prev_timestamp, f"{symbol}_value"]
                    / portfolio_df.loc[prev_timestamp, "gross_portfolio_value"]
                )

                weight_change = abs(curr_weight - prev_weight)
                turnover_by_rebalance.append(
                    {"timestamp": timestamp, "symbol": symbol, "turnover": weight_change}
                )

            # Calculate holding period
            holding_period = (timestamp - prev_timestamp).days
            holding_periods.append(holding_period)

        prev_timestamp = timestamp

    # Calculate metrics
    if turnover_by_rebalance:
        turnover_df = pd.DataFrame(turnover_by_rebalance)
        avg_turnover = turnover_df["turnover"].mean()
        turnover_vol = turnover_df["turnover"].std()

        # Calculate annualized metrics
        total_days = (rebalance_timestamps[-1] - rebalance_timestamps[0]).days
        rebalances_per_year = (
            len(rebalance_timestamps) * (252 / total_days) if total_days > 0 else 0
        )
        annualized_turnover = avg_turnover * rebalances_per_year
    else:
        avg_turnover = turnover_vol = annualized_turnover = rebalances_per_year = 0

    return {
        "avg_turnover_per_rebalance": avg_turnover,
        "turnover_volatility": turnover_vol,
        "annualized_turnover": annualized_turnover,
        "rebalances_per_year": rebalances_per_year,
        "avg_holding_period": np.mean(holding_periods) if holding_periods else 0,
        "total_rebalances": len(rebalance_timestamps),
        "exposure": portfolio_df[f"{pair}_exposure"].mean(),
    }


def calculate_turnover_metrics(
    trades_df: pd.DataFrame, portfolio_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Calculate turnover metrics for all pairs
    """
    # Get all pairs and their rebalance timestamps
    pairs = get_pairs_from_portfolio(portfolio_df)
    rebalance_times = get_rebalance_timestamps(portfolio_df)

    notional_turnover_metrics = calculate_notional_turnover(trades_df, portfolio_df)

    # Calculate metrics for each pair
    pair_metrics = {}
    portfolio_turnover = 0
    total_rebalances = 0

    for pair in pairs:
        pair_metrics[pair] = calculate_pair_turnover(portfolio_df, pair, rebalance_times[pair])

        # Accumulate portfolio-level metrics
        portfolio_turnover += pair_metrics[pair]["annualized_turnover"] * (
            pair_metrics[pair]["exposure"] / portfolio_df["total_exposure"].mean()
        )
        total_rebalances += pair_metrics[pair]["total_rebalances"]

    # Calculate portfolio-level metrics
    portfolio_metrics = {
        "portfolio_turnover": portfolio_turnover,
        "total_rebalances": total_rebalances,
        "avg_rebalances_per_year": sum(
            m["rebalances_per_year"]
            * (pair_metrics[p]["exposure"] / portfolio_df["total_exposure"].mean())
            for p, m in pair_metrics.items()
        ),
        "weighted_avg_holding_period": sum(
            m["avg_holding_period"]
            * (pair_metrics[p]["exposure"] / portfolio_df["total_exposure"].mean())
            for p, m in pair_metrics.items()
        ),
    }

    # Print summary
    print("\nPortfolio Turnover Analysis:")
    print(f"Total portfolio turnover (annualized): {portfolio_metrics['portfolio_turnover']:.2%}")
    print(f"Total rebalances: {portfolio_metrics['total_rebalances']}")
    print(f"Average holding period (days): {portfolio_metrics['weighted_avg_holding_period']:.1f}")

    print("\nPair-Level Analysis:")
    for pair, metrics in pair_metrics.items():
        print(f"\n{pair}:")
        print(f"  Annualized turnover: {metrics['annualized_turnover']:.2%}")
        print(f"  Rebalances per year: {metrics['rebalances_per_year']:.1f}")
        print(f"  Average holding period: {metrics['avg_holding_period']:.1f} days")
        print(f"  Total rebalances: {metrics['total_rebalances']}")

    return {
        "notional": notional_turnover_metrics,
        "portfolio": portfolio_metrics,
        "pairs": pair_metrics,
    }


def calculate_notional_turnover(
    trades_df: pd.DataFrame, portfolio_df: pd.DataFrame
) -> Dict[str, float]:
    """Calculate traditional turnover metrics"""
    if len(trades_df) == 0:
        return {
            "total_turnover": 0.0,
            "annualized_turnover": 0.0,
            "avg_trade_size": 0.0,
            "daily_turnover_rate": 0.0,
            "turnover_ratio": 0.0,
            "max_trade_impact": 0.0,
        }

    # Calculate average portfolio value
    avg_portfolio_value = portfolio_df["gross_portfolio_value"].mean()

    # Calculate turnover using actual trade values
    if "notional_value" not in trades_df.columns:
        trades_df["notional_value"] = abs(trades_df["size"] * trades_df["price"])
    total_turnover = trades_df["notional_value"].sum()

    # Calculate trading days and annualization
    trading_days = len(portfolio_df)
    years = trading_days / 252

    # Group trades by date
    daily_trades = trades_df.groupby(pd.to_datetime(trades_df["timestamp"]).dt.date)
    daily_turnover = daily_trades["notional_value"].sum()
    max_daily_turnover = daily_turnover.max()

    return {
        "total_turnover": total_turnover,
        "annualized_turnover": total_turnover / years if years > 0 else total_turnover,
        "avg_trade_size": trades_df["notional_value"].mean(),
        "daily_turnover_rate": (
            daily_turnover.mean() / avg_portfolio_value if avg_portfolio_value > 0 else 0
        ),
        "turnover_ratio": total_turnover / avg_portfolio_value if avg_portfolio_value > 0 else 0,
        "max_trade_impact": (
            max_daily_turnover / avg_portfolio_value if avg_portfolio_value > 0 else 0
        ),
    }


def calculate_cost_metrics(
    costs_df: pd.DataFrame, portfolio_values: pd.Series
) -> Dict[str, float]:
    """
    Calculate comprehensive cost-related metrics.

    Parameters:
        costs_df: DataFrame containing daily trading costs
        portfolio_values: Series of daily portfolio values

    Returns:
        Dictionary containing cost metrics:
            - total_fees: Total trading costs
            - avg_daily_cost: Average daily trading cost
            - cost_to_aum_ratio: Trading costs as percentage of average AUM
            - cost_drag: Average daily performance impact from costs
            - cost_volatility: Standard deviation of daily costs
            - avg_rebalance_cost: Average cost per rebalance day
            - max_daily_cost: Maximum single-day trading cost
            - cost_to_turnover_ratio: Costs as percentage of turnover
    """
    if costs_df.empty or portfolio_values.empty:
        return {
            "total_fees": 0.0,
            # "avg_daily_cost": 0.0,
            "cost_to_aum_ratio": 0.0,
            "cost_drag": 0.0,
            # "cost_volatility": 0.0,
            "avg_rebalance_cost": 0.0,
            # "max_daily_cost": 0.0,
            # "cost_to_turnover_ratio": 0.0,
        }

    # Align dates between costs and portfolio values
    aligned_data = pd.concat([costs_df, portfolio_values], axis=1).fillna(0)
    daily_costs = aligned_data.iloc[:, 0]  # First column is costs
    daily_values = aligned_data.iloc[:, 1]  # Second column is portfolio values

    # Calculate basic cost metrics
    total_costs = daily_costs.sum()
    trading_days = len(daily_costs[daily_costs > 0])
    avg_portfolio_value = daily_values.mean()

    # Calculate cost ratios
    cost_to_aum = total_costs / avg_portfolio_value if avg_portfolio_value > 0 else 0

    # Calculate daily cost metrics
    daily_cost_pct = daily_costs / daily_values
    avg_daily_cost_pct = daily_cost_pct.mean()
    # cost_volatility = daily_cost_pct.std()

    # Calculate rebalance day metrics
    rebalance_days = daily_costs[daily_costs > 0]
    avg_rebalance_cost = rebalance_days.mean() if len(rebalance_days) > 0 else 0
    max_daily_cost = daily_costs.max()

    # Calculate cost drag
    # Annualize the average daily cost percentage
    cost_drag = avg_daily_cost_pct * 252

    # Get non-zero costs for ratio calculations
    non_zero_costs = daily_costs[daily_costs > 0]

    # Calculate turnover ratio if available
    # if "turnover" in aligned_data.columns:
    #     daily_turnover = aligned_data["turnover"]
    #     turnover_days = daily_turnover[daily_turnover > 0]
    #     cost_to_turnover = (
    #         (non_zero_costs / turnover_days).mean() if not turnover_days.empty else 0.0
    #     )
    # else:
    #     cost_to_turnover = 0.0

    # Print debug information
    print(f"\nCost Metrics Calculation:")
    print(f"Total Costs: ${total_costs:,.2f}")
    print(f"Trading Days: {trading_days}")
    print(f"Average Portfolio Value: ${avg_portfolio_value:,.2f}")
    print(f"Cost to AUM Ratio: {cost_to_aum:.4%}")
    print(f"Average Daily Cost: {avg_daily_cost_pct:.4%}")
    # print(f"Cost Volatility: {cost_volatility:.4%}")
    print(f"Average Rebalance Cost: ${avg_rebalance_cost:,.2f}")
    # print(f"Maximum Daily Cost: ${max_daily_cost:,.2f}")
    print(f"Annualized Cost Drag: {cost_drag:.4%}")
    # print(f"Cost to Turnover Ratio: {cost_to_turnover:.4%}")

    return {
        "total_fees": total_costs,
        "cost_to_aum_ratio": cost_to_aum,
        "cost_drag": cost_drag,
        # "cost_volatility": cost_volatility,
        "avg_rebalance_cost": avg_rebalance_cost,
        # "max_daily_cost": max_daily_cost,
        # "cost_to_turnover_ratio": cost_to_turnover,
    }


def calculate_cost_var(
    gross_returns: pd.Series, net_returns: pd.Series, confidence: float = 0.95
) -> float:
    """Calculate Value at Risk of trading costs"""
    costs = -(gross_returns - net_returns)
    return costs.quantile(confidence)
