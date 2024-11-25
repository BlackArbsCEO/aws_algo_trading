import traceback
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from .visualization import PerformanceVisualizer
from ..analysis.metrics import (
    TradeMetrics,
    RiskMetrics,
    ReturnMetrics,
    calculate_trade_metrics,
    calculate_risk_metrics,
    calculate_return_metrics,
)


class PerformanceAnalyzer:
    """
    Analyzes backtest results using calculated metrics.
    Maintains state and coordinates analysis processes.
    """

    def __init__(
        self,
        backtest_data: Dict[str, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize analyzer with backtest data.

        Parameters:
            backtest_data: Dictionary containing backtest results
            benchmark_returns: Optional benchmark return series
            risk_free_rate: Annual risk-free rate for Sharpe calculations
        """
        # Store input data
        self.portfolio_metrics = backtest_data["portfolio_metrics"]
        self.trade_history = backtest_data["trade_history"]
        self.pair_metrics = backtest_data["pair_metrics"]
        self.costs = backtest_data["portfolio_metrics"]["daily_costs"]
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate

        # Extract return series
        self.gross_returns = self.portfolio_metrics["gross_daily_return"]
        self.net_returns = self.portfolio_metrics["net_daily_return"]

        # Calculate equity curves
        self.gross_equity = self._calculate_equity_curve(self.gross_returns)
        self.net_equity = self._calculate_equity_curve(self.net_returns)

        # Initialize results containers
        self.trade_metrics: Optional[TradeMetrics] = None
        self.risk_metrics: Optional[RiskMetrics] = None
        self.return_metrics: Optional[ReturnMetrics] = None

        # Run initial analysis
        self._run_analysis()

    def _run_analysis(self):
        """Run all analyses and store results"""
        # Calculate trade metrics
        self.trade_metrics = calculate_trade_metrics(
            trades_df=self.trade_history,
            portfolio_df=self.portfolio_metrics,
            costs_df=self.costs,
        )

        # Calculate risk metrics
        self.risk_metrics = calculate_risk_metrics(
            gross_returns=self.gross_returns,
            net_returns=self.net_returns,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=self.risk_free_rate,
        )

        # Calculate return metrics
        self.return_metrics = calculate_return_metrics(
            gross_returns=self.gross_returns,
            net_returns=self.net_returns,
            costs=self.costs,
            trades_df=self.trade_history,
        )
        # Add benchmark comparison if available
        if self.benchmark_returns is not None:
            self.benchmark_analysis = _analyze_benchmark_comparison(
                strategy_returns=self.portfolio_metrics["net_daily_return"],
                benchmark_returns=self.benchmark_returns,
            )

    def analyze_pair(
        self, pair_key: str
    ) -> Dict[str, Union[TradeMetrics, RiskMetrics, ReturnMetrics]]:
        """
        Analyze performance for a specific pair.

        Parameters:
            pair_key: Identifier for the pair to analyze

        Returns:
            Dictionary containing pair-specific metrics
        """
        if pair_key not in self.pair_metrics:
            raise KeyError(f"Pair {pair_key} not found in backtest data")

        pair_data = self.pair_metrics[pair_key]

        # Get pair-specific data
        pair_trades = self.trade_history[self.trade_history["pair"] == pair_key]
        pair_costs = self.costs.filter(regex=f"{pair_key}.*")

        # Calculate pair portfolio values
        pair_portfolio_values = pd.Series(
            pair_data["position_values"].abs().sum(axis=1), name="gross_portfolio_value"
        ).to_frame()

        # Calculate pair metrics
        pair_trade_metrics = calculate_trade_metrics(
            trades_df=pair_trades, portfolio_df=pair_portfolio_values, costs_df=pair_costs
        )

        pair_risk_metrics = calculate_risk_metrics(
            gross_returns=pair_data.get("daily_gross_returns", pd.Series()),
            net_returns=pair_data.get("daily_net_returns", pd.Series()),
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=self.risk_free_rate,
        )

        pair_return_metrics = calculate_return_metrics(
            gross_returns=pair_data.get("daily_gross_returns", pd.Series()),
            net_returns=pair_data.get("daily_net_returns", pd.Series()),
            costs=pair_costs,
            trades_df=pair_trades,
        )

        return {
            "trade_metrics": pair_trade_metrics,
            "risk_metrics": pair_risk_metrics,
            "return_metrics": pair_return_metrics,
        }

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Generate summary statistics DataFrame.

        Returns:
            DataFrame containing key performance metrics
        """
        if not all([self.trade_metrics, self.risk_metrics, self.return_metrics]):
            self._run_analysis()

        summary = pd.DataFrame(
            {
                "Metric": [
                    # Return Metrics
                    "Gross Total Return",
                    "Net Total Return",
                    "Gross Annualized Return",
                    "Net Annualized Return",
                    "Gross Monthly Return",
                    "Net Monthly Return",
                    "Gross Best Month",
                    "Net Best Month",
                    "Gross Worst Month",
                    "Net Worst Month",
                    "Gross Positive Months %",
                    "Net Positive Months %",
                    # Risk Metrics
                    "Alpha",
                    "Beta to Benchmark",
                    "Correlation to Benchmark",
                    "Gross Volatility",
                    "Net Volatility",
                    "Gross Sharpe Ratio",
                    "Net Sharpe Ratio",
                    "Gross Sortino Ratio",
                    "Net Sortino Ratio",
                    "Gross Calmar Ratio",
                    "Net Calmar Ratio",
                    "Gross Max Drawdown",
                    "Net Max Drawdown",
                    "Gross Max Drawdown Duration",
                    "Net Max Drawdown Duration",
                    "Gross VaR (95%)",
                    "Net VaR (95%)",
                    "Gross CVaR (95%)",
                    "Net CVaR (95%)",
                    # Benchmark Comparison
                    "Information Ratio",
                    "Tracking Error",
                    "Up Capture Ratio",
                    "Down Capture Ratio",
                    "Active Return",
                    "Active Risk",
                    "Monthly Win Rate vs Benchmark",
                    # Trade Metrics
                    "Total Trades",
                    "Win Rate",
                    "Gross Profit Factor",
                    "Net Profit Factor",
                    "Gross Avg Win",
                    "Net Avg Win",
                    "Gross Avg Loss",
                    "Net Avg Loss",
                    # Turnover Metrics
                    "Portfolio Turnover",
                    "Average Rebalances Per Month",
                    "Total Rebalances",
                    "Weighted Average Holding Period",
                    # Cost Metrics
                    "Total Trading Costs",
                    "Avg Monthly Costs",
                    "Cost to AUM Ratio",
                    "Cost Return Ratio",
                    "Daily Cost Drag",
                    "Cost VaR (95%)",
                    "Max Trade Impact",
                ],
                "Value": [
                    # Return Metrics
                    f"{self.return_metrics.gross_total_return:.2%}",
                    f"{self.return_metrics.net_total_return:.2%}",
                    f"{self.return_metrics.gross_annualized_return:.2%}",
                    f"{self.return_metrics.net_annualized_return:.2%}",
                    f"{self.return_metrics.gross_avg_monthly_return:.2%}",
                    f"{self.return_metrics.net_avg_monthly_return:.2%}",
                    f"{self.return_metrics.gross_best_month:.2%}",
                    f"{self.return_metrics.net_best_month:.2%}",
                    f"{self.return_metrics.gross_worst_month:.2%}",
                    f"{self.return_metrics.net_worst_month:.2%}",
                    f"{self.return_metrics.gross_positive_months:.1%}",
                    f"{self.return_metrics.net_positive_months:.1%}",
                    # Risk Metrics
                    (
                        f"{self.risk_metrics.alpha:.2%}"
                        if self.risk_metrics.alpha is not None
                        else "N/A"
                    ),
                    (
                        f"{self.risk_metrics.beta:.2f}"
                        if self.risk_metrics.beta is not None
                        else "N/A"
                    ),
                    (
                        f"{self.risk_metrics.correlation:.2f}"
                        if self.risk_metrics.correlation is not None
                        else "N/A"
                    ),
                    f"{self.risk_metrics.gross_volatility:.2%}",
                    f"{self.risk_metrics.net_volatility:.2%}",
                    f"{self.risk_metrics.gross_sharpe_ratio:.2f}",
                    f"{self.risk_metrics.net_sharpe_ratio:.2f}",
                    f"{self.risk_metrics.gross_sortino_ratio:.2f}",
                    f"{self.risk_metrics.net_sortino_ratio:.2f}",
                    f"{self.risk_metrics.gross_calmar_ratio:.2f}",
                    f"{self.risk_metrics.net_calmar_ratio:.2f}",
                    f"{self.risk_metrics.gross_max_drawdown:.2%}",
                    f"{self.risk_metrics.net_max_drawdown:.2%}",
                    f"{self.risk_metrics.gross_max_drawdown_duration:d} days",
                    f"{self.risk_metrics.net_max_drawdown_duration:d} days",
                    f"{self.risk_metrics.gross_var_95:.2%}",
                    f"{self.risk_metrics.net_var_95:.2%}",
                    f"{self.risk_metrics.gross_cvar_95:.2%}",
                    f"{self.risk_metrics.net_cvar_95:.2%}",
                    # Benchmark Comparison
                    (
                        f"{self.benchmark_analysis['information_ratio']:.2f}"
                        if hasattr(self, "benchmark_analysis")
                        else "N/A"
                    ),
                    (
                        f"{self.benchmark_analysis['tracking_error']:.2%}"
                        if hasattr(self, "benchmark_analysis")
                        else "N/A"
                    ),
                    (
                        f"{self.benchmark_analysis['up_capture']:.2f}"
                        if hasattr(self, "benchmark_analysis")
                        else "N/A"
                    ),
                    (
                        f"{self.benchmark_analysis['down_capture']:.2f}"
                        if hasattr(self, "benchmark_analysis")
                        else "N/A"
                    ),
                    (
                        f"{self.benchmark_analysis['active_return_mean']:.2%}"
                        if hasattr(self, "benchmark_analysis")
                        else "N/A"
                    ),
                    (
                        f"{self.benchmark_analysis['active_return_std']:.2%}"
                        if hasattr(self, "benchmark_analysis")
                        else "N/A"
                    ),
                    (
                        f"{self.benchmark_analysis['monthly_win_rate']:.1%}"
                        if hasattr(self, "benchmark_analysis")
                        else "N/A"
                    ),
                    # Trade Metrics
                    f"{self.trade_metrics.total_trades:,d}",
                    f"{self.trade_metrics.win_rate:.1%}",
                    f"{self.trade_metrics.gross_profit_factor:.2f}",
                    f"{self.trade_metrics.net_profit_factor:.2f}",
                    f"${self.trade_metrics.gross_avg_win:,.2f}",
                    f"${self.trade_metrics.net_avg_win:,.2f}",
                    f"${self.trade_metrics.gross_avg_loss:,.2f}",
                    f"${self.trade_metrics.net_avg_loss:,.2f}",
                    # Turnover Metrics
                    f"{self.trade_metrics.turnover_ratio:.2%}",
                    f"{self.trade_metrics.rebalance_count / (len(self.portfolio_metrics) / 21):.1f}",
                    f"{self.trade_metrics.rebalance_count:,d}",
                    f"{self.trade_metrics.weighted_avg_holding_period:.1f}",
                    # Cost Metrics
                    f"${self.trade_metrics.total_fees:,.2f}",
                    f"${self.return_metrics.avg_monthly_costs:,.2f}",
                    f"{self.trade_metrics.cost_to_aum_ratio:.2%}",
                    f"{self.return_metrics.cost_return_ratio:.2%}",
                    f"{self.trade_metrics.cost_drag:.2%}",
                    f"{self.risk_metrics.cost_var_95:.2%}",
                    f"{self.trade_metrics.max_trade_impact:.2%}",
                ],
            }
        )

        # Add descriptions for each metric
        descriptions = {
            # Return Metrics
            "Gross Total Return": "Total return before trading costs",
            "Net Total Return": "Total return after trading costs",
            "Gross Annualized Return": "Annualized return before costs",
            "Net Annualized Return": "Annualized return after costs",
            "Gross Monthly Return": "Average monthly return before costs",
            "Net Monthly Return": "Average monthly return after costs",
            "Gross Best Month": "Best monthly return before costs",
            "Net Best Month": "Best monthly return after costs",
            "Gross Worst Month": "Worst monthly return before costs",
            "Net Worst Month": "Worst monthly return after costs",
            "Gross Positive Months %": "Percentage of months with positive gross returns",
            "Net Positive Months %": "Percentage of months with positive net returns",
            # Risk Metrics
            "Alpha": "Strategy's alpha to the benchmark",
            "Beta to Benchmark": "Strategy's beta to the benchmark",
            "Correlation to Benchmark": "Strategy's correlation with benchmark",
            "Gross Volatility": "Annualized volatility before costs",
            "Net Volatility": "Annualized volatility after costs",
            "Gross Sharpe Ratio": "Risk-adjusted return before costs",
            "Net Sharpe Ratio": "Risk-adjusted return after costs",
            "Gross Sortino Ratio": "Downside risk-adjusted return before costs",
            "Net Sortino Ratio": "Downside risk-adjusted return after costs",
            "Gross Calmar Ratio": "Return to maximum drawdown ratio before costs",
            "Net Calmar Ratio": "Return to maximum drawdown ratio after costs",
            "Gross Max Drawdown": "Maximum peak-to-trough decline before costs",
            "Net Max Drawdown": "Maximum peak-to-trough decline after costs",
            "Gross Max Drawdown Duration": "Length of maximum drawdown period before costs",
            "Net Max Drawdown Duration": "Length of maximum drawdown period after costs",
            "Gross VaR (95%)": "Value at Risk before costs",
            "Net VaR (95%)": "Value at Risk after costs",
            "Gross CVaR (95%)": "Conditional Value at Risk before costs",
            "Net CVaR (95%)": "Conditional Value at Risk after costs",
            # Benchmark Comparison
            "Information Ratio": "Active return per unit of tracking error",
            "Tracking Error": "Standard deviation of returns vs benchmark",
            "Up Capture Ratio": "Performance in up markets relative to benchmark",
            "Down Capture Ratio": "Performance in down markets relative to benchmark",
            "Active Return": "Annualized return difference vs benchmark",
            "Active Risk": "Annualized standard deviation of active returns",
            "Monthly Win Rate vs Benchmark": "Percentage of months outperforming benchmark",
            # Trade Metrics
            "Total Trades": "Total number of trades executed",
            "Win Rate": "Percentage of trades with positive net P&L",
            "Gross Profit Factor": "Ratio of gross profits to gross losses",
            "Net Profit Factor": "Ratio of net profits to net losses",
            "Avg Holding Period (Days)": "Average duration of trades",
            "Gross Avg Win": "Average winning trade before costs",
            "Net Avg Win": "Average winning trade after costs",
            "Gross Avg Loss": "Average losing trade before costs",
            "Net Avg Loss": "Average losing trade after costs",
            # Turnover Metrics
            "Portfolio Turnover": "Annual portfolio turnover rate",
            "Average Rebalances Per Month": "Average number of rebalances per month",
            "Total Rebalances": "Total number of portfolio rebalances",
            "Weighted Average Holding Period": "Average holding period weighted by position size",
            # Cost Metrics
            "Total Trading Costs": "Total transaction costs",
            "Avg Monthly Costs": "Average monthly transaction costs",
            "Cost to AUM Ratio": "Trading costs as percentage of average AUM",
            "Cost Return Ratio": "Trading costs as percentage of gross returns",
            "Daily Cost Drag": "Average daily performance impact of costs",
            "Cost VaR (95%)": "95th percentile of trading costs",
            "Max Trade Impact": "Maximum trade size relative to portfolio",
        }

        summary["Description"] = summary["Metric"].map(descriptions)

        return summary

    def get_monthly_returns(self) -> pd.DataFrame:
        """Get monthly returns comparison"""
        if not self.return_metrics:
            self._run_analysis()

        return pd.DataFrame(
            {
                "Gross Returns": self.return_metrics.gross_monthly_returns,
                "Net Returns": self.return_metrics.net_monthly_returns,
                "Costs": self.return_metrics.avg_monthly_costs,
            }
        )

    @staticmethod
    def _calculate_equity_curve(returns: pd.Series) -> pd.Series:
        """Calculate cumulative equity curve from returns"""
        return (1 + returns).cumprod()


def analyze_backtest_results(
    backtest_data: Dict[str, Union[pd.DataFrame, Dict]],
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    save_tearsheet: bool = False,
    tearsheet_path: Optional[str] = None,
    use_log_scale: bool = False,
) -> Dict:
    """
    Analyze backtest results comprehensively.

    Parameters:
        backtest_data: Dictionary containing backtest results
            Required keys:
            - portfolio_metrics: DataFrame with portfolio-level metrics
            - trade_history: DataFrame of all trades
            - pair_metrics: Dictionary of pair-specific metrics
            - costs: Dictionary with daily and cumulative costs
        benchmark_returns: Optional benchmark return series for comparison
        risk_free_rate: Annual risk-free rate for Sharpe calculations
        save_tearsheet: Whether to generate and save visualization tearsheet
        tearsheet_path: Path to save tearsheet if generated
        use_log_scale: Whether to use log scale in equity curves

    Returns:
        Dictionary containing analysis results:
            - summary_stats: Overall strategy statistics
            - trade_metrics: Detailed trade analysis
            - risk_metrics: Risk and exposure metrics
            - return_metrics: Performance and return metrics
            - pair_analysis: Individual pair performance
            - correlation_matrix: Pair correlation analysis
    """
    logger.info("Starting backtest analysis...")

    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer(
        backtest_data=backtest_data,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
    )

    try:
        # Calculate portfolio-level metrics
        trade_metrics = calculate_trade_metrics(
            trades_df=backtest_data["trade_history"],
            portfolio_df=backtest_data["portfolio_metrics"],
            costs_df=backtest_data["portfolio_metrics"]["daily_costs"],
        )

        risk_metrics = calculate_risk_metrics(
            gross_returns=backtest_data["portfolio_metrics"]["gross_daily_return"],
            net_returns=backtest_data["portfolio_metrics"]["net_daily_return"],
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
        )

        return_metrics = calculate_return_metrics(
            gross_returns=backtest_data["portfolio_metrics"]["gross_daily_return"],
            net_returns=backtest_data["portfolio_metrics"]["net_daily_return"],
            costs=backtest_data["portfolio_metrics"]["daily_costs"],
            trades_df=backtest_data["trade_history"],
        )

        # Analyze individual pairs
        pair_analysis = {}
        for pair_key in backtest_data["pair_metrics"].keys():
            try:
                pair_analysis[pair_key] = analyzer.analyze_pair(pair_key)
                logger.info(f"Completed analysis for pair: {pair_key}")
            except Exception as e:
                logger.error(
                    f"Error analyzing pair {pair_key}"
                    f"Stack trace:\n"
                    f"{''.join(traceback.format_tb(e.__traceback__))}"
                )
                continue

        # Calculate correlation matrix between pairs
        correlation_matrix = _calculate_pair_correlations(backtest_data["pair_metrics"])

        # Generate summary statistics
        summary_stats = analyzer.get_summary_stats()

        # Create visualization tearsheet if requested
        if save_tearsheet:
            try:
                visualizer = PerformanceVisualizer(backtest_data, benchmark_returns)
                fig = visualizer.generate_tearsheet(
                    save_path=tearsheet_path, use_log_scale=use_log_scale
                )
                logger.info(
                    f"Generated tearsheet{' saved to ' + tearsheet_path if tearsheet_path else ''}"
                )
            except Exception as e:
                logger.error(f"Error generating tearsheet: {str(e)}")

        # Compile all results
        results = {
            "summary_stats": summary_stats,
            "trade_metrics": trade_metrics,
            "risk_metrics": risk_metrics,
            "return_metrics": return_metrics,
            "pair_analysis": pair_analysis,
            "correlation_matrix": correlation_matrix,
        }

        logger.info("Backtest analysis completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error in backtest analysis: {str(e)}")
        raise


def _calculate_pair_correlations(pair_metrics: Dict) -> pd.DataFrame:
    """Calculate correlation matrix between pair returns."""
    # Create DataFrame of pair returns
    pair_returns = pd.DataFrame()

    for pair_key, pair_data in pair_metrics.items():
        if "daily_net_returns" in pair_data:
            pair_returns[f"{pair_key}_net"] = pair_data["daily_net_returns"]
        if "daily_gross_returns" in pair_data:
            pair_returns[f"{pair_key}_gross"] = pair_data["daily_gross_returns"]

    # Calculate and format correlation matrix
    if not pair_returns.empty:
        corr_matrix = pair_returns.corr()

        # Format for readability
        corr_matrix = corr_matrix.round(3)

        # Remove self-correlations
        np.fill_diagonal(corr_matrix.values, np.nan)

        return corr_matrix

    return pd.DataFrame()


def _analyze_benchmark_comparison(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> Dict:
    """
    Analyze strategy performance relative to benchmark.

    Returns detailed comparison metrics including:
    - Tracking error
    - Information ratio
    - Up/down capture ratios
    - Active return statistics
    """
    # Align returns
    aligned_returns = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    strategy_returns = aligned_returns.iloc[:, 0]
    benchmark_returns = aligned_returns.iloc[:, 1]

    # Calculate tracking error
    active_returns = strategy_returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(252)

    # Calculate information ratio
    info_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0

    # Calculate up/down capture
    up_months = benchmark_returns > 0
    down_months = benchmark_returns < 0

    up_capture = (
        strategy_returns[up_months].mean() / benchmark_returns[up_months].mean()
        if len(benchmark_returns[up_months]) > 0
        else 0
    )

    down_capture = (
        strategy_returns[down_months].mean() / benchmark_returns[down_months].mean()
        if len(benchmark_returns[down_months]) > 0
        else 0
    )

    return {
        "tracking_error": tracking_error,
        "information_ratio": info_ratio,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "active_return_mean": active_returns.mean() * 252,
        "active_return_std": active_returns.std() * np.sqrt(252),
        "correlation": strategy_returns.corr(benchmark_returns),
        "monthly_win_rate": (
            len(active_returns[active_returns > 0]) / len(active_returns)
            if len(active_returns) > 0
            else 0
        ),
    }
