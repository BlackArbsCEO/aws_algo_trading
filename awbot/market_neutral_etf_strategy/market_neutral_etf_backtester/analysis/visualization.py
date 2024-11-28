from typing import Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ..config.style_config import MODERN_COLORS


class PerformanceVisualizer:
    """Creates interactive visualizations for backtest performance analysis."""

    def __init__(
        self,
        performance_data: Dict[str, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
        style_config: Optional[Dict] = None,
    ):
        """
        Initialize visualizer with performance data and styling.

        Parameters:
            performance_data: Dictionary containing backtest results
            style_config: Optional custom styling configuration
        """
        self.data = performance_data
        self.benchmark_returns = benchmark_returns
        self.style = style_config or MODERN_COLORS

        # Extract key data components
        self.portfolio_metrics = performance_data["portfolio_metrics"]
        self.trade_history = performance_data["trade_history"]
        self.pair_metrics = performance_data["pair_metrics"]
        self.costs = performance_data["portfolio_metrics"]["daily_costs"]

        # Calculate key series
        self.gross_returns = self.portfolio_metrics["gross_daily_return"]
        self.net_returns = self.portfolio_metrics["net_daily_return"]
        self.gross_equity = self._calculate_equity_curve(self.gross_returns)
        self.net_equity = self._calculate_equity_curve(self.net_returns)

        # Setup plotting configuration
        self._setup_plotting_theme()

    def generate_tearsheet(
        self, save_path: Optional[str] = None, use_log_scale: bool = False
    ) -> go.Figure:
        """
        Generate comprehensive performance tearsheet.

        Parameters:
            save_path: Optional path to save HTML file
            use_log_scale: Whether to use log scale for equity curves

        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=5,
            cols=2,
            subplot_titles=self._get_subplot_titles(),
            vertical_spacing=0.08,
            horizontal_spacing=0.12,
            row_heights=[0.2, 0.2, 0.15, 0.15, 0.15],
        )

        # Add all components
        self._add_equity_curves(fig, use_log_scale)
        self._add_pair_performance(fig, use_log_scale)
        self._add_drawdowns(fig)
        self._add_cost_analysis(fig)
        self._add_return_distribution(fig)
        self._add_cost_distribution(fig)
        self._add_monthly_returns(fig)
        self._add_rolling_metrics(fig)
        self._add_cost_drag(fig)
        self._add_pair_costs(fig)

        # Update layout
        self._update_layout(fig, use_log_scale)

        if save_path:
            fig.write_html(save_path)

        return fig

    def _add_equity_curves(self, fig: go.Figure, use_log_scale: bool):
        """Add equity curves to the figure."""
        # Normalize curves to start at X
        initial_value = 10000
        norm_gross = self._normalize_series(self.gross_returns, initial_value)
        norm_net = self._normalize_series(self.net_returns, initial_value)

        # Add gross equity curve
        fig.add_trace(
            go.Scatter(
                x=norm_gross.index,
                y=norm_gross.values,
                name="Gross Strategy",
                line=dict(width=2, color=self.style["accent_colors"][0]),
                hovertemplate=(
                    "Gross Performance<br>"
                    "Date: %{x}<br>"
                    "Value: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Add net equity curve
        fig.add_trace(
            go.Scatter(
                x=norm_net.index,
                y=norm_net.values,
                name="Net Strategy",
                line=dict(width=2, color=self.style["accent_colors"][1]),
                hovertemplate=(
                    "Net Performance<br>" "Date: %{x}<br>" "Value: %{y:.2f}<br>" "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        # Add benchmark if available
        if self.benchmark_returns is not None:
            norm_benchmark = self._normalize_series(self.benchmark_returns, initial_value)

            fig.add_trace(
                go.Scatter(
                    x=norm_benchmark.index,
                    y=norm_benchmark.values,
                    name="Benchmark",
                    line=dict(width=2, color=MODERN_COLORS["accent_colors"][2], dash="dash"),
                    hovertemplate="Benchmark<br>Date: %{x}<br>Value: %{y:.2f}<br><extra></extra>",
                ),
                row=1,
                col=1,
            )
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(
            title_text=f"Value (Starting={initial_value})",
            row=1,
            col=1,
            type="log" if use_log_scale else "linear",
        )

    def _add_pair_performance(self, fig: go.Figure, use_log_scale: bool):
        """Add individual pair performance plots."""
        for i, (pair_key, pair_data) in enumerate(self.pair_metrics.items()):
            pair_returns = pair_data.get("daily_net_returns", pd.Series())
            if not pair_returns.empty:
                equity = self._normalize_series(pair_returns, 100)

                fig.add_trace(
                    go.Scatter(
                        x=equity.index,
                        y=equity.values,
                        name=pair_key,
                        line=dict(
                            width=2,
                            color=self.style["accent_colors"][
                                i % len(self.style["accent_colors"])
                            ],
                        ),
                        hovertemplate=(
                            f"{pair_key}<br>"
                            "Date: %{x}<br>"
                            "Value: %{y:.2f}<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=2,
                )

        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(
            title_text="Pair Values (Starting=100)",
            row=1,
            col=2,
            type="log" if use_log_scale else "linear",
        )

    def _add_drawdowns(self, fig: go.Figure):
        """Add drawdown plots."""
        # Calculate drawdown series
        gross_dd = self._calculate_drawdown_series(self.gross_equity)
        net_dd = self._calculate_drawdown_series(self.net_equity)

        # Add gross drawdowns
        fig.add_trace(
            go.Scatter(
                x=gross_dd.index,
                y=gross_dd.values * 100,
                name="Gross Drawdown",
                line=dict(width=2, color=self.style["accent_colors"][2]),
                fill="tozeroy",
                hovertemplate=(
                    "Gross Drawdown<br>"
                    "Date: %{x}<br>"
                    "Drawdown: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        # Add net drawdowns
        fig.add_trace(
            go.Scatter(
                x=net_dd.index,
                y=net_dd.values * 100,
                name="Net Drawdown",
                line=dict(width=2, color=self.style["accent_colors"][3]),
                fill="tozeroy",
                hovertemplate=(
                    "Net Drawdown<br>" "Date: %{x}<br>" "Drawdown: %{y:.1f}%<br>" "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    def _add_cost_analysis(self, fig: go.Figure):
        """Add trading cost analysis."""
        daily_costs = self.costs

        # Daily costs bar chart
        fig.add_trace(
            go.Bar(
                x=daily_costs.index,
                y=daily_costs.values,
                name="Daily Costs",
                marker_color=self.style["accent_colors"][4],
                hovertemplate=(
                    "Trading Costs<br>" "Date: %{x}<br>" "Cost: $%{y:,.2f}<br>" "<extra></extra>"
                ),
            ),
            row=4,
            col=1,
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Trading Costs ($)", row=4, col=1)

    def _add_return_distribution(self, fig: go.Figure):
        """Add return distribution plots."""
        # Add gross returns histogram
        fig.add_trace(
            go.Histogram(
                x=self.gross_returns * 100,
                name="Gross Returns",
                nbinsx=50,
                histnorm="probability density",
                marker_color=self.style["accent_colors"][0],
                opacity=0.7,
                hovertemplate=(
                    "Gross Returns<br>"
                    "Return: %{x:.2f}%<br>"
                    "Density: %{y:.4f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )

        # Add net returns histogram
        fig.add_trace(
            go.Histogram(
                x=self.net_returns * 100,
                name="Net Returns",
                nbinsx=50,
                histnorm="probability density",
                marker_color=self.style["accent_colors"][1],
                opacity=0.7,
                hovertemplate=(
                    "Net Returns<br>"
                    "Return: %{x:.2f}%<br>"
                    "Density: %{y:.4f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )

        # Update axes
        fig.update_xaxes(title_text="Daily Return (%)", row=3, col=1)
        fig.update_yaxes(title_text="Density", row=3, col=1)

    def _add_cost_distribution(self, fig: go.Figure):
        """Add cost distribution analysis."""
        daily_costs = self.costs
        portfolio_values = self.portfolio_metrics["gross_portfolio_value"]

        # Calculate cost percentages
        cost_pct = (daily_costs / portfolio_values) * 100
        cost_pct = cost_pct[cost_pct != 0]  # Remove zero-cost days

        if not cost_pct.empty:
            fig.add_trace(
                go.Histogram(
                    x=cost_pct,
                    name="Cost Impact",
                    nbinsx=30,
                    histnorm="probability",
                    marker_color=self.style["accent_colors"][3],
                    opacity=0.7,
                    hovertemplate=(
                        "Cost Distribution<br>"
                        "Cost: %{x:.3f}%<br>"
                        "Probability: %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                ),
                row=4,
                col=2,
            )

            # Add statistics annotation
            # stats_text = (
            #     f"Cost Statistics:<br>"
            #     f"Mean: {cost_pct.mean():.3f}%<br>"
            #     f"Median: {cost_pct.median():.3f}%<br>"
            #     f"Std Dev: {cost_pct.std():.3f}%<br>"
            #     f"95th Pct: {cost_pct.quantile(0.95):.3f}%"
            # )
            #
            # fig.add_annotation(
            #     xref="x8",
            #     yref="y8",
            #     x=0.95, # expands the image beyond the values
            #     y=0.95, # expands the image beyond the values
            #     text=stats_text,
            #     showarrow=False,
            #     align="right",
            #     bgcolor=self.style["paper_color"],
            #     bordercolor=self.style["grid_color"],
            #     borderwidth=1,
            # )

        # Update axes
        fig.update_xaxes(title_text="Cost (% of Portfolio)", row=4, col=2)
        fig.update_yaxes(title_text="Probability", row=4, col=2)

    def _add_monthly_returns(self, fig: go.Figure):
        """Add monthly return comparison."""
        # Calculate monthly returns
        gross_monthly = self.gross_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
        net_monthly = self.net_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

        # Add gross monthly returns

        fig.add_trace(
            go.Bar(
                x=gross_monthly.index,
                y=gross_monthly.values,
                name="Gross Monthly",
                marker_color=self.style["accent_colors"][0],
                opacity=0.7,
                hovertemplate=(
                    "Gross Monthly Return<br>"
                    "Date: %{x}<br>"
                    "Return: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

        # Add net monthly returns
        fig.add_trace(
            go.Bar(
                x=net_monthly.index,
                y=net_monthly.values,
                name="Net Monthly",
                marker_color=self.style["accent_colors"][1],
                opacity=0.7,
                hovertemplate=(
                    "Net Monthly Return<br>"
                    "Date: %{x}<br>"
                    "Return: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Monthly Return (%)", row=2, col=2)

    def _add_rolling_metrics(self, fig: go.Figure, window: int = 63):
        """Add rolling performance metrics."""
        # Calculate rolling Sharpe ratios
        gross_sharpe = self._calculate_rolling_sharpe(self.gross_returns, window)
        net_sharpe = self._calculate_rolling_sharpe(self.net_returns, window)

        # Add gross Sharpe
        fig.add_trace(
            go.Scatter(
                x=gross_sharpe.index,
                y=gross_sharpe.values,
                name="Rolling Gross Sharpe",
                line=dict(width=2, color=self.style["accent_colors"][0]),
                hovertemplate=(
                    "Rolling Gross Sharpe<br>"
                    "Date: %{x}<br>"
                    "Sharpe: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=3,
            col=2,
        )

        # Add net Sharpe
        fig.add_trace(
            go.Scatter(
                x=net_sharpe.index,
                y=net_sharpe.values,
                name="Rolling Net Sharpe",
                line=dict(width=2, color=self.style["accent_colors"][1]),
                hovertemplate=(
                    "Rolling Net Sharpe<br>"
                    "Date: %{x}<br>"
                    "Sharpe: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=3,
            col=2,
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text=f"Rolling {window}d Sharpe Ratio", row=3, col=2)

    def _add_cost_drag(self, fig: go.Figure):
        """Add cumulative cost drag analysis."""
        # Calculate cumulative cost impact
        portfolio_values = self.portfolio_metrics["gross_portfolio_value"]
        cumulative_costs = self.costs.cumsum()

        cost_drag = -(cumulative_costs / portfolio_values) * 100
        cost_drag = cost_drag.fillna(0)

        fig.add_trace(
            go.Scatter(
                x=cost_drag.index,
                y=cost_drag.values,
                name="Cumulative Cost Drag",
                line=dict(width=2, color=self.style["accent_colors"][4]),
                fill="tozeroy",
                hovertemplate=(
                    "Cumulative Cost Drag<br>"
                    "Date: %{x}<br>"
                    "Drag: -%{y:.3f}%<br>"
                    "<extra></extra>"
                ),
            ),
            row=5,
            col=1,
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=5, col=1)
        fig.update_yaxes(title_text="Cumulative Cost Drag (%)", row=5, col=1)

    def _add_pair_costs(self, fig: go.Figure):
        """Add pair-specific cost analysis."""
        # Calculate cumulative costs per pair
        pair_costs = {}
        date_range = pd.date_range(
            self.trade_history["timestamp"].min(), self.trade_history["timestamp"].max(), freq="D"
        )

        for pair_key in self.pair_metrics.keys():
            pair_trades = self.trade_history[self.trade_history["pair"] == pair_key]
            daily_costs = pair_trades.groupby("timestamp")["cost"].sum()
            cumulative_costs = daily_costs.reindex(date_range).fillna(0).cumsum()
            pair_costs[pair_key] = cumulative_costs

        # Create stacked area chart
        for i, (pair_key, costs) in enumerate(pair_costs.items()):
            fig.add_trace(
                go.Scatter(
                    x=costs.index,
                    y=costs.values,
                    name=f"{pair_key} Costs",
                    stackgroup="one",
                    line=dict(width=0.5),
                    fillcolor=self.style["accent_colors"][i % len(self.style["accent_colors"])],
                    hovertemplate=(
                        f"{pair_key}<br>"
                        "Date: %{x}<br>"
                        "Cumulative Cost: $%{y:,.2f}<br>"
                        "<extra></extra>"
                    ),
                ),
                row=5,
                col=2,
            )

        # Update axes
        fig.update_xaxes(title_text="Date", row=5, col=2)
        fig.update_yaxes(title_text="Cumulative Costs by Pair ($)", row=5, col=2)

    def _update_layout(self, fig: go.Figure, use_log_scale: bool):
        """Update figure layout with consistent styling."""
        fig.update_layout(
            height=2000,
            width=1600,
            showlegend=True,
            template="plotly_dark",
            paper_bgcolor=self.style["paper_color"],
            plot_bgcolor=self.style["background_color"],
            title=dict(
                text=f"Strategy Performance Analysis {'(Log Scale)' if use_log_scale else ''}",
                x=0.5,
                font=dict(size=24, color=self.style["text_color"]),
            ),
            legend=dict(
                bgcolor=self.style["paper_color"],
                bordercolor=self.style["grid_color"],
                borderwidth=1,
                font=dict(color=self.style["text_color"]),
            ),
            hoverlabel=dict(
                bgcolor=self.style["paper_color"],
                font_size=14,
                font_family="Arial, sans-serif",
            ),
        )

    @staticmethod
    def _get_subplot_titles() -> tuple:
        """Get titles for all subplots."""
        return (
            "Gross vs Net Equity Curves",
            "Individual Pair Performance",
            "Strategy Drawdowns",
            "Monthly Returns Comparison",
            "Return Distributions",
            "Rolling Sharpe Ratio",
            "Trading Costs Analysis",
            "Cost Impact Distribution",
            "Cost Drag Analysis",
            "Pair Cost Analysis",
        )

    @staticmethod
    def _normalize_series(series: pd.Series, initial_value: float = 1.0) -> pd.Series:
        """Normalize a return series to start at initial_value."""
        return initial_value * (1 + series).cumprod()

    @staticmethod
    def _calculate_equity_curve(returns: pd.Series) -> pd.Series:
        """Calculate cumulative equity curve from returns."""
        return (1 + returns).cumprod()

    @staticmethod
    def _calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        return drawdowns

    @staticmethod
    def _calculate_rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        return np.sqrt(252) * (rolling_mean / rolling_std)

    def _setup_plotting_theme(self):
        """Setup custom plotting theme."""
        template = go.layout.Template()

        # Set global defaults
        template.layout = dict(
            font=dict(
                family="Arial, sans-serif",
                color=self.style["text_color"],
                size=14,
            ),
            paper_bgcolor=self.style["paper_color"],
            plot_bgcolor=self.style["background_color"],
            xaxis=dict(
                gridcolor=self.style["grid_color"],
                zerolinecolor=self.style["grid_color"],
            ),
            yaxis=dict(
                gridcolor=self.style["grid_color"],
                zerolinecolor=self.style["grid_color"],
            ),
        )

        # Set as default template
        pio.templates["custom"] = template
        pio.templates.default = "custom"
