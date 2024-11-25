from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots

# Define theme colors globally
MODERN_COLORS = {
    "background_color": "#000000",  # Pure black
    "paper_color": "#121212",  # Very dark gray
    "text_color": "#FFFFFF",  # Pure white
    "grid_color": "#333333",  # Dark gray
    "accent_colors": [
        "#00A3FF",  # Vibrant blue
        "#00FF85",  # Electric green
        "#FF00E5",  # Magenta
        "#FF9900",  # Orange
        "#FFD600",  # Yellow
        "#00E5FF",  # Cyan
        "#FF4081",  # Pink
    ],
}


def create_modern_theme():
    """Create a high-contrast theme optimized for screen recording."""
    modern = go.layout.Template()

    # Set global defaults
    modern.layout = dict(
        font=dict(
            family="Arial, sans-serif",
            color=MODERN_COLORS["text_color"],
            size=14,
        ),
        paper_bgcolor=MODERN_COLORS["paper_color"],
        plot_bgcolor=MODERN_COLORS["background_color"],
        title=dict(font=dict(size=20, color=MODERN_COLORS["text_color"]), x=0.5, xanchor="center"),
        # Grid styling
        xaxis=dict(
            gridcolor=MODERN_COLORS["grid_color"],
            linecolor=MODERN_COLORS["grid_color"],
            zerolinecolor=MODERN_COLORS["grid_color"],
            tickcolor=MODERN_COLORS["text_color"],
            tickfont=dict(color=MODERN_COLORS["text_color"], size=12),
            title=dict(font=dict(color=MODERN_COLORS["text_color"], size=14)),
        ),
        yaxis=dict(
            gridcolor=MODERN_COLORS["grid_color"],
            linecolor=MODERN_COLORS["grid_color"],
            zerolinecolor=MODERN_COLORS["grid_color"],
            tickcolor=MODERN_COLORS["text_color"],
            tickfont=dict(color=MODERN_COLORS["text_color"], size=12),
            title=dict(font=dict(color=MODERN_COLORS["text_color"], size=14)),
        ),
        legend=dict(
            bgcolor=MODERN_COLORS["paper_color"],
            font=dict(color=MODERN_COLORS["text_color"], size=12),
            bordercolor=MODERN_COLORS["grid_color"],
            borderwidth=1,
        ),
        colorway=MODERN_COLORS["accent_colors"],
    )

    # Scatter plot defaults with increased visibility
    modern.data.scatter = [
        go.Scatter(line=dict(width=3), marker=dict(size=10))  # Thicker lines  # Larger markers
    ]

    # Box plot defaults
    modern.data.box = [
        go.Box(
            marker=dict(
                outliercolor=MODERN_COLORS["accent_colors"][2],  # Magenta for outliers
                color=MODERN_COLORS["accent_colors"][0],  # Blue for boxes
                size=8,  # Larger markers
            ),
            line=dict(color=MODERN_COLORS["text_color"], width=2),
            fillcolor=MODERN_COLORS["accent_colors"][0],
        )
    ]

    # Bar plot defaults
    modern.data.bar = [go.Bar(marker=dict(line=dict(color=MODERN_COLORS["text_color"], width=1)))]

    return modern


# Register the theme
pio.templates["modern"] = create_modern_theme()
pio.templates.default = "modern"


def calculate_market_neutral_returns(
    df: pd.DataFrame, long_etf: str, inverse_etf: str, leverage: float
) -> pd.Series:
    """
    Calculate market neutral returns from a DataFrame of ETF returns.

    Parameters:
        df: DataFrame containing ETF returns
        long_etf: Long ETF symbol
        inverse_etf: Inverse ETF symbol
        leverage: Leverage multiplier for the inverse ETF

    Returns:
        Series of market neutral returns
    """
    inverse_weight = 1 / (1 + leverage)
    long_weight = 1 - inverse_weight
    returns = long_weight * df[f"{long_etf}_return"] - inverse_weight * df[f"{inverse_etf}_return"]
    returns.index = pd.to_datetime(returns.index)  # Ensure datetime index
    return returns


def calculate_rolling_vol_regimes(
    series: pd.Series, lookback: int = 21
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate rolling volatility regimes.

    Parameters:
        series: Series of returns
        lookback: Rolling window size

    Returns:
        Tuple of two Series: volatility regimes and rolling volatility values
    """
    n: int = len(series)
    vol: np.ndarray = np.full(n, np.nan)
    regimes: pd.Series = pd.Series(index=series.index, data="Unknown")

    rolling_vol: pd.Series = series.rolling(lookback).std() * np.sqrt(252)

    for i in range(lookback, n):
        vol[i] = rolling_vol.iloc[i]
        historical_vol = rolling_vol.iloc[lookback : i + 1]
        if len(historical_vol) > 0:
            quantiles = np.nanquantile(historical_vol, [0.33, 0.67])
            current_vol = rolling_vol.iloc[i]

            if current_vol <= quantiles[0]:
                regimes.iloc[i] = "Low Vol"
            elif current_vol <= quantiles[1]:
                regimes.iloc[i] = "Med Vol"
            else:
                regimes.iloc[i] = "High Vol"

    return regimes, pd.Series(vol, index=series.index)


def calculate_rolling_corr_regimes(
    returns1: pd.Series, returns2: pd.Series, lookback: int = 21
) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling correlation regime calculation.

    Parameters:
        returns1: First series of returns
        returns2: Second series of returns
        lookback: Rolling window size (default is 21)

    Returns:
        Tuple containing two Series:
        - regimes: Series of correlation regimes ("Low Corr", "Med Corr", "High Corr")
        - corr: Series of rolling correlation values
    """
    n: int = len(returns1)
    corr: np.ndarray = np.full(n, np.nan)
    regimes: pd.Series = pd.Series(index=returns1.index, data="Unknown")

    rolling_corr: pd.Series = returns1.rolling(lookback).corr(returns2)

    for i in range(lookback, n):
        corr[i] = rolling_corr.iloc[i]
        historical_corr = rolling_corr.iloc[lookback : i + 1]
        if len(historical_corr) > 0:
            quantiles = np.nanquantile(historical_corr, [0.33, 0.67])
            current_corr = rolling_corr.iloc[i]

            if current_corr <= quantiles[0]:
                regimes.iloc[i] = "Low Corr"
            elif current_corr <= quantiles[1]:
                regimes.iloc[i] = "Med Corr"
            else:
                regimes.iloc[i] = "High Corr"

    return regimes, pd.Series(corr, index=returns1.index)


def fetch_etf_data(
    pair_symbols: List[Tuple[str, str, float]],
    start_date: str = "2012-01-01",
    end_date: str = "2022-01-01",
) -> pd.DataFrame:
    """
    Fetches and formats ETF price and returns data for a given set of ETF pairs.

    Parameters:
        pair_symbols: List of tuples containing the long, inverse, and leverage for each ETF pair
        start_date: Start date for data download (optional, default: "2012-01-01")
        end_date: End date for data download (optional, default: "2022-01-01")

    Returns:
        DataFrame containing daily returns and prices for each ETF pair
    """
    data: Dict[str, pd.Series] = {}
    for long_etf, inverse_etf, leverage in pair_symbols:
        # Fetch data for both ETFs
        long_data = yf.download(long_etf, start=start_date, end=end_date)["Adj Close"]
        inverse_data = yf.download(inverse_etf, start=start_date, end=end_date)["Adj Close"]

        # Calculate daily returns
        data[f"{long_etf}_return"] = long_data.pct_change()
        data[f"{inverse_etf}_return"] = inverse_data.pct_change()

        # Store prices
        data[f"{long_etf}_price"] = long_data
        data[f"{inverse_etf}_price"] = inverse_data

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)  # Ensure datetime index
    return df


def create_cumulative_returns_plot(
    df: pd.DataFrame, pair_symbols: List[Tuple[str, str, float]], use_log_scale: bool = False
) -> go.Figure:
    """
    Create a cumulative returns plot for a given set of ETF pairs.

    Parameters:
        df: DataFrame containing daily returns and prices for each ETF pair
        pair_symbols: List of tuples containing the long, inverse, and leverage for each ETF pair
        use_log_scale: Whether to use a log scale for the y-axis (default: False)

    Returns:
        A Plotly figure for the cumulative returns plot
    """

    fig = go.Figure()

    for long_etf, inverse_etf, leverage in pair_symbols:
        market_neutral_rets = calculate_market_neutral_returns(df, long_etf, inverse_etf, leverage)
        cum_returns = (1 + market_neutral_rets).cumprod()

        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns,
                name=f"{long_etf}-{inverse_etf}",
                line=dict(width=3),
            )
        )
        # Update y-axis type for first equity plot
        fig.update_yaxes(type="log" if use_log_scale else "linear")
    fig.update_layout(
        title="Cumulative Returns",
        height=800,
        width=1200,
        showlegend=True,
        yaxis_title="Cumulative Return",
        xaxis_title="Date",
    )
    return fig


def create_regime_distributions_plot(
    df: pd.DataFrame, pair_symbols: List[Tuple[str, str, float]]
) -> go.Figure:
    """
    Create a plot of daily returns distributions for each ETF pair, split by regime.

    Parameters:
        df: DataFrame containing daily returns and prices for each ETF pair
        pair_symbols: List of tuples containing the long, inverse, and leverage for each ETF pair

    Returns:
        A Plotly figure for the distributions plot
    """
    fig = go.Figure()

    vol_regime_order = ["Low Vol", "Med Vol", "High Vol"]
    corr_regime_order = ["Low Corr", "Med Corr", "High Corr"]
    regime_combinations = [
        f"{vol} / {corr}" for vol in vol_regime_order for corr in corr_regime_order
    ]

    for i, (long_etf, inverse_etf, leverage) in enumerate(pair_symbols):
        market_neutral_rets = calculate_market_neutral_returns(df, long_etf, inverse_etf, leverage)
        vol_regimes_rt, _ = calculate_rolling_vol_regimes(df[f"{long_etf}_return"])
        corr_regimes_rt, _ = calculate_rolling_corr_regimes(
            df[f"{long_etf}_return"], df[f"{inverse_etf}_return"]
        )

        regime_data = pd.DataFrame(
            {
                "Returns": market_neutral_rets,
                "Vol_Regime": pd.Categorical(
                    vol_regimes_rt, categories=vol_regime_order, ordered=True
                ),
                "Corr_Regime": pd.Categorical(
                    corr_regimes_rt, categories=corr_regime_order, ordered=True
                ),
            }
        )

        regime_data["Regime_Combined"] = (
            regime_data["Vol_Regime"].astype(str) + " / " + regime_data["Corr_Regime"].astype(str)
        )

        for regime in regime_combinations:
            if regime in regime_data["Regime_Combined"].values:
                regime_returns = regime_data[regime_data["Regime_Combined"] == regime]["Returns"]
                fig.add_trace(
                    go.Box(
                        y=regime_returns,
                        name=regime,
                        legendgroup=f"{long_etf}-{inverse_etf}",
                        showlegend=True,
                        marker_color=MODERN_COLORS["accent_colors"][
                            i % len(MODERN_COLORS["accent_colors"])
                        ],
                    )
                )

    fig.update_layout(
        title="Return Distributions by Regime",
        height=800,
        width=1200,
        showlegend=True,
        yaxis_title="Daily Returns",
        xaxis_title="Regime",
    )
    return fig


def create_regime_heatmaps(
    df: pd.DataFrame, pair_symbols: List[Tuple[str, str, float]]
) -> go.Figure:
    """
    Create standalone regime Sharpe ratio heatmap plot.

    Parameters:
        df: DataFrame containing ETF returns
        pair_symbols: List of tuples containing long ETF symbol, inverse ETF symbol, and leverage multiplier

    Returns:
        Plotly figure object
    """
    n_pairs = len(pair_symbols)
    fig = make_subplots(
        rows=1,
        cols=n_pairs,
        subplot_titles=[
            f"{long_etf}-{inverse_etf} Regime Sharpe" for long_etf, inverse_etf, _ in pair_symbols
        ],
    )

    vol_regime_order = ["Low Vol", "Med Vol", "High Vol"]
    corr_regime_order = ["Low Corr", "Med Corr", "High Corr"]

    for col, (long_etf, inverse_etf, leverage) in enumerate(pair_symbols, 1):
        market_neutral_rets = calculate_market_neutral_returns(df, long_etf, inverse_etf, leverage)
        vol_regimes_rt, _ = calculate_rolling_vol_regimes(df[f"{long_etf}_return"])
        corr_regimes_rt, _ = calculate_rolling_corr_regimes(
            df[f"{long_etf}_return"], df[f"{inverse_etf}_return"]
        )

        regime_data = pd.DataFrame(
            {
                "Returns": market_neutral_rets,
                "Vol_Regime": pd.Categorical(
                    vol_regimes_rt, categories=vol_regime_order, ordered=True
                ),
                "Corr_Regime": pd.Categorical(
                    corr_regimes_rt, categories=corr_regime_order, ordered=True
                ),
            }
        )

        regime_stats = (
            regime_data.groupby(["Vol_Regime", "Corr_Regime"], observed=False)["Returns"]
            .agg(lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)))
            .reset_index()
        )

        heatmap_data = regime_stats.pivot(
            index="Vol_Regime", columns="Corr_Regime", values="Returns"
        ).loc[vol_regime_order, corr_regime_order]

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=corr_regime_order,
                y=vol_regime_order,
                colorscale=[
                    [0, MODERN_COLORS["accent_colors"][3]],  # Red for negative
                    [0.5, MODERN_COLORS["paper_color"]],  # Dark gray for zero
                    [1, MODERN_COLORS["accent_colors"][1]],  # Green for positive
                ],
                zmid=0,
                text=np.round(heatmap_data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 14},
                showscale=False,
            ),
            row=1,
            col=col,
        )

    fig.update_layout(height=400, width=1200, title_text="Regime Sharpe Ratios", showlegend=False)

    return fig


def create_annual_sharpe_plot(
    df: pd.DataFrame, pair_symbols: List[Tuple[str, str, float]]
) -> go.Figure:
    """
    Create plot of annual Sharpe ratios for each pair.

    Parameters:
        df: DataFrame containing daily returns and prices for each ETF pair
        pair_symbols: List of tuples containing the long, inverse, and leverage for each ETF pair

    Returns:
        A Plotly figure for the annual Sharpe plot
    """
    fig = go.Figure()

    for i, (long_etf, inverse_etf, leverage) in enumerate(pair_symbols):
        market_neutral_rets = calculate_market_neutral_returns(df, long_etf, inverse_etf, leverage)

        annual_stats = []
        for year in market_neutral_rets.index.year.unique():
            year_mask = market_neutral_rets.index.year == year
            year_rets = market_neutral_rets[year_mask]
            annual_mean = year_rets.mean() * 252
            annual_vol = year_rets.std() * np.sqrt(252)
            sharpe = annual_mean / annual_vol if annual_vol != 0 else 0
            annual_stats.append({"Year": year, "Sharpe": sharpe})

        annual_df = pd.DataFrame(annual_stats)

        fig.add_trace(
            go.Bar(
                x=annual_df["Year"],
                y=annual_df["Sharpe"],
                name=f"{long_etf}-{inverse_etf}",
                marker_color=MODERN_COLORS["accent_colors"][
                    i % len(MODERN_COLORS["accent_colors"])
                ],
                text=annual_df["Sharpe"].round(2),
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Annual Sharpe Ratios",
        height=600,
        width=1200,
        showlegend=True,
        yaxis_title="Sharpe Ratio",
        xaxis_title="Year",
    )

    return fig


def create_regime_dashboard(
    df: pd.DataFrame, pair_symbols: List[Tuple[str, str, float]]
) -> go.Figure:
    """
    Creates a plotly dashboard with four subplots for each ETF pair:

    1. Cumulative returns plot with thicker lines
    2. Box plot with enhanced visibility
    3. Regime Sharpe heatmap with enhanced visibility
    4. Annual Sharpe Analysis with enhanced visibility

    Parameters:
        df: DataFrame containing ETF returns
        pair_symbols: List of tuples containing long ETF symbol, inverse ETF symbol, and leverage multiplier

    Returns:
        Plotly figure object
    """
    n_pairs = len(pair_symbols)

    titles = []
    for _ in range(4):
        for pair in pair_symbols:
            if len(titles) < n_pairs:
                titles.append(f"{pair[0]}-{pair[1]} Cumulative Returns")
            elif len(titles) < 2 * n_pairs:
                titles.append(f"{pair[0]}-{pair[1]} Return Distribution")
            elif len(titles) < 3 * n_pairs:
                titles.append(f"{pair[0]}-{pair[1]} Regime Sharpe")
            else:
                titles.append(f"{pair[0]}-{pair[1]} Annual Sharpe")

    fig = make_subplots(
        rows=4,
        cols=n_pairs,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    vol_regime_order = ["Low Vol", "Med Vol", "High Vol"]
    corr_regime_order = ["Low Corr", "Med Corr", "High Corr"]

    for col, (long_etf, inverse_etf, leverage) in enumerate(pair_symbols, 1):
        market_neutral_rets = calculate_market_neutral_returns(df, long_etf, inverse_etf, leverage)
        vol_regimes_rt, vol = calculate_rolling_vol_regimes(df[f"{long_etf}_return"])
        corr_regimes_rt, corr = calculate_rolling_corr_regimes(
            df[f"{long_etf}_return"], df[f"{inverse_etf}_return"]
        )

        regime_data = pd.DataFrame(
            {
                "Returns": market_neutral_rets,
                "Vol_Regime": pd.Categorical(
                    vol_regimes_rt, categories=vol_regime_order, ordered=True
                ),
                "Corr_Regime": pd.Categorical(
                    corr_regimes_rt, categories=corr_regime_order, ordered=True
                ),
                "Volatility": vol,
                "Correlation": corr,
                "Date": df.index,
            }
        )

        # Remove unknown regimes
        regime_data = regime_data[
            (regime_data["Vol_Regime"] != "Unknown") & (regime_data["Corr_Regime"] != "Unknown")
        ]

        # Calculate and print regime statistics
        regime_stats = (
            regime_data.groupby(["Vol_Regime", "Corr_Regime"], observed=False)["Returns"]
            .agg(
                [
                    ("Mean", lambda x: x.mean() * 252),
                    ("Std", lambda x: x.std() * np.sqrt(252)),
                    ("Sharpe", lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252))),
                    ("Count", "count"),
                ]
            )
            .round(3)
            .reset_index()
        )

        print(f"\nRegime Statistics for {long_etf}-{inverse_etf}:")
        print(regime_stats.to_string(index=False))

        # Create heatmap data
        heatmap_data = regime_stats.pivot(
            index="Vol_Regime", columns="Corr_Regime", values="Sharpe"
        ).loc[vol_regime_order, corr_regime_order]

        # 1. Cumulative returns plot with thicker lines
        cum_returns = (1 + regime_data["Returns"]).cumprod()
        fig.add_trace(
            go.Scatter(
                x=regime_data["Date"],
                y=cum_returns,
                name=f"{long_etf}-{inverse_etf}",
                showlegend=False,
                line=dict(width=3, color=MODERN_COLORS["accent_colors"][col - 1]),
            ),
            row=1,
            col=col,
        )

        # 2. Box plot with enhanced visibility
        regime_data["Regime_Combined"] = (
            regime_data["Vol_Regime"].astype(str) + " / " + regime_data["Corr_Regime"].astype(str)
        )
        regime_combinations = [
            f"{vol} / {corr}" for vol in vol_regime_order for corr in corr_regime_order
        ]

        for regime in regime_combinations:
            if regime in regime_data["Regime_Combined"].values:
                regime_returns = regime_data[regime_data["Regime_Combined"] == regime]["Returns"]
                fig.add_trace(
                    go.Box(
                        y=regime_returns,
                        name=regime,
                        showlegend=False,
                        marker=dict(color=MODERN_COLORS["accent_colors"][col - 1], size=8),
                        line=dict(width=2, color=MODERN_COLORS["text_color"]),
                    ),
                    row=2,
                    col=col,
                )

        # 3. Regime Sharpe heatmap with enhanced visibility
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=corr_regime_order,
                y=vol_regime_order,
                colorscale=[
                    [0, MODERN_COLORS["accent_colors"][3]],  # Red for negative
                    [0.5, MODERN_COLORS["paper_color"]],  # Dark gray for zero
                    [1, MODERN_COLORS["accent_colors"][1]],  # Green for positive
                ],
                zmid=0,
                text=np.round(heatmap_data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 14, "color": MODERN_COLORS["text_color"]},
                showscale=False,
            ),
            row=3,
            col=col,
        )

        # 4. Annual Sharpe Analysis with enhanced visibility
        annual_stats = []
        for year in market_neutral_rets.index.year.unique():
            year_mask = market_neutral_rets.index.year == year
            year_rets = market_neutral_rets[year_mask]
            annual_mean = year_rets.mean() * 252
            annual_vol = year_rets.std() * np.sqrt(252)
            sharpe = annual_mean / annual_vol if annual_vol != 0 else 0
            annual_stats.append({"Year": year, "Sharpe": sharpe, "Returns": len(year_rets)})

        annual_df = pd.DataFrame(annual_stats)

        fig.add_trace(
            go.Bar(
                x=annual_df["Year"],
                y=annual_df["Sharpe"],
                marker_color=MODERN_COLORS["accent_colors"][col - 1],
                showlegend=False,
                text=annual_df["Sharpe"].round(2),
                textposition="auto",
                textfont=dict(size=12, color=MODERN_COLORS["text_color"]),
            ),
            row=4,
            col=col,
        )

    # Update layout with modern theme
    fig.update_layout(
        height=2000,
        width=1800,
        title=dict(
            text="ETF Pair Analysis by Regime",
            font=dict(size=24, color=MODERN_COLORS["text_color"]),
        ),
        showlegend=False,
        template="modern",
        margin=dict(t=150),
        paper_bgcolor=MODERN_COLORS["paper_color"],
        plot_bgcolor=MODERN_COLORS["background_color"],
        hoverlabel=dict(
            font=dict(family="Arial, sans-serif", size=14), bgcolor=MODERN_COLORS["paper_color"]
        ),
    )

    # Update subplot styling
    for row in range(1, 5):
        for col in range(1, n_pairs + 1):
            # Update axes with modern theme
            fig.update_xaxes(
                row=row,
                col=col,
                showgrid=True,
                tickangle=45 if row > 1 else 0,
                gridcolor=MODERN_COLORS["grid_color"],
                linecolor=MODERN_COLORS["grid_color"],
                tickfont=dict(size=12, color=MODERN_COLORS["text_color"]),
                title_font=dict(size=14, color=MODERN_COLORS["text_color"]),
            )
            fig.update_yaxes(
                row=row,
                col=col,
                showgrid=True,
                gridcolor=MODERN_COLORS["grid_color"],
                linecolor=MODERN_COLORS["grid_color"],
                tickfont=dict(size=12, color=MODERN_COLORS["text_color"]),
                title_font=dict(size=14, color=MODERN_COLORS["text_color"]),
            )

            # Add reference line for Sharpe ratio plots
            if row in [4]:
                fig.add_hline(
                    y=0,
                    row=row,
                    col=col,
                    line_color=MODERN_COLORS["grid_color"],
                    line_width=1,
                    line_dash="dash",
                )

    # Update specific row titles with larger font
    for col in range(1, n_pairs + 1):
        fig.update_yaxes(
            title=dict(
                text="Cumulative Return", font=dict(size=14, color=MODERN_COLORS["text_color"])
            ),
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title=dict(
                text="Daily Returns", font=dict(size=14, color=MODERN_COLORS["text_color"])
            ),
            row=2,
            col=col,
        )
        fig.update_yaxes(
            title=dict(
                text="Volatility Regime", font=dict(size=14, color=MODERN_COLORS["text_color"])
            ),
            row=3,
            col=col,
        )
        fig.update_yaxes(
            title=dict(
                text="Annual Sharpe", font=dict(size=14, color=MODERN_COLORS["text_color"])
            ),
            row=4,
            col=col,
        )
        fig.update_xaxes(
            title=dict(
                text="Correlation Regime", font=dict(size=14, color=MODERN_COLORS["text_color"])
            ),
            row=3,
            col=col,
        )

    return fig


# grab data
pair_symbols = [
    ("QQQ", "SQQQ", 3.0),
    ("TLT", "TBT", 2.0),
    ("SMH", "SOXS", 3.0),
    ("XLF", "FAZ", 3.0),
    ("XLE", "ERY", 2.0),
]

df = fetch_etf_data(pair_symbols)

# Create individual plots
cum_returns_fig = create_cumulative_returns_plot(df, pair_symbols, use_log_scale=True)
distributions_fig = create_regime_distributions_plot(df, pair_symbols)
heatmaps_fig = create_regime_heatmaps(df, pair_symbols)
annual_sharpe_fig = create_annual_sharpe_plot(df, pair_symbols)

# Save or display plots
cum_returns_fig.write_html("cumulative_returns.html")
distributions_fig.write_html("distributions.html")
heatmaps_fig.write_html("heatmaps.html")
annual_sharpe_fig.write_html("annual_sharpe.html")

# create full dashboard
fig = create_regime_dashboard(df, pair_symbols)
fig.write_html("full_dashboard.html")
