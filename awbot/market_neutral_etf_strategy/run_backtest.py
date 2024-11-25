from pathlib import Path

import yfinance as yf
from blk_utils import timer

from awbot.market_neutral_etf_strategy.market_neutral_etf_backtester.analysis.performance import (
    analyze_backtest_results,
)
from awbot.market_neutral_etf_strategy.market_neutral_etf_backtester.config.backtest_config import (
    RebalanceFrequency,
    PairConfig,
    BacktestConfig,
)
from awbot.market_neutral_etf_strategy.market_neutral_etf_backtester.core.backtester import (
    MarketNeutralBacktester,
)
from awbot.market_neutral_etf_strategy.market_neutral_etf_backtester.utils.data_utils import (
    prepare_backtest_data,
    load_market_data,
)


@timer(human_str=True)
def run_market_neutral_strategy():
    # Initialize configurations
    config = BacktestConfig(
        initial_capital=100_000_000,
        rebalance_frequency=RebalanceFrequency.DAILY,
        weight_tolerance=0.1,
    )

    # Initialize backtester
    backtester = MarketNeutralBacktester(config)

    # Load and add pairs
    pairs_config = [
        ("QQQ", "SQQQ", 3.0),
        ("TLT", "TBT", 2.0),
        ("SMH", "SOXS", 3.0),
        ("XLF", "FAZ", 3.0),
        ("XLE", "ERY", 2.0),
    ]
    symbols = [x[0] for x in pairs_config] + [x[1] for x in pairs_config]

    # get market data
    start_date = "2017-01-01"
    end_date = "2024-01-01"
    market_data = load_market_data(symbols=symbols, start_date=start_date, end_date=end_date)

    # Prepare data for backtesting
    pairs_data = prepare_backtest_data(market_data, pairs_config)

    for (base, inverse, leverage), data in pairs_data.items():
        pair_config = PairConfig(
            base_symbol=base,
            inverse_symbol=inverse,
            inverse_leverage=leverage,
            base_data=data[base],
            inverse_data=data[inverse],
            pair_allocation=1.0 / len(pairs_data),
        )
        backtester.add_pair(pair_config, data)

    # Run backtest
    results = backtester.backtest()

    # Analyze results
    backtest_data = backtester.get_backtest_data()
    benchmark_returns = yf.download("SPY", start=start_date, end=end_date)[
        "Adj Close"
    ].pct_change()
    benchmark_returns.index = benchmark_returns.index.tz_localize("UTC")
    analysis = analyze_backtest_results(
        backtest_data=backtest_data,
        benchmark_returns=benchmark_returns,
        save_tearsheet=True,
        tearsheet_path=Path.cwd().as_posix() + "/backtest_tearsheet.html",
    )
    print("\nStrategy Summary Statistics:")
    print(analysis["summary_stats"].to_string())
    return results, analysis


if __name__ == "__main__":
    results, analysis = run_market_neutral_strategy()
