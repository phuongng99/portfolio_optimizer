"""portfolio_optimiser — modern portfolio theory in Python.

You are building this package step by step:

    data_loader.py       prices → returns → μ, Σ, correlation      (steps 2-4)
    portfolio_math.py    portfolio_return, volatility, Sharpe       (steps 5-6)
    optimizer.py         min-var, frontier, max-Sharpe, risk-parity (steps 7-12)
    constraints.py       PortfolioConstraints                       (step 13)
    costs.py             TransactionCostModel                       (step 14)
    backtester.py        rolling backtest + metrics                 (steps 15-16)
    report.py            risk decomposition + dashboard             (steps 17-18)
"""

# TODO (Step 1): give your engine a name, e.g. "Quantt Portfolio Optimiser v0.1"
ENGINE_NAME = "Quantt Portfolio Optimizer v0.1"
