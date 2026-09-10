"""Data loading and return statistics."""

import numpy as np
import pandas as pd


def load_prices(filepath):
    """Read a CSV with a Date column and one column per asset. Returns a
    DataFrame indexed by date, columns=tickers, values=close prices.

    TODO (Step 2):
        return pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    """
    return pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    pass


def calculate_returns(prices):
    """Daily percentage returns. Drops the first row (undefined).

    TODO (Step 2): prices.pct_change().dropna().
    """
    return prices.pct_change().dropna()
    pass


def annualize_returns(daily_returns):
    """Annualise mean daily returns to yearly expected returns.

    TODO (Step 3): daily_returns.mean() * 252.
    """
    return daily_returns.mean()*252
    pass


def annualize_volatility(daily_returns):
    """Annualise daily standard deviations to yearly volatility.

    Variance scales linearly with time; volatility scales with √time.

    TODO (Step 3): daily_returns.std() * np.sqrt(252).
    """
    return daily_returns.std()*np.sqrt(252)
    pass


def calculate_covariance(daily_returns):
    """Annualised covariance matrix.

    TODO (Step 4): daily_returns.cov() * 252.
    """
    return daily_returns.cov()*252
    pass


def calculate_correlation(daily_returns):
    """Correlation matrix (unit-less; no annualisation).

    TODO (Step 4): daily_returns.corr().
    """
    return daily_returns.corr()
    pass


def print_summary(daily_returns):
    """Print per-asset annualised return + vol and the correlation matrix."""
    mu = annualize_returns(daily_returns)
    sigma = annualize_volatility(daily_returns)
    corr = calculate_correlation(daily_returns)
    print("Per-asset annualised stats:")
    for name in mu.index:
        print(f"  {name:>6}  μ={mu[name]*100:>+6.2f}%   σ={sigma[name]*100:>5.2f}%")
    print("\nCorrelation matrix:")
    print(corr.round(2))
