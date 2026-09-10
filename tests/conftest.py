"""Shared fixtures. Everything here is deterministic - no random seeds to drift."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the package importable when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stock_prices.csv")


@pytest.fixture
def two_asset_cov():
    """Uncorrelated pair: vol 20% and 30%. Hand-computable answers.

    Min-variance weights are proportional to 1/variance:
        w1 = (1/0.04) / (1/0.04 + 1/0.09) = 0.6923...
    """
    return np.array([[0.04, 0.00],
                     [0.00, 0.09]])


@pytest.fixture
def three_asset_cov():
    """Mild positive correlation, distinct vols - stresses the optimisers."""
    vols = np.array([0.15, 0.20, 0.25])
    corr = np.array([[1.0, 0.3, 0.2],
                     [0.3, 1.0, 0.4],
                     [0.2, 0.4, 1.0]])
    return np.outer(vols, vols) * corr


@pytest.fixture
def three_asset_mu():
    return np.array([0.06, 0.10, 0.14])


@pytest.fixture
def prices():
    """Two synthetic assets with exactly known behaviour.

    A doubles over 4 steps via +100%, 0%, 0%, 0%.
    B is flat.
    """
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame({"A": [100.0, 200.0, 200.0, 200.0, 200.0],
                         "B": [50.0, 50.0, 50.0, 50.0, 50.0]}, index=idx)


@pytest.fixture
def real_prices():
    """The project's own CSV - used for integration-level checks."""
    return pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")


@pytest.fixture
def month_spanning_prices():
    """Business days across a month AND quarter boundary, so the
    rebalance-schedule logic has something real to fire on."""
    idx = pd.bdate_range("2024-02-15", "2024-04-15")
    rng = np.random.default_rng(0)
    n = len(idx)
    data = {}
    for k, drift in enumerate([0.0004, 0.0002, 0.0006]):
        steps = rng.normal(drift, 0.01, n)
        data[f"S{k}"] = 100.0 * np.cumprod(1 + steps)
    return pd.DataFrame(data, index=idx)
