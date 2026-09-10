"""data_loader.py - prices in, returns and moments out."""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.data_loader import (
    annualize_returns, annualize_volatility, calculate_correlation,
    calculate_covariance, calculate_returns, load_prices,
)


# ---------------------------------------------------------------- load_prices

def test_load_prices_indexes_by_parsed_date(real_prices):
    assert isinstance(real_prices.index, pd.DatetimeIndex)
    assert "Date" not in real_prices.columns, "Date should be the index, not a column"


def test_load_prices_is_chronological(real_prices):
    assert real_prices.index.is_monotonic_increasing


def test_load_prices_has_no_missing_values(real_prices):
    assert not real_prices.isna().any().any()


def test_load_prices_are_all_positive(real_prices):
    assert (real_prices > 0).all().all(), "a non-positive price breaks pct_change"


# ------------------------------------------------------------ calculate_returns

def test_returns_known_values(prices):
    r = calculate_returns(prices)
    # A: +100% then flat. B: flat throughout.
    assert r["A"].iloc[0] == pytest.approx(1.0)
    assert r["A"].iloc[1:].eq(0.0).all()
    assert r["B"].eq(0.0).all()


def test_returns_drops_exactly_one_row(prices):
    assert len(calculate_returns(prices)) == len(prices) - 1


def test_returns_first_date_is_second_price_date(prices):
    assert calculate_returns(prices).index[0] == prices.index[1]


def test_returns_reconstruct_prices(real_prices):
    """Compounding the returns back up must reproduce the price path."""
    r = calculate_returns(real_prices)
    rebuilt = real_prices.iloc[0] * (1 + r).cumprod()
    np.testing.assert_allclose(rebuilt.values, real_prices.iloc[1:].values, rtol=1e-10)


# ---------------------------------------------------------------- annualisation

def test_annualize_returns_uses_252_factor():
    r = pd.DataFrame({"X": [0.001] * 10})
    assert annualize_returns(r)["X"] == pytest.approx(0.001 * 252)


def test_annualize_volatility_uses_sqrt_252():
    rng = np.random.default_rng(1)
    r = pd.DataFrame({"X": rng.normal(0, 0.01, 5000)})
    expected = r["X"].std() * np.sqrt(252)
    assert annualize_volatility(r)["X"] == pytest.approx(expected)


def test_zero_volatility_for_constant_returns():
    r = pd.DataFrame({"X": [0.005] * 50})
    assert annualize_volatility(r)["X"] == pytest.approx(0.0)


# ------------------------------------------------------------------- covariance

def test_covariance_is_symmetric(real_prices):
    cov = calculate_covariance(calculate_returns(real_prices)).values
    np.testing.assert_allclose(cov, cov.T, atol=1e-15)


def test_covariance_is_positive_semidefinite(real_prices):
    """Every eigenvalue >= 0. If this fails, no optimiser can be trusted."""
    cov = calculate_covariance(calculate_returns(real_prices)).values
    assert np.linalg.eigvalsh(cov).min() > -1e-12


def test_covariance_diagonal_equals_variance(real_prices):
    r = calculate_returns(real_prices)
    cov = calculate_covariance(r)
    vol = annualize_volatility(r)
    np.testing.assert_allclose(np.diag(cov.values), (vol ** 2).values, rtol=1e-10)


# ------------------------------------------------------------------ correlation

def test_correlation_diagonal_is_one(real_prices):
    corr = calculate_correlation(calculate_returns(real_prices)).values
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)


def test_correlation_bounded_by_one(real_prices):
    corr = calculate_correlation(calculate_returns(real_prices)).values
    assert corr.min() >= -1.0 - 1e-12 and corr.max() <= 1.0 + 1e-12


def test_correlation_is_unannualised_covariance(real_prices):
    """corr_ij = cov_ij / (sigma_i * sigma_j) - the 252 factors cancel."""
    r = calculate_returns(real_prices)
    cov, vol = calculate_covariance(r).values, annualize_volatility(r).values
    np.testing.assert_allclose(calculate_correlation(r).values,
                               cov / np.outer(vol, vol), rtol=1e-10)
