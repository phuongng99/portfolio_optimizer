"""portfolio_math.py - the three primitives everything else is built on."""

import numpy as np
import pytest

from portfolio_optimizer.portfolio_math import (
    portfolio_return, portfolio_volatility, sharpe_ratio,
)


# -------------------------------------------------------------- portfolio_return

def test_return_is_weighted_average():
    assert portfolio_return([0.5, 0.5], [0.10, 0.20]) == pytest.approx(0.15)


def test_return_full_weight_on_one_asset():
    assert portfolio_return([0.0, 1.0, 0.0], [0.05, 0.11, 0.20]) == pytest.approx(0.11)


def test_return_accepts_lists_and_arrays_alike():
    a = portfolio_return([0.3, 0.7], [0.08, 0.12])
    b = portfolio_return(np.array([0.3, 0.7]), np.array([0.08, 0.12]))
    assert a == pytest.approx(b)


# ---------------------------------------------------------- portfolio_volatility

def test_volatility_single_asset_equals_its_own_vol(two_asset_cov):
    assert portfolio_volatility([1.0, 0.0], two_asset_cov) == pytest.approx(0.20)
    assert portfolio_volatility([0.0, 1.0], two_asset_cov) == pytest.approx(0.30)


def test_volatility_uncorrelated_pair_by_hand(two_asset_cov):
    """sqrt(0.5^2*0.04 + 0.5^2*0.09) = sqrt(0.0325)"""
    assert portfolio_volatility([0.5, 0.5], two_asset_cov) == pytest.approx(np.sqrt(0.0325))


def test_volatility_is_never_negative(three_asset_cov):
    rng = np.random.default_rng(7)
    for _ in range(200):
        w = rng.dirichlet(np.ones(3))
        assert portfolio_volatility(w, three_asset_cov) >= 0.0


def test_diversification_beats_the_weighted_average(three_asset_cov):
    """With correlation < 1, blending must lower risk below the naive average."""
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    naive = float(w @ np.sqrt(np.diag(three_asset_cov)))
    assert portfolio_volatility(w, three_asset_cov) < naive


def test_perfectly_negatively_correlated_pair_can_be_riskless():
    """Two assets, same vol, correlation -1, equal weights -> zero risk."""
    cov = np.array([[0.04, -0.04], [-0.04, 0.04]])
    assert portfolio_volatility([0.5, 0.5], cov) == pytest.approx(0.0, abs=1e-12)


# -------------------------------------------------------------------- sharpe_ratio

def test_sharpe_known_value(two_asset_cov):
    """(0.15 - 0.03) / sqrt(0.0325)"""
    got = sharpe_ratio([0.5, 0.5], [0.10, 0.20], two_asset_cov, risk_free_rate=0.03)
    assert got == pytest.approx(0.12 / np.sqrt(0.0325))


def test_sharpe_default_risk_free_is_zero(two_asset_cov):
    a = sharpe_ratio([0.5, 0.5], [0.10, 0.20], two_asset_cov)
    b = sharpe_ratio([0.5, 0.5], [0.10, 0.20], two_asset_cov, risk_free_rate=0.0)
    assert a == pytest.approx(b)


def test_sharpe_is_negative_when_return_trails_risk_free(two_asset_cov):
    assert sharpe_ratio([0.5, 0.5], [0.01, 0.01], two_asset_cov, risk_free_rate=0.05) < 0


def test_sharpe_returns_zero_on_zero_volatility():
    """Guard clause: must not raise ZeroDivisionError."""
    assert sharpe_ratio([1.0], [0.10], np.array([[0.0]]), 0.02) == 0


def test_sharpe_is_scale_invariant_in_time_units(three_asset_mu, three_asset_cov):
    """Doubling the horizon scales mu by 2 and vol by sqrt(2): Sharpe scales by sqrt(2)."""
    w = np.array([0.2, 0.3, 0.5])
    base = sharpe_ratio(w, three_asset_mu, three_asset_cov)
    scaled = sharpe_ratio(w, three_asset_mu * 2, three_asset_cov * 2)
    assert scaled == pytest.approx(base * np.sqrt(2))
