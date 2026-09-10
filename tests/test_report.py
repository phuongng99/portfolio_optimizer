"""report.py - risk decomposition. Weights are not risk shares; these
tests pin down the difference."""

import numpy as np
import pytest

from portfolio_optimizer.optimizer import min_variance, risk_parity
from portfolio_optimizer.portfolio_math import portfolio_volatility
from portfolio_optimizer.report import (
    diversification_ratio, pct_risk_contribution, risk_contribution,
)


# -------------------------------------------------------------- risk_contribution

def test_risk_contributions_sum_to_portfolio_volatility(three_asset_cov):
    """Euler's theorem: the parts add up to the whole. Non-negotiable."""
    w = np.array([0.2, 0.3, 0.5])
    rc = risk_contribution(w, three_asset_cov)
    assert rc.sum() == pytest.approx(portfolio_volatility(w, three_asset_cov))


def test_single_asset_contributes_all_the_risk(two_asset_cov):
    rc = risk_contribution([1.0, 0.0], two_asset_cov)
    assert rc[0] == pytest.approx(0.20)
    assert rc[1] == pytest.approx(0.0)


def test_uncorrelated_equal_weights_by_hand(two_asset_cov):
    """sigma_p = sqrt(0.0325). RC_i = w_i^2 * var_i / sigma_p."""
    w = np.array([0.5, 0.5])
    sigma = np.sqrt(0.0325)
    expected = np.array([0.25 * 0.04, 0.25 * 0.09]) / sigma
    np.testing.assert_allclose(risk_contribution(w, two_asset_cov), expected)


def test_one_entry_per_asset(three_asset_cov):
    assert risk_contribution([0.2, 0.3, 0.5], three_asset_cov).shape == (3,)


def test_long_only_contributions_are_non_negative(three_asset_cov):
    rng = np.random.default_rng(5)
    for _ in range(200):
        w = rng.dirichlet(np.ones(3))
        assert risk_contribution(w, three_asset_cov).min() >= -1e-12


def test_riskier_asset_contributes_more_at_equal_weight(two_asset_cov):
    rc = risk_contribution([0.5, 0.5], two_asset_cov)
    assert rc[1] > rc[0], "the 30%-vol asset must dominate the 20%-vol one"


# ---------------------------------------------------------- pct_risk_contribution

def test_percentages_sum_to_one_hundred(three_asset_cov):
    pct = pct_risk_contribution([0.2, 0.3, 0.5], three_asset_cov)
    assert pct.sum() == pytest.approx(100.0)


def test_identical_assets_split_risk_evenly():
    cov = np.eye(4) * 0.04
    np.testing.assert_allclose(pct_risk_contribution([0.25] * 4, cov), 25.0)


def test_risk_share_is_not_the_same_as_weight_share(two_asset_cov):
    """The lesson of the whole module: 50/50 money is not 50/50 risk."""
    pct = pct_risk_contribution([0.5, 0.5], two_asset_cov)
    assert abs(pct[0] - 50.0) > 5.0


def test_risk_parity_portfolio_gives_equal_risk_shares(three_asset_cov):
    """Closing the loop with optimizer.py."""
    pct = pct_risk_contribution(risk_parity(three_asset_cov), three_asset_cov)
    np.testing.assert_allclose(pct, 100 / 3, atol=0.5)


def test_equal_weight_portfolio_does_not_give_equal_risk_shares(three_asset_cov):
    pct = pct_risk_contribution([1 / 3] * 3, three_asset_cov)
    assert np.abs(pct - 100 / 3).max() > 1.0


def test_percentages_are_scale_invariant(three_asset_cov):
    """Doubling every covariance doubles sigma but leaves the shares alone."""
    w = [0.2, 0.3, 0.5]
    np.testing.assert_allclose(pct_risk_contribution(w, three_asset_cov),
                               pct_risk_contribution(w, three_asset_cov * 2))


# -------------------------------------------------------- diversification_ratio

def test_ratio_is_one_for_a_single_asset(two_asset_cov):
    assert diversification_ratio([1.0, 0.0], two_asset_cov) == pytest.approx(1.0)


def test_ratio_is_one_when_assets_are_perfectly_correlated():
    """Correlation 1 means no diversification benefit at all."""
    cov = np.array([[0.04, 0.04], [0.04, 0.04]])
    assert diversification_ratio([0.5, 0.5], cov) == pytest.approx(1.0)


def test_ratio_exceeds_one_when_assets_are_uncorrelated(two_asset_cov):
    assert diversification_ratio([0.5, 0.5], two_asset_cov) > 1.0


def test_ratio_known_value(two_asset_cov):
    """(0.5*0.2 + 0.5*0.3) / sqrt(0.0325)"""
    assert diversification_ratio([0.5, 0.5], two_asset_cov) == pytest.approx(
        0.25 / np.sqrt(0.0325))


def test_ratio_is_never_below_one_for_long_only(three_asset_cov):
    rng = np.random.default_rng(9)
    for _ in range(300):
        w = rng.dirichlet(np.ones(3))
        assert diversification_ratio(w, three_asset_cov) >= 1.0 - 1e-9


def test_ratio_grows_as_correlation_falls():
    def ratio(rho):
        cov = np.array([[0.04, rho * 0.04], [rho * 0.04, 0.04]])
        return diversification_ratio([0.5, 0.5], cov)
    assert ratio(0.9) < ratio(0.5) < ratio(0.0) < ratio(-0.5)


def test_ratio_on_real_data_is_meaningful(real_prices):
    from portfolio_optimizer.data_loader import calculate_covariance, calculate_returns
    cov = calculate_covariance(calculate_returns(real_prices)).values
    n = real_prices.shape[1]
    assert 1.0 <= diversification_ratio(np.ones(n) / n, cov) < 3.0


def test_min_variance_does_not_maximise_the_diversification_ratio(three_asset_cov):
    """A common misconception, worth pinning down. Minimising volatility and
    maximising the diversification ratio are DIFFERENT objectives: min-variance
    concentrates in the low-vol asset, which can lower the ratio below plain
    equal weight. If this ever flips, someone has confused the two."""
    n = 3
    mv = diversification_ratio(min_variance(three_asset_cov), three_asset_cov)
    eq = diversification_ratio(np.ones(n) / n, three_asset_cov)
    assert mv < eq
    assert mv > 1.0, "it should still be diversified, just not maximally so"
