"""optimizer.py - the portfolios themselves. Optimisers are easy to get
subtly wrong, so these check mathematical properties, not just 'it ran'."""

import numpy as np
import pytest

from portfolio_optimizer.optimizer import (
    compare_strategies, efficient_frontier, equal_weighted_portfolio,
    inverse_volatility, max_sharpe, min_variance, min_vol_target_return,
    risk_budget, risk_parity
)
from portfolio_optimizer.portfolio_math import (
    portfolio_return, portfolio_volatility, sharpe_ratio,
)

TOL = 1e-6


def _valid(w, n):
    assert w is not None, "optimiser failed to converge"
    assert len(w) == n
    assert w.sum() == pytest.approx(1.0, abs=1e-6), "weights must sum to 1"
    assert w.min() >= -1e-9, "long-only: no negative weights"


# --------------------------------------------------------- equal_weighted_portfolio

def test_equal_weight_is_uniform_and_sums_to_one():
    w = equal_weighted_portfolio(5)
    np.testing.assert_allclose(w, 0.2)
    assert w.sum() == pytest.approx(1.0)


def test_equal_weight_single_asset():
    np.testing.assert_allclose(equal_weighted_portfolio(1), [1.0])


# -------------------------------------------------------------------- min_variance

def test_min_variance_matches_closed_form(two_asset_cov):
    """For uncorrelated assets, w_i is proportional to 1/variance_i.

    NOTE: SLSQP stops at its default ftol, so it lands ~5e-4 away from the
    exact 0.6923/0.3077. That is convergence slack, not a modelling error.
    Tighten it with options={'ftol': 1e-12} in min_variance if you need more.
    """
    w = min_variance(two_asset_cov)
    _valid(w, 2)
    inv = 1 / np.array([0.04, 0.09])
    np.testing.assert_allclose(w, inv / inv.sum(), atol=1e-3)


def test_min_variance_beats_every_random_portfolio(three_asset_cov):
    """The defining property: nothing feasible has lower volatility."""
    w = min_variance(three_asset_cov)
    _valid(w, 3)
    best = portfolio_volatility(w, three_asset_cov)
    rng = np.random.default_rng(42)
    for _ in range(500):
        cand = rng.dirichlet(np.ones(3))
        assert portfolio_volatility(cand, three_asset_cov) >= best - TOL


def test_min_variance_respects_a_weight_cap(three_asset_cov):
    bounds = [(0.0, 0.4)] * 3
    w = min_variance(three_asset_cov, bounds=bounds)
    _valid(w, 3)
    assert w.max() <= 0.4 + 1e-6


def test_min_variance_capped_is_riskier_than_uncapped(three_asset_cov):
    """Adding a constraint can never improve the objective."""
    free = portfolio_volatility(min_variance(three_asset_cov), three_asset_cov)
    capped = portfolio_volatility(min_variance(three_asset_cov, [(0.0, 0.4)] * 3),
                                  three_asset_cov)
    assert capped >= free - TOL


# ------------------------------------------------------------ min_vol_target_return

def test_target_return_constraint_is_actually_hit(three_asset_mu, three_asset_cov):
    for target in [0.07, 0.09, 0.11, 0.13]:
        w = min_vol_target_return(three_asset_cov, three_asset_mu, target)
        _valid(w, 3)
        assert portfolio_return(w, three_asset_mu) == pytest.approx(target, abs=1e-6)


def test_target_return_at_the_extremes_picks_the_corner(three_asset_mu, three_asset_cov):
    """Demanding the highest possible return forces 100% into the best asset."""
    w = min_vol_target_return(three_asset_cov, three_asset_mu, three_asset_mu.max())
    _valid(w, 3)
    assert w[np.argmax(three_asset_mu)] == pytest.approx(1.0, abs=1e-4)


def test_unreachable_target_does_not_return_a_bogus_portfolio(three_asset_mu, three_asset_cov):
    """No long-only mix can return 50% here. Expect None, not nonsense."""
    w = min_vol_target_return(three_asset_cov, three_asset_mu, 0.50)
    assert w is None or portfolio_return(w, three_asset_mu) == pytest.approx(0.50, abs=1e-6)


# ---------------------------------------------------------------------- max_sharpe

def test_max_sharpe_beats_every_random_portfolio(three_asset_mu, three_asset_cov):
    w = max_sharpe(three_asset_cov, three_asset_mu, risk_free_rate=0.02)
    _valid(w, 3)
    best = sharpe_ratio(w, three_asset_mu, three_asset_cov, 0.02)
    rng = np.random.default_rng(11)
    for _ in range(500):
        cand = rng.dirichlet(np.ones(3))
        assert sharpe_ratio(cand, three_asset_mu, three_asset_cov, 0.02) <= best + TOL


def test_max_sharpe_beats_the_other_named_strategies(three_asset_mu, three_asset_cov):
    ms = max_sharpe(three_asset_cov, three_asset_mu, 0.02)
    best = sharpe_ratio(ms, three_asset_mu, three_asset_cov, 0.02)
    for other in [equal_weighted_portfolio(3),
                  min_variance(three_asset_cov),
                  risk_parity(three_asset_cov)]:
        assert sharpe_ratio(other, three_asset_mu, three_asset_cov, 0.02) <= best + TOL


def test_max_sharpe_prefers_the_dominant_asset():
    """Same vol, uncorrelated, but asset 1 returns far more - it should win."""
    cov = np.eye(2) * 0.04
    w = max_sharpe(cov, np.array([0.02, 0.20]), risk_free_rate=0.0)
    _valid(w, 2)
    assert w[1] > w[0]


def test_max_sharpe_respects_bounds(three_asset_mu, three_asset_cov):
    w = max_sharpe(three_asset_cov, three_asset_mu, 0.02, bounds=[(0.1, 0.5)] * 3)
    _valid(w, 3)
    assert w.min() >= 0.1 - 1e-6 and w.max() <= 0.5 + 1e-6


# --------------------------------------------------------------------- risk_parity

def test_risk_parity_equalises_risk_contributions(three_asset_cov):
    """The whole point: every asset supplies the same share of total risk."""
    w = risk_parity(three_asset_cov)
    _valid(w, 3)
    sigma = portfolio_volatility(w, three_asset_cov)
    rc = w * (three_asset_cov @ w) / sigma
    pct = rc / rc.sum()
    np.testing.assert_allclose(pct, 1 / 3, atol=1e-3)


def test_risk_parity_is_equal_weight_when_assets_are_identical():
    cov = np.eye(4) * 0.04
    np.testing.assert_allclose(risk_parity(cov), 0.25, atol=1e-3)


def test_risk_parity_underweights_the_riskier_asset(two_asset_cov):
    """20% vol vs 30% vol, uncorrelated -> less money in the 30% one."""
    w = risk_parity(two_asset_cov)
    _valid(w, 2)
    assert w[0] > w[1]


def test_risk_parity_sits_between_equal_weight_and_min_variance(three_asset_cov):
    """A well-known ordering of the three long-only volatilities."""
    vols = {k: portfolio_volatility(v, three_asset_cov) for k, v in {
        "eq": equal_weighted_portfolio(3),
        "rp": risk_parity(three_asset_cov),
        "mv": min_variance(three_asset_cov)}.items()}
    assert vols["mv"] <= vols["rp"] + TOL <= vols["eq"] + 2 * TOL


# --------------------------------------------------------------------- risk_budget

def _risk_contributions(w, cov):
    """Each asset's share of total portfolio risk. Shares sum to 1."""
    return w * (cov @ w) / (w @ cov @ w)


def test_risk_budget_equal_budgets_equalise_risk(three_asset_cov):
    """budgets=None means equal shares, i.e. risk parity."""
    w = risk_budget(three_asset_cov)
    _valid(w, 3)
    np.testing.assert_allclose(_risk_contributions(w, three_asset_cov),
                               1 / 3, atol=1e-8)


def test_risk_budget_honours_custom_budgets(three_asset_cov):
    """The defining property: asset i supplies exactly budgets[i] of total risk."""
    budgets = np.array([0.5, 0.3, 0.2])
    w = risk_budget(three_asset_cov, budgets)
    _valid(w, 3)
    np.testing.assert_allclose(_risk_contributions(w, three_asset_cov),
                               budgets, atol=1e-8)


def test_risk_budget_moves_off_its_starting_point(three_asset_cov):
    """Regression: the solver once returned its inverse-volatility seed untouched.

    That failure was silent - the seed is itself a plausible normalised
    long-only vector - so assert the iteration changed something.
    """
    w = risk_budget(three_asset_cov)
    assert np.abs(w - inverse_volatility(three_asset_cov)).max() > 1e-3


def test_risk_budget_agrees_with_risk_parity(three_asset_cov):
    """Same problem, two independent solvers: CCD here, SLSQP in risk_parity."""
    np.testing.assert_allclose(risk_budget(three_asset_cov),
                               risk_parity(three_asset_cov), atol=1e-3)


def test_risk_budget_is_equal_weight_when_assets_are_identical():
    cov = np.eye(4) * 0.04
    np.testing.assert_allclose(risk_budget(cov), 0.25, atol=1e-8)


def test_risk_budget_raising_a_budget_raises_the_weight(three_asset_cov):
    """Monotonicity: ask an asset for more risk, it gets more money."""
    base = risk_budget(three_asset_cov, np.full(3, 1 / 3))
    tilted = risk_budget(three_asset_cov, np.array([0.6, 0.2, 0.2]))
    assert tilted[0] > base[0]


def test_risk_budget_only_relative_budgets_matter(three_asset_cov):
    """Weights are normalised at the end, so scaling every budget changes nothing."""
    np.testing.assert_allclose(risk_budget(three_asset_cov, np.array([0.5, 0.3, 0.2])),
                               risk_budget(three_asset_cov, np.array([5.0, 3.0, 2.0])),
                               atol=1e-8)


def test_risk_budget_on_real_data(real_prices):
    """Many correlated assets - the case where a lazy solver drifts."""
    from portfolio_optimizer.data_loader import (
        calculate_covariance, calculate_returns)
    cov = calculate_covariance(calculate_returns(real_prices)).values
    n = cov.shape[0]
    w = risk_budget(cov)
    _valid(w, n)
    np.testing.assert_allclose(_risk_contributions(w, cov), 1 / n, atol=1e-8)


# --------------------------------------------------------------- efficient_frontier

def test_frontier_returns_the_documented_shape(three_asset_mu, three_asset_cov):
    f = efficient_frontier(three_asset_cov, three_asset_mu, n_points=25)
    assert set(f) == {"Volatilities", "Returns", "Weights"}
    assert len(f["Volatilities"]) == len(f["Returns"]) == len(f["Weights"])
    assert f["Weights"].shape[1] == 3


def test_frontier_returns_are_increasing(three_asset_mu, three_asset_cov):
    f = efficient_frontier(three_asset_cov, three_asset_mu, n_points=30)
    assert np.all(np.diff(f["Returns"]) > -1e-9)


def test_frontier_is_convex_left_to_right(three_asset_cov, three_asset_mu):
    """Volatility falls to the min-variance point, then rises. Never zig-zags."""
    f = efficient_frontier(three_asset_cov, three_asset_mu, n_points=40)
    vol = f["Volatilities"]
    turning = int(np.argmin(vol))
    assert np.all(np.diff(vol[:turning + 1]) <= 1e-6)
    assert np.all(np.diff(vol[turning:]) >= -1e-6)


def test_frontier_low_point_equals_min_variance(three_asset_mu, three_asset_cov):
    f = efficient_frontier(three_asset_cov, three_asset_mu, n_points=200)
    mv = portfolio_volatility(min_variance(three_asset_cov), three_asset_cov)
    assert f["Volatilities"].min() == pytest.approx(mv, abs=1e-4)


def test_no_frontier_point_beats_max_sharpe(three_asset_mu, three_asset_cov):
    """The tangency portfolio must dominate the whole curve."""
    f = efficient_frontier(three_asset_cov, three_asset_mu, n_points=60)
    best = sharpe_ratio(max_sharpe(three_asset_cov, three_asset_mu, 0.02),
                        three_asset_mu, three_asset_cov, 0.02)
    for w in f["Weights"]:
        assert sharpe_ratio(w, three_asset_mu, three_asset_cov, 0.02) <= best + 1e-4


def test_every_frontier_weight_vector_is_valid(three_asset_mu, three_asset_cov):
    f = efficient_frontier(three_asset_cov, three_asset_mu, n_points=30)
    for w in f["Weights"]:
        _valid(w, 3)


# -------------------------------------------------------------- compare_strategies

def test_compare_strategies_returns_all_strategies(three_asset_mu, three_asset_cov):
    out = compare_strategies(three_asset_cov, three_asset_mu)
    assert out is not None, "compare_strategies must return its results"
    assert {d["name"] for d in out} == {"Equal-Weight", "Min-Variance",
                                        "Max-Sharpe", "Risk-Parity",
                                        "Risk-Budget", "Inverse-Volatility"}


def test_compare_strategies_metrics_are_self_consistent(three_asset_mu, three_asset_cov):
    """Recompute each row's numbers from its own weights - they must agree."""
    for d in compare_strategies(three_asset_cov, three_asset_mu, risk_free_rate=0.02):
        w = d["weights"]
        assert d["returns"] == pytest.approx(portfolio_return(w, three_asset_mu))
        assert d["volatility"] == pytest.approx(portfolio_volatility(w, three_asset_cov))
        assert d["sharpe"] == pytest.approx(
            sharpe_ratio(w, three_asset_mu, three_asset_cov, 0.02))


def test_compare_strategies_ranking_is_sane(three_asset_mu, three_asset_cov):
    """Min-Variance has the lowest vol; Max-Sharpe has the highest Sharpe."""
    rows = {d["name"]: d for d in compare_strategies(three_asset_cov, three_asset_mu)}
    assert rows["Min-Variance"]["volatility"] == pytest.approx(
        min(d["volatility"] for d in rows.values()), abs=TOL)
    assert rows["Max-Sharpe"]["sharpe"] == pytest.approx(
        max(d["sharpe"] for d in rows.values()), abs=TOL)


def test_compare_strategies_on_real_data(real_prices):
    from portfolio_optimizer.data_loader import (
        annualize_returns, calculate_covariance, calculate_returns)
    r = calculate_returns(real_prices)
    out = compare_strategies(calculate_covariance(r).values,
                             annualize_returns(r).values)
    assert len(out) == 6
    for d in out:
        _valid(np.asarray(d["weights"]), real_prices.shape[1])


def test_compare_strategies_never_reports_weights_that_break_bounds(three_asset_mu,
                                                                    three_asset_cov):
    """Equal-Weight, Risk-Budget and Inverse-Volatility cannot honour bounds.

    They are dropped rather than returned in violation. A uniform cap can never
    catch this - any feasible cap is at least 1/n, which equal weight meets
    exactly - so use an uneven one, where 1/3 breaches the 20% limit on asset 1.
    """
    bounds = [(0.0, 0.5), (0.0, 0.2), (0.0, 0.6)]
    out = compare_strategies(three_asset_cov, three_asset_mu, bounds=bounds)
    assert {d["name"] for d in out} == {"Min-Variance", "Max-Sharpe", "Risk-Parity"}
    for d in out:
        w = np.asarray(d["weights"])
        for wi, (lo, hi) in zip(w, bounds):
            assert lo - 1e-6 <= wi <= hi + 1e-6, d["name"]


# ------------------------------------------------------------------ input handling

@pytest.mark.parametrize("fn", [min_variance, risk_parity, risk_budget,
                                inverse_volatility])
def test_a_plain_nested_list_is_accepted(fn):
    """These used to read .shape before np.asarray, so a list raised AttributeError.

    Only validity is asserted: the four strategies legitimately disagree on the
    answer (min_variance weights by 1/variance, the rest by 1/volatility).
    """
    w = fn([[0.04, 0.0], [0.0, 0.09]])
    _valid(np.asarray(w), 2)
    assert w[0] > w[1], "the 20% vol asset should outweigh the 30% one"


@pytest.mark.parametrize("fn", [inverse_volatility, risk_budget])
def test_zero_variance_asset_raises_instead_of_returning_nan(fn):
    """A cash row has zero variance; 1/sigma is inf and poisons every weight."""
    with pytest.raises(ValueError, match="positive variance"):
        fn(np.diag([0.04, 0.09, 0.0]))


@pytest.mark.parametrize("fn", [efficient_frontier, compare_strategies])
def test_swapping_cov_and_mu_raises(fn, three_asset_mu, three_asset_cov):
    """Both arguments are float arrays, so a swap would otherwise pass silently."""
    with pytest.raises(ValueError, match="square 2-D"):
        fn(three_asset_mu, three_asset_cov)
