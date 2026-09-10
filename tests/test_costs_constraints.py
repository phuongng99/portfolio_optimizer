"""costs.py and constraints.py - the friction and the guard rails."""

import numpy as np
import pytest

from portfolio_optimizer.constraints import PortfolioConstraints
from portfolio_optimizer.costs import TransactionCostModel


@pytest.fixture
def model():
    # 10 bps commission + 5 bps spread = 15 bps, $1 minimum
    return TransactionCostModel(commission_rate=0.001, spread_cost=0.0005,
                                min_commission=1.0)


# ============================== TransactionCostModel ==============================

def test_zero_trade_is_free(model):
    assert model.trade_cost(0) == 0.0, "not trading must never trigger the minimum"


def test_large_trade_uses_the_variable_rate(model):
    assert model.trade_cost(200_000) == pytest.approx(200_000 * 0.0015)


def test_small_trade_hits_the_minimum_floor(model):
    """$300 * 15bps = $0.45, below the $1 floor."""
    assert model.trade_cost(300) == pytest.approx(1.0)


def test_the_floor_crossover_point(model):
    """Below $666.67 the floor binds; above it, the rate does."""
    crossover = 1.0 / 0.0015
    assert model.trade_cost(crossover * 0.99) == pytest.approx(1.0)
    assert model.trade_cost(crossover * 1.01) > 1.0


def test_sells_cost_the_same_as_buys(model):
    assert model.trade_cost(-50_000) == pytest.approx(model.trade_cost(50_000))


def test_cost_is_monotonic_in_size(model):
    sizes = [1e3, 1e4, 1e5, 1e6]
    costs = [model.trade_cost(s) for s in sizes]
    assert costs == sorted(costs)


def test_commission_and_spread_are_additive():
    a = TransactionCostModel(0.002, 0.000, 0.0).trade_cost(100_000)
    b = TransactionCostModel(0.000, 0.002, 0.0).trade_cost(100_000)
    both = TransactionCostModel(0.001, 0.001, 0.0).trade_cost(100_000)
    assert a == pytest.approx(b) == pytest.approx(both)


# ------------------------------------------------------------------ rebalance_cost

def test_no_rebalance_costs_nothing(model):
    w = np.array([0.2] * 5)
    assert model.rebalance_cost(w, w, 1_000_000) == 0.0


def test_rebalance_cost_worked_example(model):
    """20/20/20/20/20 -> 40/30/10/10/10 on $1m.
    Trades: 200k, 100k, 100k, 100k, 100k = $600k at 15bps = $900."""
    cur = np.array([0.20] * 5)
    tgt = np.array([0.40, 0.30, 0.10, 0.10, 0.10])
    assert model.rebalance_cost(cur, tgt, 1_000_000) == pytest.approx(900.0)


def test_rebalance_cost_scales_with_portfolio_value(model):
    cur, tgt = np.array([0.5, 0.5]), np.array([0.7, 0.3])
    small = model.rebalance_cost(cur, tgt, 1_000_000)
    big = model.rebalance_cost(cur, tgt, 2_000_000)
    assert big == pytest.approx(2 * small)


def test_rebalance_cost_is_symmetric(model):
    """Going there and coming back cost the same."""
    a, b = np.array([0.3, 0.7]), np.array([0.6, 0.4])
    assert model.rebalance_cost(a, b, 500_000) == pytest.approx(
        model.rebalance_cost(b, a, 500_000))


def test_untouched_assets_are_not_charged_the_minimum():
    """Only assets that actually moved should pay. With 3 of 5 unchanged,
    a naive implementation would wrongly add 3 x min_commission."""
    m = TransactionCostModel(0.001, 0.0005, min_commission=1.0)
    cur = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    tgt = np.array([0.3, 0.1, 0.2, 0.2, 0.2])
    expected = 2 * (0.1 * 1_000_000 * 0.0015)
    assert m.rebalance_cost(cur, tgt, 1_000_000) == pytest.approx(expected)


def test_tiny_drift_is_dominated_by_the_minimum(model):
    """A 0.01% drift on $1m is a $100 trade -> the $1 floor bites, twice."""
    cur = np.array([0.5, 0.5])
    tgt = np.array([0.5001, 0.4999])
    assert model.rebalance_cost(cur, tgt, 1_000_000) == pytest.approx(2.0)


# ------------------------------------------------------------------------ turnover

def test_turnover_is_zero_when_nothing_moves(model):
    w = np.array([0.25] * 4)
    assert model.turn_over(w, w) == 0.0


def test_turnover_worked_example(model):
    """Absolute changes sum to 0.6; one-way turnover is half of that."""
    cur = np.array([0.20] * 5)
    tgt = np.array([0.40, 0.30, 0.10, 0.10, 0.10])
    assert model.turn_over(cur, tgt) == pytest.approx(0.30)


def test_complete_flip_is_100_percent_turnover(model):
    assert model.turn_over([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_turnover_never_exceeds_one_for_long_only(model):
    rng = np.random.default_rng(3)
    for _ in range(200):
        a, b = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
        assert 0.0 <= model.turn_over(a, b) <= 1.0 + 1e-12


def test_turnover_is_symmetric(model):
    a, b = np.array([0.1, 0.9]), np.array([0.6, 0.4])
    assert model.turn_over(a, b) == pytest.approx(model.turn_over(b, a))


def test_cost_equals_twice_turnover_times_rate_when_floor_is_inactive():
    """The identity traders actually use: cost = 2 x turnover x value x rate."""
    m = TransactionCostModel(0.001, 0.0005, min_commission=0.0)
    cur, tgt = np.array([0.2] * 5), np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    value = 1_000_000
    assert m.rebalance_cost(cur, tgt, value) == pytest.approx(
        2 * m.turn_over(cur, tgt) * value * 0.0015)


# ============================== PortfolioConstraints ==============================

def test_bounds_have_one_entry_per_asset():
    b = PortfolioConstraints(0.4, 0.05).get_bounds(6)
    assert len(b) == 6 and all(x == (0.05, 0.4) for x in b)


def test_default_bounds_are_long_only_unconstrained():
    assert PortfolioConstraints().get_bounds(3) == [(0.0, 1.0)] * 3


def test_equal_weight_is_feasible_under_defaults():
    assert PortfolioConstraints().is_feasible([0.25] * 4)


def test_weights_not_summing_to_one_are_rejected():
    c = PortfolioConstraints()
    assert not c.is_feasible([0.3, 0.3, 0.3])
    assert not c.is_feasible([0.5, 0.6])


def test_weight_above_the_cap_is_rejected():
    assert not PortfolioConstraints(max_weight=0.4).is_feasible([0.5, 0.5])


def test_weight_below_the_floor_is_rejected():
    assert not PortfolioConstraints(min_weight=0.1).is_feasible([0.95, 0.05])


def test_negative_weight_is_rejected_by_default():
    assert not PortfolioConstraints().is_feasible([1.2, -0.2])


def test_boundary_weights_are_accepted():
    """Exactly at the cap must pass - the check is inclusive."""
    assert PortfolioConstraints(max_weight=0.5, min_weight=0.0).is_feasible([0.5, 0.5])


def test_feasibility_tolerates_floating_point_dust():
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    assert PortfolioConstraints().is_feasible(w)


def test_bounds_and_feasibility_agree(three_asset_cov):
    """A portfolio optimised under the bounds must pass the feasibility check."""
    from portfolio_optimizer.optimizer import min_variance
    c = PortfolioConstraints(max_weight=0.5, min_weight=0.1)
    w = min_variance(three_asset_cov, bounds=c.get_bounds(3))
    assert c.is_feasible(w)
