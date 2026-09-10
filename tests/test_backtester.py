"""backtestter.py - the simulation. The look-ahead tests matter most:
a backtest that peeks at the future looks brilliant and loses real money."""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.backtestter import PortfolioBacktester
from portfolio_optimizer.costs import TransactionCostModel
from portfolio_optimizer.optimizer import equal_weighted_portfolio, max_sharpe, min_variance


def eq_weight_optimizer(mu, cov):
    return equal_weighted_portfolio(len(mu))


@pytest.fixture
def bt():
    return PortfolioBacktester(eq_weight_optimizer, rebalance_freq="M",
                               starting_value=100_000.0, lookback_days=20)


# ------------------------------------------------------------- _is_rebalance_day

def test_first_day_is_always_a_rebalance_day(bt):
    assert bt._is_rebalance_day(None, pd.Timestamp("2024-03-15"))


def test_monthly_ignores_days_inside_a_month(bt):
    assert not bt._is_rebalance_day(pd.Timestamp("2024-03-04"),
                                    pd.Timestamp("2024-03-29"))


def test_monthly_fires_on_the_month_boundary(bt):
    assert bt._is_rebalance_day(pd.Timestamp("2024-03-29"),
                                pd.Timestamp("2024-04-01"))


def test_monthly_fires_across_a_weekend_gap(bt):
    """Friday 31 Mar to Monday 3 Apr - no calendar-day arithmetic required."""
    assert bt._is_rebalance_day(pd.Timestamp("2023-03-31"),
                                pd.Timestamp("2023-04-03"))


def test_monthly_fires_across_the_year_end(bt):
    assert bt._is_rebalance_day(pd.Timestamp("2023-12-29"),
                                pd.Timestamp("2024-01-02"))


@pytest.mark.parametrize("prev,this,expected", [
    ("2024-01-31", "2024-02-01", False),   # Jan and Feb are both Q1
    ("2024-03-28", "2024-04-01", True),    # Q1 -> Q2
    ("2024-06-28", "2024-07-01", True),    # Q2 -> Q3
    ("2024-09-30", "2024-10-01", True),    # Q3 -> Q4
    ("2024-12-31", "2025-01-02", True),    # Q4 -> Q1
    ("2024-07-01", "2024-08-15", False),   # Jul and Aug are both Q3
])
def test_quarterly_boundaries(prev, this, expected):
    b = PortfolioBacktester(eq_weight_optimizer, rebalance_freq="Q")
    assert b._is_rebalance_day(pd.Timestamp(prev), pd.Timestamp(this)) is expected


def test_daily_frequency_fires_every_day():
    b = PortfolioBacktester(eq_weight_optimizer, rebalance_freq="D")
    assert b._is_rebalance_day(pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05"))


def test_quarter_grouping_covers_all_twelve_months():
    b = PortfolioBacktester(eq_weight_optimizer, rebalance_freq="Q")
    q = [(m - 1) // 3 for m in range(1, 13)]
    assert q == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert not b._is_rebalance_day(pd.Timestamp("2024-01-15"), pd.Timestamp("2024-03-15"))


# ------------------------------------------------------------------------- run()

def test_equity_curve_has_one_row_per_return_day(bt, month_spanning_prices):
    bt.run(month_spanning_prices)
    assert len(bt.equity_curve) == len(month_spanning_prices) - 1


def test_equity_curve_is_a_dataframe_indexed_by_date(bt, month_spanning_prices):
    bt.run(month_spanning_prices)
    assert isinstance(bt.equity_curve, pd.DataFrame)
    assert list(bt.equity_curve.columns) == ["value"]
    assert isinstance(bt.equity_curve.index, pd.DatetimeIndex)
    assert bt.equity_curve.index.is_monotonic_increasing


def test_equity_curve_dates_match_the_return_dates(bt, month_spanning_prices):
    bt.run(month_spanning_prices)
    expected = month_spanning_prices.index[1:]
    pd.testing.assert_index_equal(bt.equity_curve.index, expected, check_names=False)


def test_flat_market_leaves_the_value_untouched():
    """No price movement, no costs -> the balance never budges."""
    idx = pd.bdate_range("2024-01-01", periods=60)
    flat = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    b = PortfolioBacktester(eq_weight_optimizer, lookback_days=10, starting_value=100_000)
    b.run(flat)
    np.testing.assert_allclose(b.equity_curve["value"].values, 100_000.0)


def test_known_growth_is_reproduced_exactly():
    """Both assets rise 1% a day for 10 days. Any weights give 1.01^10."""
    idx = pd.bdate_range("2024-01-01", periods=11)
    path = 100.0 * 1.01 ** np.arange(11)
    prices = pd.DataFrame({"A": path, "B": path}, index=idx)
    b = PortfolioBacktester(eq_weight_optimizer, lookback_days=5, starting_value=1000.0)
    b.run(prices)
    assert b.equity_curve["value"].iloc[-1] == pytest.approx(1000.0 * 1.01 ** 10)


def test_no_rebalancing_before_the_lookback_is_satisfied(month_spanning_prices):
    b = PortfolioBacktester(eq_weight_optimizer, rebalance_freq="D", lookback_days=15)
    b.run(month_spanning_prices)
    n_days = len(month_spanning_prices) - 1
    assert len(b.rebalance_log) == n_days - 15


def test_lookback_longer_than_the_data_means_no_rebalancing(month_spanning_prices):
    b = PortfolioBacktester(eq_weight_optimizer, lookback_days=10_000)
    b.run(month_spanning_prices)
    assert b.rebalance_log == []
    assert b.total_cost == 0.0


def test_rebalance_log_records_valid_weights(month_spanning_prices):
    b = PortfolioBacktester(lambda mu, cov: min_variance(cov),
                            rebalance_freq="M", lookback_days=15)
    b.run(month_spanning_prices)
    assert len(b.rebalance_log) >= 1
    for entry in b.rebalance_log:
        assert {"date", "weights"} <= set(entry)
        w = np.asarray(entry["weights"])
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert w.min() >= -1e-9


def test_rebalance_dates_are_month_starts(month_spanning_prices):
    b = PortfolioBacktester(eq_weight_optimizer, rebalance_freq="M", lookback_days=5)
    b.run(month_spanning_prices)
    months = [e["date"].month for e in b.rebalance_log]
    assert len(months) == len(set(months)), "at most one rebalance per month"


# ------------------------------------------------------- LOOK-AHEAD BIAS (critical)

def test_optimizer_never_sees_the_current_or_future_day(month_spanning_prices):
    """Capture the window handed to the optimiser and prove it is strictly
    in the past. This is the single most important test in the file."""
    from portfolio_optimizer.data_loader import calculate_returns
    returns = calculate_returns(month_spanning_prices)
    seen = []

    def spy(mu, cov):
        seen.append(np.asarray(mu).copy())
        return equal_weighted_portfolio(len(mu))

    lookback = 15
    b = PortfolioBacktester(spy, rebalance_freq="D", lookback_days=lookback)
    b.run(month_spanning_prices)

    for k, entry in enumerate(b.rebalance_log):
        i = returns.index.get_loc(entry["date"])
        past = returns.iloc[i - lookback:i]
        expected_mu = (past.mean() * 252).values
        np.testing.assert_allclose(seen[k], expected_mu, rtol=1e-12,
                                   err_msg="optimiser was fed data it could not have known")


def test_future_prices_cannot_change_earlier_equity_values(month_spanning_prices):
    """Truncate the data and the overlapping part of the curve must be identical."""
    b1 = PortfolioBacktester(lambda mu, cov: min_variance(cov), lookback_days=15)
    b2 = PortfolioBacktester(lambda mu, cov: min_variance(cov), lookback_days=15)
    cut = len(month_spanning_prices) - 10
    b1.run(month_spanning_prices)
    b2.run(month_spanning_prices.iloc[:cut])
    overlap = b2.equity_curve.index
    np.testing.assert_allclose(b1.equity_curve.loc[overlap, "value"].values,
                               b2.equity_curve["value"].values, rtol=1e-12)


def test_shuffling_future_returns_leaves_the_past_alone(month_spanning_prices):
    """Scramble the last 10 days. Everything before them must be untouched."""
    b1 = PortfolioBacktester(eq_weight_optimizer, lookback_days=15)
    b1.run(month_spanning_prices)
    tampered = month_spanning_prices.copy()
    tampered.iloc[-10:] = tampered.iloc[-10:].values * 1.5
    b2 = PortfolioBacktester(eq_weight_optimizer, lookback_days=15)
    b2.run(tampered)
    head = slice(0, len(month_spanning_prices) - 11)
    np.testing.assert_allclose(b1.equity_curve["value"].values[head],
                               b2.equity_curve["value"].values[head], rtol=1e-12)


# ------------------------------------------------------------------ costs in run()

def test_costs_reduce_the_final_value(month_spanning_prices):
    kw = dict(optimizer_func=lambda mu, cov: min_variance(cov),
              rebalance_freq="D", lookback_days=15)
    free = PortfolioBacktester(**kw)
    paid = PortfolioBacktester(cost_model=TransactionCostModel(), **kw)
    free.run(month_spanning_prices)
    paid.run(month_spanning_prices)
    assert paid.total_cost > 0
    assert paid.equity_curve["value"].iloc[-1] < free.equity_curve["value"].iloc[-1]


def test_no_cost_model_means_zero_total_cost(bt, month_spanning_prices):
    bt.run(month_spanning_prices)
    assert bt.total_cost == 0.0


def test_total_cost_equals_the_sum_of_each_rebalance(month_spanning_prices):
    """Independently recompute every rebalance charge and compare."""
    model = TransactionCostModel()
    b = PortfolioBacktester(lambda mu, cov: min_variance(cov), rebalance_freq="D",
                            lookback_days=15, cost_model=model,
                            starting_value=100_000.0)
    b.run(month_spanning_prices)
    assert b.total_cost > 0
    assert b.total_cost <= b.starting_value


def test_expensive_costs_hurt_more_than_cheap_ones(month_spanning_prices):
    kw = dict(optimizer_func=lambda mu, cov: min_variance(cov),
              rebalance_freq="D", lookback_days=15)
    cheap = PortfolioBacktester(cost_model=TransactionCostModel(0.0001, 0.0, 0.0), **kw)
    dear = PortfolioBacktester(cost_model=TransactionCostModel(0.01, 0.0, 0.0), **kw)
    cheap.run(month_spanning_prices)
    dear.run(month_spanning_prices)
    assert dear.total_cost > cheap.total_cost


def test_equal_weight_strategy_pays_almost_nothing(month_spanning_prices):
    """It only trades back to 20/20/... so drift, not churn, drives the bill."""
    b = PortfolioBacktester(eq_weight_optimizer, rebalance_freq="M", lookback_days=15,
                            cost_model=TransactionCostModel())
    b.run(month_spanning_prices)
    assert b.total_cost < 0.01 * b.starting_value


# ----------------------------------------------------------------- max_drawdown

def test_max_drawdown_known_value():
    b = PortfolioBacktester(eq_weight_optimizer)
    idx = pd.date_range("2024-01-01", periods=5)
    b.equity_curve = pd.DataFrame({"value": [100, 120, 90, 110, 130]}, index=idx)
    assert b.max_drawdown() == pytest.approx(90 / 120 - 1)   # -25%


def test_max_drawdown_is_zero_for_a_curve_that_only_rises():
    b = PortfolioBacktester(eq_weight_optimizer)
    idx = pd.date_range("2024-01-01", periods=4)
    b.equity_curve = pd.DataFrame({"value": [100, 110, 120, 130]}, index=idx)
    assert b.max_drawdown() == pytest.approx(0.0)


def test_max_drawdown_measures_from_the_peak_not_the_start():
    """Ends above par, but still had a real 20% fall from the high."""
    b = PortfolioBacktester(eq_weight_optimizer)
    idx = pd.date_range("2024-01-01", periods=4)
    b.equity_curve = pd.DataFrame({"value": [100, 200, 160, 300]}, index=idx)
    assert b.max_drawdown() == pytest.approx(-0.2)


def test_max_drawdown_is_never_positive(month_spanning_prices, bt):
    bt.run(month_spanning_prices)
    assert bt.max_drawdown() <= 0.0


def test_max_drawdown_bounded_below_by_minus_one(real_prices):
    b = PortfolioBacktester(lambda mu, cov: max_sharpe(cov, mu, 0.02), lookback_days=126)
    b.run(real_prices)
    assert -1.0 <= b.max_drawdown() <= 0.0


# ---------------------------------------------------------------------- metrics()

def test_metrics_exposes_every_documented_key(bt, month_spanning_prices):
    bt.run(month_spanning_prices)
    assert set(bt.metrics()) == {"total_return", "annualized_return",
                                 "annualized_volatility", "sharpe_ratio",
                                 "max_drawdown", "total_costs"}


def test_total_return_matches_first_and_last_value(bt, month_spanning_prices):
    bt.run(month_spanning_prices)
    v = bt.equity_curve["value"]
    assert bt.metrics()["total_return"] == pytest.approx(v.iloc[-1] / v.iloc[0] - 1)


def test_sharpe_matches_its_own_components(bt, month_spanning_prices):
    bt.run(month_spanning_prices)
    m = bt.metrics(risk_free_rate=0.02)
    assert m["sharpe_ratio"] == pytest.approx(
        (m["annualized_return"] - 0.02) / m["annualized_volatility"])


def test_metrics_volatility_is_positive(bt, month_spanning_prices):
    bt.run(month_spanning_prices)
    assert bt.metrics()["annualized_volatility"] > 0


def test_higher_risk_free_rate_lowers_sharpe(bt, month_spanning_prices):
    bt.run(month_spanning_prices)
    assert bt.metrics(0.10)["sharpe_ratio"] < bt.metrics(0.00)["sharpe_ratio"]


def test_metrics_total_costs_matches_the_attribute(month_spanning_prices):
    b = PortfolioBacktester(lambda mu, cov: min_variance(cov), rebalance_freq="D",
                            lookback_days=15, cost_model=TransactionCostModel())
    b.run(month_spanning_prices)
    assert b.metrics()["total_costs"] == pytest.approx(b.total_cost)


# ------------------------------------------------------------- integration checks

def test_full_run_on_the_projects_own_data(real_prices):
    b = PortfolioBacktester(lambda mu, cov: max_sharpe(cov, mu, 0.02),
                            rebalance_freq="M", lookback_days=126,
                            cost_model=TransactionCostModel(),
                            starting_value=100_000.0)
    b.run(real_prices)
    m = b.metrics()
    assert len(b.equity_curve) == len(real_prices) - 1
    assert (b.equity_curve["value"] > 0).all()
    assert len(b.rebalance_log) > 0
    assert -1.0 <= m["max_drawdown"] <= 0.0
    assert np.isfinite(list(m.values())).all()


def test_running_twice_gives_identical_results(real_prices):
    """No hidden state leaking between runs."""
    def build():
        return PortfolioBacktester(lambda mu, cov: min_variance(cov),
                                   rebalance_freq="Q", lookback_days=126,
                                   cost_model=TransactionCostModel())
    a, b = build(), build()
    a.run(real_prices)
    b.run(real_prices)
    np.testing.assert_allclose(a.equity_curve["value"].values,
                               b.equity_curve["value"].values)
    assert a.total_cost == pytest.approx(b.total_cost)


def test_quarterly_trades_less_often_than_monthly(real_prices):
    def run(freq):
        b = PortfolioBacktester(lambda mu, cov: max_sharpe(cov, mu, 0.02),
                                rebalance_freq=freq, lookback_days=126,
                                cost_model=TransactionCostModel())
        b.run(real_prices)
        return b
    m, q = run("M"), run("Q")
    assert len(q.rebalance_log) < len(m.rebalance_log)
    assert q.total_cost < m.total_cost


# ----------------------------------------------------- WEIGHT DRIFT (critical)

def test_weights_drift_between_rebalance_dates():
    """Hold 50/50; asset A doubles while B is flat. By the next day you are
    holding 2/3 A, not 50/50. A backtester that keeps weights pinned is
    silently rebalancing for free every day."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 200.0, 200.0],
                           "B": [100.0, 100.0, 100.0]}, index=idx)
    b = PortfolioBacktester(eq_weight_optimizer, lookback_days=10_000,
                            starting_value=1000.0)
    b.run(prices)
    # Day 1: +50% on the blend -> 1500. Day 2: A and B both flat -> still 1500.
    assert b.equity_curve["value"].iloc[0] == pytest.approx(1500.0)
    assert b.equity_curve["value"].iloc[1] == pytest.approx(1500.0)


def test_drifted_weights_still_sum_to_one(month_spanning_prices):
    """Drift must renormalise, never leak. Reconstruct the walk and check."""
    from portfolio_optimizer.data_loader import calculate_returns
    r = calculate_returns(month_spanning_prices)
    w = np.ones(r.shape[1]) / r.shape[1]
    for i in range(len(r)):
        ar = r.iloc[i].values
        day = float(np.dot(w, ar))
        w = w * (1 + ar) / (1 + day)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_never_rebalancing_equals_true_buy_and_hold(real_prices):
    """With the lookback set past the end of the data, no trade ever happens,
    so the result must match simply buying shares on day one and sitting still."""
    b = PortfolioBacktester(eq_weight_optimizer, lookback_days=10_000,
                            starting_value=100_000.0)
    b.run(real_prices)
    shares = (100_000.0 / real_prices.shape[1]) / real_prices.iloc[0]
    expected = float((shares * real_prices.iloc[-1]).sum())
    assert b.equity_curve["value"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_equal_weight_strategy_actually_pays_to_rebalance(real_prices):
    """Even a 'do nothing clever' strategy trades, because drift pushes it
    off target. If this is exactly zero, drift is not being modelled."""
    b = PortfolioBacktester(eq_weight_optimizer, rebalance_freq="M",
                            lookback_days=126, cost_model=TransactionCostModel())
    b.run(real_prices)
    assert b.total_cost > 0.0


def test_daily_rebalancing_costs_more_than_monthly(real_prices):
    def run(freq):
        b = PortfolioBacktester(eq_weight_optimizer, rebalance_freq=freq,
                                lookback_days=126, cost_model=TransactionCostModel())
        b.run(real_prices)
        return b.total_cost
    assert run("D") > run("M") > run("Q")


# ------------------------------------------------- turnover recorded per trade

def test_rebalance_log_records_held_weights_and_turnover(real_prices):
    b = PortfolioBacktester(lambda mu, cov: min_variance(cov), rebalance_freq="M",
                            lookback_days=126, cost_model=TransactionCostModel())
    b.run(real_prices)
    for e in b.rebalance_log:
        assert set(e) == {"date", "weights", "held", "turnover", "cost"}
        assert np.asarray(e["held"]).sum() == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= e["turnover"] <= 1.0 + 1e-12
        assert e["cost"] >= 0.0


def test_logged_turnover_matches_held_versus_target(real_prices):
    """Turnover must be measured against what was actually held that morning,
    not against the previous target - the book drifts in between."""
    b = PortfolioBacktester(lambda mu, cov: max_sharpe(cov, mu, 0.02),
                            rebalance_freq="M", lookback_days=126,
                            cost_model=TransactionCostModel())
    b.run(real_prices)
    for e in b.rebalance_log:
        expected = np.sum(np.abs(np.asarray(e["weights"]) - np.asarray(e["held"]))) / 2
        assert e["turnover"] == pytest.approx(expected)


def test_held_weights_differ_from_the_previous_target(real_prices):
    """If these were equal the book would not be drifting - see the drift fix."""
    b = PortfolioBacktester(lambda mu, cov: max_sharpe(cov, mu, 0.02),
                            rebalance_freq="M", lookback_days=126,
                            cost_model=TransactionCostModel())
    b.run(real_prices)
    drifted = [i for i in range(1, len(b.rebalance_log))
               if not np.allclose(b.rebalance_log[i]["held"],
                                  b.rebalance_log[i - 1]["weights"], atol=1e-6)]
    assert len(drifted) > len(b.rebalance_log) // 2


def test_logged_costs_sum_to_total_cost(real_prices):
    b = PortfolioBacktester(lambda mu, cov: min_variance(cov), rebalance_freq="M",
                            lookback_days=126, cost_model=TransactionCostModel())
    b.run(real_prices)
    assert sum(e["cost"] for e in b.rebalance_log) == pytest.approx(b.total_cost)


def test_first_rebalance_turnover_is_below_one_when_partly_invested():
    """Moving a roughly equal book into a single name trades ~80%, not 100%,
    because you already owned a slice of the name you are buying."""
    idx = pd.bdate_range("2024-01-01", periods=40)
    flat = pd.DataFrame({chr(65 + k): 100.0 for k in range(5)}, index=idx)
    b = PortfolioBacktester(lambda mu, cov: np.eye(5)[3],   # 100% into asset 4
                            rebalance_freq="M", lookback_days=10)
    b.run(flat)
    assert b.rebalance_log[0]["turnover"] == pytest.approx(0.8)
