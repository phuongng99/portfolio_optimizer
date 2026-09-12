"""Run every strategy through the backtester and dump the results as JSON
for the interactive dashboard. Re-run this whenever the price data updates.

    python build_dashboard_data.py

Each strategy is run three times - with no weight cap, and with a 40% and a
25% per-asset cap - so the dashboard can switch between them without
recomputing. Uncapped optimisers, max-Sharpe especially, will happily put
100% of the book in one name; the caps are how you stop that.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

# Let this run as a plain script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimizer import ENGINE_NAME
from portfolio_optimizer.backtestter import PortfolioBacktester
from portfolio_optimizer.constraints import PortfolioConstraints
from portfolio_optimizer.costs import TransactionCostModel
from portfolio_optimizer.data_loader import (
    annualize_returns, annualize_volatility, calculate_correlation,
    calculate_covariance, calculate_returns, load_prices,
)
from portfolio_optimizer.optimizer import (
    efficient_frontier, equal_weighted_portfolio, inverse_volatility, max_sharpe,
    min_variance, risk_budget, risk_parity,
)
from portfolio_optimizer.report import diversification_ratio, pct_risk_contribution

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RISK_FREE = 0.02
LOOKBACK = 126
START = 100_000.0

# key -> max weight per asset (None = unconstrained, i.e. 0-100%)
CAPS = {"none": None, "40": 0.40, "25": 0.25}


def strategies_for(bounds):
    """The strategies, each built against one set of weight bounds.

    Risk-Budget and Inverse-Volatility are closed-form/unconstrained methods
    that cannot honour a weight cap, so they appear only in the uncapped
    variant rather than being shown in violation of one.
    """
    s = {
        "Equal-Weight": lambda mu, cov: equal_weighted_portfolio(len(mu)),
        "Min-Variance": lambda mu, cov: min_variance(cov, bounds=bounds),
        "Max-Sharpe":   lambda mu, cov: max_sharpe(cov, mu, RISK_FREE, bounds=bounds),
        "Risk-Parity":  lambda mu, cov: risk_parity(cov, bounds=bounds),
    }
    if bounds is None:
        s["Risk-Budget"] = lambda mu, cov: risk_budget(cov)
        s["Inverse-Volatility"] = lambda mu, cov: inverse_volatility(cov)
    return s


def describe(weights, mu, cov):
    w = np.asarray(weights)
    sigma = float(np.sqrt(w @ cov @ w))
    ret = float(w @ mu)
    return {
        "weights": [round(float(x), 4) for x in w],
        "ret": ret,
        "vol": sigma,
        "sharpe": (ret - RISK_FREE) / sigma if sigma > 0 else 0.0,
        "pct_risk": [round(float(x), 2) for x in pct_risk_contribution(w, cov)],
        "div_ratio": float(diversification_ratio(w, cov)),
    }


def run_variant(prices, returns, mu_all, cov_all, bounds):
    """Backtest every strategy for this cap setting, plus the frontier."""
    v = {"backtests": {}, "static_strategies": [], "frontier": []}

    for name, fn in strategies_for(bounds).items():
        bt = PortfolioBacktester(fn, rebalance_freq="M", starting_value=START,
                                 cost_model=TransactionCostModel(), lookback_days=LOOKBACK)
        bt.run(prices)
        eq = bt.equity_curve["value"]
        dd = eq / eq.cummax() - 1.0
        trough = dd.idxmin()
        pk_date = eq[:trough].idxmax()
        after = eq[trough:]
        rec = after[after >= eq[pk_date]]

        v["backtests"][name] = {
            "equity": [round(float(x), 2) for x in eq.values],
            "drawdown": [round(float(x) * 100, 3) for x in dd.values],
            "metrics": {k: float(x) for k, x in bt.metrics(RISK_FREE).items()},
            "max_holding": float(max(max(e["weights"]) for e in bt.rebalance_log))
                           if bt.rebalance_log else 1.0 / prices.shape[1],
            "drawdown_detail": {
                "peak_date": str(pk_date.date()), "peak_value": float(eq[pk_date]),
                "trough_date": str(trough.date()), "trough_value": float(eq[trough]),
                "recovery_date": str(rec.index[0].date()) if len(rec) else None,
            },
            "rebalances": [{"date": str(e["date"].date()),
                            "weights": [round(float(x), 4) for x in np.asarray(e["weights"])],
                            "held": [round(float(x), 4) for x in np.asarray(e["held"])],
                            "turnover": round(e["turnover"], 5),
                            "cost": round(e["cost"], 2)}
                           for e in bt.rebalance_log],
        }

        w = fn(mu_all.values, cov_all.values)
        d = describe(w, mu_all.values, cov_all.values)
        d["name"] = name
        v["static_strategies"].append(d)

    fr = efficient_frontier(cov_all.values, mu_all.values, n_points=60, bounds=bounds)
    v["frontier"] = [{"vol": float(a), "ret": float(b),
                      "weights": [round(float(x), 4) for x in c]}
                     for a, b, c in zip(fr["Volatilities"], fr["Returns"], fr["Weights"])]
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prices", default="stock_prices.csv",
                    help="CSV inside data/, or a path. e.g. multi_asset_prices.csv")
    ap.add_argument("--out", default="dashboard_data.json")
    args = ap.parse_args()

    src = args.prices if os.path.sep in args.prices else os.path.join(HERE, "data", args.prices)
    prices = load_prices(src)
    print(f"using {os.path.basename(src)}: {prices.shape[1]} assets, {len(prices)} days")
    returns = calculate_returns(prices)
    tickers = list(prices.columns)
    n = len(tickers)
    mu_all = annualize_returns(returns)
    vol_all = annualize_volatility(returns)
    cov_all = calculate_covariance(returns)

    out = {
        "engine": ENGINE_NAME,
        "tickers": tickers,
        "risk_free": RISK_FREE,
        "lookback_days": LOOKBACK,
        "starting_value": START,
        "caps": list(CAPS.keys()),
        "period": {"start": str(prices.index[0].date()),
                   "end": str(prices.index[-1].date()),
                   "trading_days": int(len(returns))},
        "dates": [str(d.date()) for d in returns.index],
    }

    out["assets"] = [{
        "ticker": t,
        "annual_return": float(mu_all[t]),
        "annual_vol": float(vol_all[t]),
        "sharpe": float((mu_all[t] - RISK_FREE) / vol_all[t]),
        "total_return": float(prices[t].iloc[-1] / prices[t].iloc[0] - 1),
    } for t in tickers]
    out["correlation"] = calculate_correlation(returns).round(4).values.tolist()

    # random long-only mixes, for context behind the frontier
    rng = np.random.default_rng(12345)
    out["cloud"] = []
    for _ in range(700):
        w = rng.dirichlet(np.ones(n))
        out["cloud"].append({"vol": float(np.sqrt(w @ cov_all.values @ w)),
                             "ret": float(w @ mu_all.values)})

    # true buy and hold: buy shares on day one, never trade again
    shares = (START / n) / prices.iloc[0]
    bh = (prices * shares).sum(axis=1).iloc[1:]
    out["buy_and_hold"] = {"equity": [round(float(x), 2) for x in bh.values],
                           "final": float(bh.iloc[-1])}

    out["variants"] = {}
    for key, cap in CAPS.items():
        bounds = None if cap is None else PortfolioConstraints(max_weight=cap).get_bounds(n)
        out["variants"][key] = run_variant(prices, returns, mu_all, cov_all, bounds)
        label = "no cap" if cap is None else f"cap {cap:.0%}"
        print(f"\n{label}")
        for name, b in out["variants"][key]["backtests"].items():
            m = b["metrics"]
            print(f"  {name:<18} ${b['equity'][-1]:>11,.0f}  sharpe {m['sharpe_ratio']:>5.2f}  "
                  f"maxDD {m['max_drawdown']*100:>6.1f}%  costs ${m['total_costs']:>7,.0f}  "
                  f"biggest holding {b['max_holding']*100:>3.0f}%")

    path = args.out if os.path.sep in args.out else os.path.join(HERE, args.out)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {path}  ({os.path.getsize(path) / 1024:.0f} KB)")
    print(f"  Buy & hold   ${out['buy_and_hold']['final']:>11,.0f}  (no trading, no fees)")


if __name__ == "__main__":
    main()
