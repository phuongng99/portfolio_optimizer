"""
Portfolio Optimiser — entry point.

    python main.py                                   # your original 5 stocks
    python main.py --prices multi_asset_prices.csv   # stocks + bonds + gold + REITs
    python main.py --prices multi_asset_prices.csv --cap 0.40
    python main.py --prices multi_asset_prices.csv --cap-sweep
    python main.py --dashboard                       # also rebuild and open the dashboard

Runs the whole pipeline: load prices, summarise each asset, compare the four
strategies on the full sample, then backtest each one month by month with
transaction costs and report where the risk actually came from.
"""

import argparse
import os
import sys

import numpy as np

# Run from anywhere - the package lives one directory up from this file.
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
    efficient_frontier, equal_weighted_portfolio, max_sharpe, min_variance, risk_parity,
)
from portfolio_optimizer.report import diversification_ratio, pct_risk_contribution

HERE = os.path.dirname(os.path.abspath(__file__))
RISK_FREE = 0.02
LOOKBACK = 126


def rule(title):
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")


def cap_sweep(prices, start_value):
    """Run every strategy at a range of caps, then check whether picking the
    best cap in hindsight actually survives out of sample."""
    n = prices.shape[1]
    floor = 1.0 / n
    grid = [None] + [c for c in (.60, .50, .40, .35, .30, .25, .20)
                     if c > floor + 1e-9] + [round(floor + 0.005, 3)]

    def build(cap):
        b = PortfolioConstraints(max_weight=cap).get_bounds(n) if cap else None
        return {
            "Equal-Weight": lambda m, c: equal_weighted_portfolio(len(m)),
            "Min-Variance": lambda m, c: min_variance(c, bounds=b),
            "Max-Sharpe":   lambda m, c: max_sharpe(c, m, RISK_FREE, bounds=b),
            "Risk-Parity":  lambda m, c: risk_parity(c, bounds=b),
        }

    def score(px, cap, name):
        bt = PortfolioBacktester(build(cap)[name], rebalance_freq="M",
                                 starting_value=start_value,
                                 cost_model=TransactionCostModel(), lookback_days=LOOKBACK)
        bt.run(px)
        m = bt.metrics(RISK_FREE)
        return {"final": float(bt.equity_curve["value"].iloc[-1]),
                "sharpe": m["sharpe_ratio"], "cost": m["total_costs"],
                "dd": m["max_drawdown"],
                "turn": sum(e["turnover"] for e in bt.rebalance_log)}

    names = ["Equal-Weight", "Min-Variance", "Max-Sharpe", "Risk-Parity"]

    rule(f"CAP SWEEP  ·  {n} assets, smallest possible cap {floor:.1%}")
    print("  " + " " * 7 + "".join(f"{x:^18}" for x in names))
    print(f"  {'cap':<7}" + "".join(f"{'final':>11}{'sharpe':>7}" for _ in names))
    table = {}
    for cap in grid:
        table[cap] = {nm: score(prices, cap, nm) for nm in names}
        lab = "none" if cap is None else f"{cap:.0%}"
        row = "".join("{:>11}{:>7.2f}".format("$" + format(table[cap][nm]["final"], ",.0f"),
                                               table[cap][nm]["sharpe"])
                      for nm in names)
        print(f"  {lab:<7}{row}")

    print(f"\n  {'strategy':<15}{'best cap':>10}{'sharpe':>9}{'costs there':>14}{'turnover':>10}")
    picks = {}
    for nm in names:
        best = max(grid, key=lambda c: table[c][nm]["sharpe"])
        picks[nm] = best
        r = table[best][nm]
        lab = "none" if best is None else f"{best:.0%}"
        print("  {:<15}{:>10}{:>9.2f}{:>14}{:>9.0f}%".format(
            nm, lab, r["sharpe"], "$" + format(r["cost"], ",.0f"), r["turn"] * 100))

    # ---- the honest test -------------------------------------------------
    half = len(prices) // 2
    first, second = prices.iloc[:half], prices.iloc[half:]
    if len(second) <= LOOKBACK + 21:
        print("\n  (not enough data for an out-of-sample check)")
        return

    rule("OUT-OF-SAMPLE CHECK  ·  cap chosen on the first half, scored on the second")
    print(f"  first half  {first.index[0].date()} to {first.index[-1].date()}  ({len(first)} days)")
    print(f"  second half {second.index[0].date()} to {second.index[-1].date()}  ({len(second)} days)")
    print(f"\n  {'strategy':<15}{'cap picked':>11}{'2nd-half':>10}{'no cap':>9}{'verdict':>26}")
    for nm in names:
        chosen = max(grid, key=lambda c: score(first, c, nm)["sharpe"])
        got = score(second, chosen, nm)["sharpe"]
        base = score(second, None, nm)["sharpe"]
        lab = "none" if chosen is None else f"{chosen:.0%}"
        if abs(got - base) < 0.01:
            verdict = "no difference"
        elif got > base:
            verdict = f"held up (+{got - base:.2f})"
        else:
            verdict = f"did NOT hold ({got - base:+.2f})"
        print(f"  {nm:<15}{lab:>11}{got:>10.2f}{base:>9.2f}{verdict:>26}")
    print("\n  A cap that wins on the first half but not the second was fitted to")
    print("  the past, not to anything real. Prefer a round number you would")
    print("  defend before seeing results.")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", default="stock_prices.csv",
                    help="CSV inside data/, or a full path")
    ap.add_argument("--cap", type=float, default=None,
                    help="max weight per asset, e.g. 0.40. Default: no limit")
    ap.add_argument("--start-value", type=float, default=100_000.0)
    ap.add_argument("--dashboard", action="store_true",
                    help="rebuild the interactive dashboard afterwards")
    ap.add_argument("--cap-sweep", action="store_true",
                    help="try a range of caps and check they survive out of sample")
    args = ap.parse_args()

    src = args.prices if os.path.sep in args.prices else os.path.join(HERE, "data", args.prices)
    if not os.path.exists(src):
        sys.exit(f"no such price file: {src}")

    print(f"📈  {ENGINE_NAME or '(unnamed engine)'}")

    # ── load ────────────────────────────────────────────────────────────────
    prices = load_prices(src)
    returns = calculate_returns(prices)
    tickers = list(prices.columns)
    n = len(tickers)
    mu = annualize_returns(returns)
    vol = annualize_volatility(returns)
    cov = calculate_covariance(returns)

    # A cap below 1/n is impossible: even holding the maximum of everything
    # cannot add up to 100% of the portfolio. Fail with an explanation rather
    # than letting every optimiser silently return None.
    if args.cap is not None:
        if args.cap < 1.0 / n:
            sys.exit(f"--cap {args.cap:.0%} is impossible with {n} assets: "
                     f"{n} x {args.cap:.0%} = {n*args.cap:.0%}, which cannot reach 100%. "
                     f"The smallest workable cap here is {1/n:.1%} (equal weight).")
        if args.cap > 1.0:
            sys.exit(f"--cap {args.cap} is above 1.0. Pass a fraction, e.g. 0.40 for 40%.")

    bounds = PortfolioConstraints(max_weight=args.cap).get_bounds(n) if args.cap else None
    cap_note = f"capped at {args.cap:.0%} per asset" if args.cap else "no weight cap"

    rule(f"DATA  ·  {os.path.basename(src)}")
    print(f"{len(prices)} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"{n} assets · risk-free {RISK_FREE:.0%} · {LOOKBACK}-day lookback · {cap_note}\n")
    print(f"  {'asset':<8}{'ann return':>12}{'ann vol':>10}{'sharpe':>9}{'total':>10}")
    for t in tickers:
        tot = prices[t].iloc[-1] / prices[t].iloc[0] - 1
        print(f"  {t:<8}{mu[t]*100:>11.1f}%{vol[t]*100:>9.1f}%"
              f"{(mu[t]-RISK_FREE)/vol[t]:>9.2f}{tot*100:>9.1f}%")
    print(f"\n  volatility spread: {vol.min()*100:.1f}% to {vol.max()*100:.1f}% "
          f"({vol.max()/vol.min():.1f}x)")

    rule("CORRELATION")
    print(calculate_correlation(returns).round(2).to_string())

    # ── full-sample strategies ──────────────────────────────────────────────
    built = {
        "Equal-Weight": equal_weighted_portfolio(n),
        "Min-Variance": min_variance(cov.values, bounds=bounds),
        "Max-Sharpe":   max_sharpe(cov.values, mu.values, RISK_FREE, bounds=bounds),
        "Risk-Parity":  risk_parity(cov.values, bounds=bounds),
    }

    rule("STRATEGIES  ·  full sample, in-sample (no trading)")
    print(f"  {'strategy':<15}{'return':>9}{'vol':>8}{'sharpe':>8}{'div':>7}   weights")
    for name, w in built.items():
        if w is None:
            print(f"  {name:<15}  did not converge")
            continue
        w = np.asarray(w)
        sigma = float(np.sqrt(w @ cov.values @ w))
        ret = float(w @ mu.values)
        wtxt = "  ".join(f"{t} {x*100:>4.0f}%" for t, x in zip(tickers, w))
        print(f"  {name:<15}{ret*100:>8.1f}%{sigma*100:>7.1f}%"
              f"{(ret-RISK_FREE)/sigma:>8.2f}"
              f"{diversification_ratio(w, cov.values):>7.2f}   {wtxt}")

    fr = efficient_frontier(mu.values, cov.values, n_points=40, bounds=bounds)
    print(f"\n  efficient frontier: {len(fr['Returns'])} portfolios, "
          f"volatility {fr['Volatilities'].min()*100:.1f}% to {fr['Volatilities'].max()*100:.1f}%")

    # ── backtest ────────────────────────────────────────────────────────────
    strategies = {
        "Equal-Weight": lambda m, c: equal_weighted_portfolio(len(m)),
        "Min-Variance": lambda m, c: min_variance(c, bounds=bounds),
        "Max-Sharpe":   lambda m, c: max_sharpe(c, m, RISK_FREE, bounds=bounds),
        "Risk-Parity":  lambda m, c: risk_parity(c, bounds=bounds),
    }

    rule(f"BACKTEST  ·  monthly rebalance, costs charged, {cap_note}")
    print(f"  {'strategy':<15}{'final':>12}{'annual':>9}{'vol':>8}{'sharpe':>8}"
          f"{'max fall':>10}{'costs':>10}{'turnover':>9}")
    results = {}
    for name, fn in strategies.items():
        bt = PortfolioBacktester(fn, rebalance_freq="M", starting_value=args.start_value,
                                 cost_model=TransactionCostModel(), lookback_days=LOOKBACK)
        bt.run(prices)
        m = bt.metrics(RISK_FREE)
        results[name] = bt
        turn = sum(e["turnover"] for e in bt.rebalance_log)
        print(f"  {name:<15}${bt.equity_curve['value'].iloc[-1]:>11,.0f}"
              f"{m['annualized_return']*100:>8.1f}%{m['annualized_volatility']*100:>7.1f}%"
              f"{m['sharpe_ratio']:>8.2f}{m['max_drawdown']*100:>9.1f}%"
              f"  ${m['total_costs']:>7,.0f}{turn*100:>9.0f}%")

    shares = (args.start_value / n) / prices.iloc[0]
    bh = float((shares * prices.iloc[-1]).sum())
    print(f"  {'Buy & hold':<15}${bh:>11,.0f}    no trading, no fees, no decisions")

    best = max(results, key=lambda k: results[k].metrics(RISK_FREE)["sharpe_ratio"])
    beat = [k for k, v in results.items() if v.equity_curve["value"].iloc[-1] > bh]
    print(f"\n  best risk-adjusted: {best}")
    print(f"  beat buy & hold:    {', '.join(beat) if beat else 'none of them'}")

    # ── risk decomposition ──────────────────────────────────────────────────
    rule("RISK DECOMPOSITION  ·  final book of each strategy")
    print(f"  {'strategy':<15}{'asset':<8}{'money':>8}{'risk':>8}{'gap':>9}")
    for name, bt in results.items():
        w = np.asarray(bt.rebalance_log[-1]["weights"])
        risk = pct_risk_contribution(w, cov.values)
        for i, t in enumerate(tickers):
            label = name if i == 0 else ""
            gap = risk[i] - w[i] * 100
            flag = "  <-- carries more risk than money" if gap > 8 else ""
            print(f"  {label:<15}{t:<8}{w[i]*100:>7.0f}%{risk[i]:>7.0f}%{gap:>+8.0f}pp{flag}")
        print()

    if args.cap_sweep:
        cap_sweep(prices, args.start_value)

    if args.dashboard:
        import subprocess
        rule("DASHBOARD")
        # Flush first: our own prints are buffered while the child writes
        # straight to the terminal, so without this the child's output
        # appears above ours and the report reads out of order.
        sys.stdout.flush()
        subprocess.run([sys.executable, os.path.join(HERE, "build_dashboard_data.py"),
                        "--prices", args.prices], check=True)
        subprocess.run([sys.executable, os.path.join(HERE, "build_dashboard.py")], check=True)
    else:
        print("\n  (add --dashboard to rebuild and open the interactive chart)")


if __name__ == "__main__":
    main()
