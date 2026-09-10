"""Generate realistic multi-asset price history: stocks, bonds, gold,
real estate and commodities.

    python generate_data.py                       # 8 years, default seed
    python generate_data.py --years 12 --seed 7
    python generate_data.py --out data/other.csv

Why this exists: five similar tech-ish equities all have volatility in a
narrow 21-30% band, so risk parity and min-variance barely differ from equal
weight. Real portfolios span 4% (short bonds) to 20% (real estate) - a 5x
spread - and that is where those strategies earn their keep.

This is SYNTHETIC data. It is built to have believable statistics, not to
reproduce any actual history. For real prices use fetch_yahoo.py.

What is modelled deliberately:

  * Volatility levels and long-run returns taken from each asset class's
    real behaviour.
  * A correlation structure with the sign that matters: bonds move OPPOSITE
    to stocks in normal times, which is the entire reason to hold them.
  * Fat tails (Student-t innovations). Real returns have far more extreme
    days than a normal distribution allows, and drawdowns come from those days.
  * Four regimes, because a single-regime series flatters whichever strategy
    happens to suit it:
      1. calm bull      - stocks grind up, bonds quiet
      2. crash          - stocks fall hard, bonds rally, risk assets converge
      3. recovery       - stocks rebound, bonds give some back
      4. inflation shock - stocks AND bonds fall together, commodities win
    Regime 4 is the one that breaks the classic 60/40 and punishes anyone who
    assumed bonds always hedge.
"""

import argparse
import os

import numpy as np
import pandas as pd

TICKERS = ["SPY", "TLT", "IEF", "GLD", "VNQ", "DBC"]
DESCRIPTIONS = {
    "SPY": "US large-cap stocks", "TLT": "20+ year Treasuries",
    "IEF": "7-10 year Treasuries", "GLD": "Gold",
    "VNQ": "US real estate", "DBC": "Broad commodities",
}
START_PRICES = {"SPY": 200.0, "TLT": 120.0, "IEF": 105.0,
                "GLD": 115.0, "VNQ": 80.0, "DBC": 16.0}

# Long-run annualised return each asset should actually DELIVER over the whole
# sample, after compounding. The per-regime drifts below set the shape - which
# asset wins in which regime - and a calibration pass then shifts them all by a
# constant so the realised figures land here. Without that pass, volatility
# drag (worse with fat tails than the textbook sigma^2/2) quietly eats several
# percent a year and every asset looks like a dud.
TARGET_ANNUAL = {"SPY": .095, "TLT": .020, "IEF": .015,
                 "GLD": .050, "VNQ": .070, "DBC": .020}

# annualised volatility per asset, per regime
VOL = {
    "calm":      {"SPY": .13, "TLT": .12, "IEF": .045, "GLD": .14, "VNQ": .15, "DBC": .15},
    "crash":     {"SPY": .34, "TLT": .19, "IEF": .075, "GLD": .22, "VNQ": .42, "DBC": .30},
    "recovery":  {"SPY": .18, "TLT": .13, "IEF": .050, "GLD": .15, "VNQ": .22, "DBC": .19},
    "inflation": {"SPY": .22, "TLT": .18, "IEF": .070, "GLD": .16, "VNQ": .26, "DBC": .23},
}

# annualised drift per asset, per regime
MU = {
    "calm":      {"SPY": .14, "TLT": .035, "IEF": .025, "GLD": .04, "VNQ": .11, "DBC": .03},
    "crash":     {"SPY": -.45, "TLT": .22, "IEF": .11, "GLD": .10, "VNQ": -.55, "DBC": -.30},
    "recovery":  {"SPY": .26, "TLT": -.02, "IEF": .005, "GLD": .07, "VNQ": .24, "DBC": .14},
    "inflation": {"SPY": -.16, "TLT": -.24, "IEF": -.09, "GLD": .06, "VNQ": -.20, "DBC": .30},
}


def correlation(regime):
    """Correlations by regime. In a crash, risk assets converge toward each
    other and the flight to quality makes the stock/bond link more negative."""
    base = np.array([
        # SPY   TLT   IEF   GLD   VNQ   DBC
        [1.00, -.35, -.30,  .05,  .72,  .35],   # SPY
        [-.35, 1.00,  .92,  .25, -.10, -.20],   # TLT
        [-.30,  .92, 1.00,  .28, -.08, -.18],   # IEF
        [.05,   .25,  .28, 1.00,  .12,  .30],   # GLD
        [.72,  -.10, -.08,  .12, 1.00,  .25],   # VNQ
        [.35,  -.20, -.18,  .30,  .25, 1.00],   # DBC
    ])
    if regime == "crash":
        risk = [0, 4, 5]                       # SPY, VNQ, DBC
        for i in risk:
            for j in risk:
                if i != j:
                    base[i, j] = min(0.92, base[i, j] + 0.30)
        for i in risk:
            for j in (1, 2):                   # bonds
                base[i, j] = base[j, i] = base[i, j] - 0.20
    if regime == "inflation":
        # the 2022 problem: stocks and bonds stop hedging each other
        for i in (0, 4):
            for j in (1, 2):
                base[i, j] = base[j, i] = 0.45
    np.fill_diagonal(base, 1.0)
    return nearest_psd(base)


def nearest_psd(c):
    """Clip any negative eigenvalues so the matrix is a usable correlation
    matrix. Hand-written correlations are rarely positive semi-definite."""
    vals, vecs = np.linalg.eigh(c)
    if vals.min() > 1e-10:
        return c
    vals = np.clip(vals, 1e-8, None)
    out = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


def t_innovations(rng, n_days, n_assets, df=4.0):
    """Student-t shocks, rescaled to unit variance. Fat tails are why real
    portfolios have drawdowns a normal model says are impossible."""
    z = rng.standard_t(df, size=(n_days, n_assets))
    return z / np.sqrt(df / (df - 2.0))


def build_schedule(total_days):
    """Regime order and length. Proportions roughly match how markets spend
    their time: mostly calm, punctuated by shorter violent stretches."""
    plan = [("calm", .30), ("crash", .09), ("recovery", .26),
            ("calm", .13), ("inflation", .13), ("recovery", .09)]
    out = []
    for name, share in plan:
        out.append((name, int(round(total_days * share))))
    out[-1] = (out[-1][0], total_days - sum(n for _, n in out[:-1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--out", default=os.path.join("data", "multi_asset_prices.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_days = int(round(args.years * 252))
    n = len(TICKERS)

    dates = pd.bdate_range(args.start, periods=n_days)
    returns = np.zeros((n_days, n))
    regime_of_day = []

    day = 0
    for regime, length in build_schedule(n_days):
        length = min(length, n_days - day)
        if length <= 0:
            continue
        L = np.linalg.cholesky(correlation(regime))
        vol = np.array([VOL[regime][t] for t in TICKERS]) / np.sqrt(252)
        mu = np.array([MU[regime][t] for t in TICKERS]) / 252
        shocks = t_innovations(rng, length, n) @ L.T
        returns[day:day + length] = mu + shocks * vol
        regime_of_day += [regime] * length
        day += length

    # --- calibration: shift each asset's drift so the REALISED compound
    # return matches TARGET_ANNUAL. Two passes is plenty; the relationship
    # between a constant daily shift and the compound result is near-linear.
    for _ in range(2):
        realised = np.array([
            (np.prod(1 + returns[:, k]) ** (252 / n_days)) - 1 for k in range(n)
        ])
        target = np.array([TARGET_ANNUAL[t] for t in TICKERS])
        returns += (np.log1p(target) - np.log1p(realised)) / 252

    prices = np.vstack([np.ones(n), np.cumprod(1 + returns, axis=0)])
    prices *= np.array([START_PRICES[t] for t in TICKERS])
    df = pd.DataFrame(prices[1:], index=dates, columns=TICKERS).round(4)
    df.index.name = "Date"

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out)

    daily = df.pct_change().dropna()
    print(f"wrote {out}")
    print(f"  {len(df)} trading days, {df.index[0].date()} to {df.index[-1].date()}\n")
    print(f"  {'ticker':<7}{'':<24}{'ann ret':>9}{'ann vol':>9}{'sharpe':>8}{'worst day':>11}")
    for t in TICKERS:
        mu_ = daily[t].mean() * 252
        sd = daily[t].std() * np.sqrt(252)
        print(f"  {t:<7}{DESCRIPTIONS[t]:<24}{mu_*100:>8.1f}%{sd*100:>8.1f}%"
              f"{(mu_-0.02)/sd:>8.2f}{daily[t].min()*100:>10.1f}%")
    print(f"\n  volatility spread: {daily.std().min()*np.sqrt(252)*100:.1f}% "
          f"to {daily.std().max()*np.sqrt(252)*100:.1f}%  "
          f"({daily.std().max()/daily.std().min():.1f}x)")
    print(f"  SPY vs TLT correlation: {daily['SPY'].corr(daily['TLT']):+.2f}")

    seg = pd.Series(regime_of_day, index=dates)
    print("\n  regimes:")
    for name in ["calm", "crash", "recovery", "inflation"]:
        d = seg[seg == name].index
        if len(d):
            r = daily.loc[daily.index.isin(d), "SPY"]
            print(f"    {name:<10} {len(d):>4} days   SPY {((1+r).prod()-1)*100:>+7.1f}%")
    print("\nnext:")
    print(f"  python build_dashboard_data.py --prices {os.path.basename(out)}")


if __name__ == "__main__":
    main()
