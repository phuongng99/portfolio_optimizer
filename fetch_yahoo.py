"""Download real daily prices from Yahoo Finance into the project's CSV format.

Run this in YOUR OWN terminal (it needs internet):

    pip install yfinance
    python fetch_yahoo.py                      # default universe, 10 years
    python fetch_yahoo.py --start 2005-01-01   # include the 2008 crash
    python fetch_yahoo.py --tickers SPY TLT GLD
    python fetch_yahoo.py --out data/my_prices.csv

Output is Date + one column per ticker of adjusted closes - exactly what
load_prices() expects, so nothing else in the project changes.

Adjusted closes matter: they fold dividends and splits back in. Bond ETFs
like TLT and IEF pay most of their return as coupons, so using raw closes
would make bonds look far worse than they were.
"""

import argparse
import os
import sys

# The default universe: six asset classes whose volatilities span roughly
# 4% to 30% a year. That spread is the point - with five similar equities,
# risk parity and min-variance have almost nothing to do.
DEFAULT_TICKERS = {
    "SPY": "US large-cap stocks",
    "TLT": "20+ year Treasury bonds",
    "IEF": "7-10 year Treasury bonds",
    "GLD": "Gold",
    "VNQ": "US real estate",
    "DBC": "Broad commodities",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None, help="defaults to today")
    ap.add_argument("--out", default=os.path.join("data", "stock_prices.csv"))
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed. Run:  pip install yfinance")

    print(f"downloading {len(args.tickers)} tickers from {args.start}...")
    for t in args.tickers:
        if t in DEFAULT_TICKERS:
            print(f"  {t:<6} {DEFAULT_TICKERS[t]}")

    raw = yf.download(args.tickers, start=args.start, end=args.end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        sys.exit("Yahoo returned nothing - check the tickers and the date range.")

    # yfinance returns a column MultiIndex for multiple tickers, flat for one.
    prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if hasattr(prices, "to_frame"):
        prices = prices.to_frame(args.tickers[0])
    prices = prices[args.tickers]          # keep the order you asked for

    # Drop any day where a ticker has no price. ETFs have different inception
    # dates, so this trims the history to where they ALL exist - otherwise the
    # covariance matrix is built from mismatched windows.
    before = len(prices)
    prices = prices.dropna()
    if len(prices) < before:
        print(f"  dropped {before - len(prices)} rows with missing prices "
              f"(different ETF start dates)")
    if prices.empty:
        sys.exit("No dates where every ticker has a price. Try a later --start.")

    prices.index.name = "Date"
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    prices.round(4).to_csv(out)

    print(f"\nwrote {out}")
    print(f"  {len(prices)} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")

    daily = prices.pct_change().dropna()
    print(f"\n  {'ticker':<8}{'ann return':>12}{'ann vol':>10}{'total':>10}")
    for t in prices.columns:
        mu = daily[t].mean() * 252
        sd = daily[t].std() * (252 ** 0.5)
        tot = prices[t].iloc[-1] / prices[t].iloc[0] - 1
        print(f"  {t:<8}{mu*100:>11.1f}%{sd*100:>9.1f}%{tot*100:>9.1f}%")

    print("\nnext:")
    print("  python main.py")
    print("  python build_dashboard_data.py && python build_dashboard.py")


if __name__ == "__main__":
    main()
