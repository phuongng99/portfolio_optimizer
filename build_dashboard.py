"""Build the interactive dashboard as a single self-contained HTML file.

    python build_dashboard.py            # build, then open it in your browser
    python build_dashboard.py --no-open  # just build it

It reads dashboard_data.json (run build_dashboard_data.py first) and injects
it into dashboard/template.html. The result opens straight off disk - no web
server needed, because the data is embedded rather than fetched.

Typical loop after new prices land in data/stock_prices.csv:

    python build_dashboard_data.py     # re-run every strategy through the backtester
    python build_dashboard.py          # rebuild the page and open it
"""

import json
import os
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dashboard_data.json")
TEMPLATE = os.path.join(HERE, "dashboard", "template.html")
OUTPUT = os.path.join(HERE, "dashboard.html")

PLACEHOLDER = "/*__DATA__*/null/*__END__*/"


def main():
    if not os.path.exists(DATA):
        sys.exit("dashboard_data.json not found - run:  python build_dashboard_data.py")
    if not os.path.exists(TEMPLATE):
        sys.exit(f"template not found at {TEMPLATE}")

    with open(DATA) as f:
        payload = json.load(f)

    # Trim the random scatter cloud so the page stays light.
    payload["cloud"] = payload["cloud"][:700]
    blob = json.dumps(payload, separators=(",", ":"))

    with open(TEMPLATE) as f:
        html = f.read()
    if PLACEHOLDER not in html:
        sys.exit("template is missing its data placeholder - did it get edited?")

    with open(OUTPUT, "w") as f:
        f.write(html.replace(PLACEHOLDER, blob))

    size = os.path.getsize(OUTPUT) / 1024
    print(f"built {OUTPUT}  ({size:.0f} KB)")
    default_variant = payload["variants"].get("none")
    if default_variant is None:
        default_variant = next(iter(payload["variants"].values()))
    for name, bt in default_variant["backtests"].items():
        m = bt["metrics"]
        print(f"  {name:<13} ${bt['equity'][-1]:>11,.0f}   sharpe {m['sharpe_ratio']:>5.2f}   "
              f"maxDD {m['max_drawdown']*100:>6.1f}%")

    if "--no-open" not in sys.argv:
        webbrowser.open("file://" + OUTPUT)
        print("opening in your browser...")


if __name__ == "__main__":
    main()
