"""Rolling backtester: rebalance on a schedule using only past data."""

import numpy as np
import pandas as pd

from portfolio_optimizer.data_loader import (
    annualize_returns, annualize_volatility, calculate_covariance, calculate_returns,
)
from portfolio_optimizer.portfolio_math import sharpe_ratio
from portfolio_optimizer.optimizer import max_sharpe

class PortfolioBacktester:
    """Simulates a rebalancing strategy on historical prices.

    optimizer_func(expected_returns, cov_matrix) → weights (numpy array).
    """

    def __init__(self, optimizer_func, rebalance_freq="M",
                 starting_value=100_000.0, cost_model=None,
                 lookback_days=126):
        self.optimizer_func = optimizer_func
        self.rebalance_freq = rebalance_freq  # 'M' = month, 'Q' = quarter, 'D' = every day
        self.starting_value = starting_value
        self.cost_model = cost_model
        self.lookback_days = lookback_days

    def _is_rebalance_day(self, prev_date, this_date):
        if prev_date is None:
            return True
        if self.rebalance_freq == "M":
            return prev_date.month != this_date.month
        if self.rebalance_freq == "Q":
            return (prev_date.month - 1) // 3 != (this_date.month - 1) // 3
        return True

    def run(self, prices):
        """Replay `prices` day by day.

        """
        returns = calculate_returns(prices)
        dates   = returns.index
        n_assets = prices.shape[1]
        weights = np.ones(n_assets)/n_assets
        value = self.starting_value
        self.equity_curve = []
        self.rebalance_log = [] 
        self.total_cost = 0.0
        prev_date = None 
        for i, date in enumerate(dates):
            if self._is_rebalance_day(prev_date, date) and i>= self.lookback_days:
                window = returns.iloc[i - self.lookback_days: i]
                mu = annualize_returns(window).values
                cov = calculate_covariance(window).values
                target = np.asarray(self.optimizer_func(mu, cov))
                cost = 0.0
                if self.cost_model is not None:
                    cost = self.cost_model.rebalance_cost(weights, target, value)
                    value -= cost
                    self.total_cost += cost
                # Record what we were actually holding when the trade was placed,
                # not just where we were going. Turnover has to be measured against
                # the drifted book - measuring it against the previous *target*
                # ignores the drift and misstates how much really changed hands.
                self.rebalance_log.append({
                    "date": date,
                    "weights": target,
                    "held": weights.copy(),
                    "turnover": float(np.sum(np.abs(target - weights)) / 2),
                    "cost": float(cost),
                })
                weights = target
            
            asset_returns = returns.iloc[i].values
            day_return = float(np.dot(weights, asset_returns))
            value *= (1 + day_return)

            # Weights DRIFT with prices. Winners become a bigger share of the
            # book on their own; you are not holding the target mix between
            # rebalance dates. Without this line the simulation silently
            # rebalances back to target every single day, for free - which
            # understates both turnover and transaction costs.
            weights = weights * (1 + asset_returns) / (1 + day_return)

            self.equity_curve.append((date, value))
            prev_date = date

        self.equity_curve = pd.DataFrame(self.equity_curve, columns=["date", "value"]).set_index("date")

    def metrics(self, risk_free_rate=0.02):
        """Total return, annualised return, annualised vol, Sharpe, max drawdown.

        TODO (Step 16).
        """
        returns = calculate_returns(self.equity_curve['value'])
        total_return = float(self.equity_curve['value'].iloc[-1]/ self.equity_curve['value'].iloc[0] - 1)
        annualize_return = annualize_returns(returns)
        annualize_vol = annualize_volatility(returns)
        sharpe = (annualize_return - risk_free_rate)/annualize_vol
       
        drawdown = self.max_drawdown()
        return {
            "total_return": total_return, 
            "annualized_return": annualize_return,
            "annualized_volatility": annualize_vol,
            "sharpe_ratio": sharpe, 
            "max_drawdown": drawdown,
            "total_costs": self.total_cost
        }


        pass

    def max_drawdown(self):
        """Largest peak-to-trough dip as a fraction (negative).

        TODO (Step 16).
        
        """
        peak = self.equity_curve['value'].cummax()
        drawdown = self.equity_curve['value']/peak - 1
        return float(drawdown.min())
        pass
