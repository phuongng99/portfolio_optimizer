import numpy as np 

class TransactionCostModel:

    def __init__(self, commission_rate = 0.001, spread_cost = 0.0005, min_commission = 1.0):
        self.commission_rate = commission_rate
        self.spread_cost = spread_cost
        self.min_commission = min_commission

    def trade_cost(self, trade_value):
        """Cost for a single trade of "trade_value" dollars."""
        return max(abs(trade_value)* (self.commission_rate + self.spread_cost), 
                   self.min_commission if trade_value != 0 else 0)

    def rebalance_cost(self, current_weights, target_weights, portfolio_value):
        """
        Calculate the total transaction cost accross every asset that actually moved 
        """

        cur = np.asarray(current_weights)
        target = np.asarray(target_weights)
        trade_values = np.abs(target - cur) * portfolio_value
        return float(sum(self.trade_cost(tv) for tv in trade_values))


    def turn_over(self, current_weights, target_weights):
        """
        Calculate the turnover of a portfolio rebalance.
        Turnover is defined as half of the sum of absolute changes in weights.
        """
        cur = np.asarray(current_weights)
        target = np.asarray(target_weights)
        return float(np.sum(np.abs(target - cur)) / 2)