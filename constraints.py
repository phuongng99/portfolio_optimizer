
import numpy as np
class PortfolioConstraints:

    def __init__(self, max_weight = 1.0, min_weight = 0.0):
        self.max_weight = max_weight
        self.min_weight = min_weight

    def get_bounds(self, num_assets):
        """Return bounds for each asset in the portfolio.
        Each asset's weight is constrained between min_weight and max_weight.
        """
        return [(self.min_weight, self.max_weight) for _ in range(num_assets)]

    def is_feasible(self, weights):
        """Check if the given weights satisfy the constraints."""
        w = np.asarray(weights)
        return all(self.min_weight <= w <= self.max_weight for w in weights) and abs(sum(weights) - 1) < 1e-6