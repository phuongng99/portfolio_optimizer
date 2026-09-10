"""Risk decomposition and diversification metrics."""

import numpy as np


def risk_contribution(weights, cov_matrix):
    """Absolute risk contribution per asset.

    RC_i = w_i * (Σw)_i / σ_p

    Returns a numpy array. Should sum to σ_p.
    """
    w = np.asarray(weights)
    cov = np.asarray(cov_matrix)
    sigma_p = float(np.sqrt(w @ cov @ w))
    marginal = cov @ w 
    return w * marginal / sigma_p

    pass


def pct_risk_contribution(weights, cov_matrix):
    """Percentage of total risk from each asset.

    Should sum to 100.

    TODO (Step 17):
        rc = risk_contribution(weights, cov_matrix)
        return rc / rc.sum() * 100.0
    """
    rc = risk_contribution(weights, cov_matrix)
    return rc/rc.sum() * 100
    pass


def diversification_ratio(weights, cov_matrix):
    """Σ(w_i · σ_i) / σ_p.

    > 1 means diversification is reducing risk (higher is better).
    """
    w = np.asarray(weights)
    cov = np.asarray(cov_matrix)
    individual_vol = np.sqrt(np.diag(cov))
    weighted_avg_vol = float(np.dot(weights, individual_vol))
    portfolio_vol = float(np.sqrt(w @ cov @ w))
    return  weighted_avg_vol/portfolio_vol

    