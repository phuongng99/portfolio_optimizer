import numpy as np 

from portfolio_optimizer.portfolio_math import portfolio_return, portfolio_volatility, sharpe_ratio     

from scipy.optimize import minimize

def equal_weighted_portfolio(num_assets):
    """Generate equal weights for a portfolio with num_assets."""
    return np.ones(num_assets) / num_assets

def _bounds(num_assets, bounds = None):
    """Generate bounds for weights: each weight between 0 and 1."""
    if bounds is None:
        return [(0, 1) for _ in range(num_assets)]
    return bounds

def _sum_to_one():
    """Constraint: weights must sum to 1."""
    return {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

def min_variance(cov_matrix, bounds = None):
    """
    Find the portfolio weights that minimize variance given a covariance matrix and optional bounds.(Long only minimum variance portfolio)
    Uses the SLSQP optimization method from scipy.optimize.minimize.
    """

    num_assets = cov_matrix.shape[0]
    w0 = equal_weighted_portfolio(num_assets)
    constraints = [_sum_to_one()]
    bounds = _bounds(num_assets, bounds)
    cov = np.asarray(cov_matrix)
    res = minimize(portfolio_volatility, w0, args = (cov, ), method = 'SLSQP',
                   bounds = bounds, constraints = constraints)
    return res.x if res.success else None

def min_vol_target_return(cov_matrix, expected_returns, target_returns, bounds = None):

    num_assets = cov_matrix.shape[0]
    cov = np.asarray(cov_matrix)
    mu = np.asarray(expected_returns)
    w0 = equal_weighted_portfolio(num_assets)
    constraints = [_sum_to_one(),
                   {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - target_returns}]
    bounds = _bounds(num_assets, bounds)
    res = minimize(portfolio_volatility, w0, args = (cov,),
                   method = 'SLSQP', bounds = bounds, constraints = constraints)
    return res.x if res.success else None


def max_sharpe(cov_matrix, expected_returns, risk_free_rate = 0, bounds = None):
    n_assets = cov_matrix.shape[0]
    cov = np.asarray(cov_matrix)
    mu = np.asarray(expected_returns)
    w0 = equal_weighted_portfolio(n_assets)


    def negative_sharpe(w):
        return -sharpe_ratio(w, mu, cov, risk_free_rate)
    bounds = _bounds(n_assets, bounds)
    constraints = [_sum_to_one()]
    res = minimize(negative_sharpe, w0, method = 'SLSQP', bounds = bounds, constraints = constraints)
    return res.x if res.success else None


def efficient_frontier(expected_returns, cov_matrix, n_points = 100, bounds = None):

    mu = np.asarray(expected_returns)
    cov = np.asarray(cov_matrix)
    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    vol, res, weights = [], [], []
    for target in target_returns:
        w = min_vol_target_return(cov, mu, target, bounds)
        if w is  None:
            continue
        weights.append(w)
        vol.append(portfolio_volatility(w, cov))
        res.append(portfolio_return(w, mu))

    return {"Volatilities": np.array(vol), 
            "Returns": np.array(res),
            "Weights": np.array(weights)}


def risk_parity(cov_matrix, tol = 1e-8, bounds = None):

    n = cov_matrix.shape[0]
    cov = np.asarray(cov_matrix)
    target = 1/n
    def objective(w):
        sigma_p = np.sqrt(w @ cov @ w + tol)
        marginal_risk_contributions = cov @ w 
        risk_contributions = w * marginal_risk_contributions / sigma_p
        pct = risk_contributions / sigma_p
        return np.sum((pct - target)**2)
    w0 = equal_weighted_portfolio(n)
    bounds = _bounds(n, bounds)
    constraints = [_sum_to_one()]
    res = minimize(objective, w0, method = 'SLSQP', bounds = bounds, constraints = constraints)
    return res.x if res.success else None

def compare_strategies(expected_returns, cov_matrix, risk_free_rate = 0.02, bounds = None):

    mu = np.asarray(expected_returns)
    cov = np.asarray(cov_matrix)
    n_assets = cov.shape[0]
    min_var_weights = min_variance(cov, bounds)
    max_sharpe_weights = max_sharpe(cov, mu, risk_free_rate, bounds)
    risk_parity_weights = risk_parity(cov, bounds=bounds)
    stategies = {
        "Equal-Weight": equal_weighted_portfolio(n_assets),
        "Min-Variance": min_var_weights,
        "Max-Sharpe": max_sharpe_weights,
        "Risk-Parity": risk_parity_weights
    }

    out= []

    for name, weights in stategies.items():
        if weights is None:
            continue 
        out.append({
            "name": name,
            "weights": weights, 
            "returns": portfolio_return(weights, mu),
            "volatility": portfolio_volatility(weights, cov),
            "sharpe": sharpe_ratio(weights, mu, cov, risk_free_rate)
        })

    return out
