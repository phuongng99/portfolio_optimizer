import numpy as np

from portfolio_optimizer.portfolio_math import portfolio_return, portfolio_volatility, sharpe_ratio

from scipy.optimize import minimize

def equal_weighted_portfolio(num_assets):
    """Generate equal weights for a portfolio with num_assets."""
    return np.ones(num_assets) / num_assets

def _as_cov(cov_matrix):
    """Coerce to a float ndarray and reject anything that is not square."""
    cov = np.asarray(cov_matrix, dtype = float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance matrix must be square 2-D, got shape {cov.shape}")
    return cov

def _as_mu(expected_returns, num_assets):
    """Coerce to a float ndarray; catches expected_returns/cov_matrix passed the wrong way round."""
    mu = np.asarray(expected_returns, dtype = float)
    if mu.shape != (num_assets,):
        raise ValueError(f"expected_returns must have shape ({num_assets},), got {mu.shape}")
    return mu

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

    cov = _as_cov(cov_matrix)
    num_assets = cov.shape[0]
    w0 = equal_weighted_portfolio(num_assets)
    constraints = [_sum_to_one()]
    bounds = _bounds(num_assets, bounds)
    res = minimize(portfolio_volatility, w0, args = (cov, ), method = 'SLSQP',
                   bounds = bounds, constraints = constraints)
    return res.x if res.success else None

def min_vol_target_return(cov_matrix, expected_returns, target_returns, bounds = None):

    cov = _as_cov(cov_matrix)
    num_assets = cov.shape[0]
    mu = _as_mu(expected_returns, num_assets)
    w0 = equal_weighted_portfolio(num_assets)
    constraints = [_sum_to_one(),
                   {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - target_returns}]
    bounds = _bounds(num_assets, bounds)
    res = minimize(portfolio_volatility, w0, args = (cov,),
                   method = 'SLSQP', bounds = bounds, constraints = constraints)
    return res.x if res.success else None


def max_sharpe(cov_matrix, expected_returns, risk_free_rate = 0, bounds = None):
    cov = _as_cov(cov_matrix)
    n_assets = cov.shape[0]
    mu = _as_mu(expected_returns, n_assets)
    w0 = equal_weighted_portfolio(n_assets)


    def negative_sharpe(w):
        return -sharpe_ratio(w, mu, cov, risk_free_rate)
    bounds = _bounds(n_assets, bounds)
    constraints = [_sum_to_one()]
    res = minimize(negative_sharpe, w0, method = 'SLSQP', bounds = bounds, constraints = constraints)
    return res.x if res.success else None


def efficient_frontier(cov_matrix, expected_returns, n_points = 100, bounds = None):

    cov = _as_cov(cov_matrix)
    mu = _as_mu(expected_returns, cov.shape[0])
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


def inverse_volatility(cov_matrix):
    """
    Inverse volatility weighting: weight assets inversely proportional to volatility.
    Assets with lower volatility receive higher weights.
    weights_i = (1/σ_i) / Σ(1/σ_j)

    Ignores bounds - it is a closed form, not an optimisation.
    """
    cov = _as_cov(cov_matrix)
    variances = np.diag(cov)
    if np.any(variances <= 0):
        raise ValueError(f"every asset needs a positive variance; "
                         f"assets {np.flatnonzero(variances <= 0).tolist()} do not")
    weights = 1.0 / np.sqrt(variances)
    return weights / weights.sum()

def risk_budget( cov_matrix, budgets=None, max_iter=500, tol=1e-12):
    """Weights where each asset supplies its assigned share of total risk.
    budgets=None means equal shares (risk parity).

    Ignores bounds - weight caps would break the risk-contribution property.
    """

    cov = _as_cov(cov_matrix)
    n = cov.shape[0]
    if budgets is None:
        budgets = np.ones(n) / n
    else:
        budgets = np.asarray(budgets, dtype = float)
        if budgets.shape != (n,):
            raise ValueError(f"budgets must have shape ({n},), got {budgets.shape}")
        if np.any(budgets < 0) or budgets.sum() <= 0:
            raise ValueError("budgets must be non-negative and not all zero")
    w = inverse_volatility(cov)
    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(n):
            #c  = (Σw)_i − Σ_ii · w_i     # cross terms only
            c =  cov[i, :] @ w - cov[i, i] * w[i]
            # w_i = (−c + √(c² + 4·Σ_ii·b_i)) / (2·Σ_ii)
            w[i] = (-c + np.sqrt(c**2 + 4 * cov[i, i] * budgets[i])) / (2 * cov[i, i])
        if np.abs(w - w_old).max() < tol:
            break
    return w/w.sum()




def risk_parity(cov_matrix, tol = 1e-8, bounds = None):

    cov = _as_cov(cov_matrix)
    n = cov.shape[0]
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

def compare_strategies(cov_matrix, expected_returns, risk_free_rate = 0.02, bounds = None):
    """Run every strategy on one (cov, mu) pair and report its metrics.

    Equal-Weight, Risk-Budget and Inverse-Volatility cannot honour weight
    bounds, so they are omitted when bounds is given rather than returned in
    violation of it.
    """

    cov = _as_cov(cov_matrix)
    n_assets = cov.shape[0]
    mu = _as_mu(expected_returns, n_assets)
    strategies = {
        "Min-Variance": min_variance(cov, bounds),
        "Max-Sharpe": max_sharpe(cov, mu, risk_free_rate, bounds),
        "Risk-Parity": risk_parity(cov, bounds=bounds)
    }
    if bounds is None:
        strategies["Equal-Weight"] = equal_weighted_portfolio(n_assets)
        strategies["Risk-Budget"] = risk_budget(cov)
        strategies["Inverse-Volatility"] = inverse_volatility(cov)

    out= []

    for name, weights in strategies.items():
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
