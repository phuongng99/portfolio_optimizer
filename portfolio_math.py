import numpy as np 

def portfolio_return(weights, expected_returns):
    """Calculate the expected return of a portfolio given weights and expected returns."""
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    return float(np.dot(w, mu))

def portfolio_volatility(weights, covariance_matrix):
    """Calculate the volatility (standard deviation) of a portfolio given weights and covariance matrix."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance_matrix, dtype=float)
    return float(np.sqrt(w @ cov @ w))

def sharpe_ratio(weights, expected_returns, covariance_matrix, risk_free_rate = 0):
    """
    Compute the Sharpe ratio of a portfolio given weights, expected returns, covariance matrix, and risk-free rate.
    The Sharpe ratio is defined as the excess return (portfolio return - risk-free rate)
    divided by the portfolio volatility.
    """
    returns = portfolio_return(weights, expected_returns)
    volatility = portfolio_volatility(weights, covariance_matrix)
    return (returns - risk_free_rate) / volatility if volatility >0 else 0 