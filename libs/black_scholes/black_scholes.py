from typing import Union
import numpy as np
import scipy.stats as st


def blackScholesMerton(
    S: Union[float, np.ndarray],
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    q: float,
    flag="c"
    ):
    """
    Return the Black-Scholes-Merton option price
    :param S: stock value, S >= 0
    :param strike: fixed strike, K >= 0
    :param tau: time to maturity
    :param r: risk-free interest rate
    :param sigma: volatility
    :param q: some kind of dividend yield
    :param phi: 1 for call, -1 for put
    :return: Black Scholes value for the given inputs
    """
    if flag == "c":
        alpha = 1
    elif flag == "p":
        alpha = -1
    else:
        raise ValueError("flag must be c for call or p for put!")
    if np.isclose(tau, 0.):
        return np.maximum(alpha * (S - strike), 0.)
    
    d1 = (np.log(S / strike) + (r - q + 0.5 * np.power(sigma, 2)) * tau) \
        / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    V = alpha * S * np.exp(- q * tau) * st.norm.cdf(alpha * d1) \
        - alpha * strike * np.exp(-r * tau) * st.norm.cdf(alpha * d2)

    return V


def xVABlackScholesMerton(
    S: Union[float, np.ndarray],
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    q: float,
    lamB: float,
    lamC: float,
    rB: float,
    rC: float,
    flag="c"
):
    adjustment = - (1. - np.exp(- tau * (lamB * (1. - rB) + lamC * (1. - rC))))
    v = blackScholesMerton(S, strike, tau, r, sigma, q, flag)
    return adjustment * v
