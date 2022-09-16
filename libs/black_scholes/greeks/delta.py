from typing import Union
import numpy as np
import scipy.stats as st


def deltaBlackScholesMerton(
    S:np.ndarray,
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    q: float,
    flag="c"
    ):
    """Compute delta greeks of Black-Scholes-Merton model"""
    d1 = (np.log(S / strike) + (r - q + 0.5 * np.power(sigma, 2)) * tau) \
        / (sigma * np.sqrt(tau))
    N = st.norm.cdf(d1)
    if flag == "c":
        return np.exp(- q * tau) * N
    elif flag == "p":
        return np.exp(-q * tau) * (N - 1.)
    else:
        raise ValueError("flag must be c for call or p for put!")