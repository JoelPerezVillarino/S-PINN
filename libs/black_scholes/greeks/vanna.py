import numpy as np
import scipy.stats as st

from .vega import vegaBlackScholesMerton


def vannaBlackScholesMerton( 
    S: np.ndarray,
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    q: float,
) -> np.ndarray:
    """ vanna is the second order partial derivative of V with respect to S and sigma."""
    vega = vegaBlackScholesMerton(S, strike, tau, r, sigma, q)
    d1 = (np.log(S / strike) + (r - q + 0.5 * np.power(sigma, 2)) * tau) \
        / (sigma * np.sqrt(tau))
    vanna = vega / S * (1. - d1 / (sigma * np.sqrt(tau)))
    return vanna