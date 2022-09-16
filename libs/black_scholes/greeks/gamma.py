from typing import Union
import numpy as np
import scipy.stats as st


def gammaBlackScholesMerton(
    S: Union[float, np.ndarray],
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    q: float,
):
    d1 = (np.log(S / strike) + (r - q + 0.5 * np.power(sigma, 2)) * tau) \
        / (sigma * np.sqrt(tau))
    gamma = np.exp(- q * tau) / (S * sigma * np.sqrt(tau)) \
        * st.norm.cdf(d1)
    return gamma