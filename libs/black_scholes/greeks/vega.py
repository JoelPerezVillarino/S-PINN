import numpy as np
import scipy.stats as st


def vegaBlackScholesMerton( 
    S: np.ndarray,
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    q: float,
) -> np.ndarray:
    
    d1 = (np.log(S / strike) + (r - q + 0.5 * np.power(sigma, 2)) * tau) \
        / (sigma * np.sqrt(tau))
    vega = S * np.exp(-q * tau) * st.norm.pdf(d1) * np.sqrt(tau)
    return vega