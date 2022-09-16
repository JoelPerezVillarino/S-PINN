
class LognormalProcess1D(object):
    """ Dynamics: dS_t = (r - q) S_t dt + sigma S_t dW_t. r, q, sigma are constants """
    def __init__(self, r: float, q: float, sigma: float, name=None) -> None:
        self._name = name

        # Deriva params
        self.r = r
        self.q = q
        self.rR = r - q
        # Volatility params
        self.sigma = sigma
