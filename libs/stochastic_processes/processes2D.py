
class HestonProcess(object):

    def __init__(
        self, 
        r: float, 
        q: float, 
        kappa: float, 
        eta: float, 
        sigma: float, 
        rho: float
    ) -> None:
        
        self.r = r
        self.q = q
        self.rR = self.r - self.q
        # Variance parameters
        self.kappa = kappa
        self.eta = eta
        self.sigma = sigma
        # Correlation between brownian motions
        self.rho = rho

    def checkFeller(self):
        if 2. * self.kappa * self.eta > self.sigma * self.sigma:
            return True
        else:
            return False


class LognormalProcess2D(object):
    """ Dynamics: dS_t = (r - q) S_t dt + sigma S_t dW_t. r, q, sigma are constants """
    def __init__(self, r: float, q: list, sigma: list, rho: float, name=None) -> None:
        self._dim = len(q)
        self._name = name

        if self._dim != len(sigma):
            raise ValueError("rR and sigma must be the same dimension.")
        
        # Deriva params
        self.r = r
        self.q = q
        self.rR = [self.r - self.q[0], self.r - self.q[1]]
        # Volatility params
        self.sigma = sigma
        # Correlation between brownian motions
        self.rho = rho