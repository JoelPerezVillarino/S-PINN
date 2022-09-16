import numpy as np


class Payoff(object):
    
    def __init__(self, CP: str, strike: float) -> None:
        if CP == "c":
            self.CP = CP
        elif CP == "p":
            self.CP = CP
        else:
            raise ValueError("CP must be c for Call, p for Put!")

        self.strike = strike
    
    def __call__(self, *args, **kwds):
        pass


class VanillaPayoff(Payoff):

    def __init__(self, CP: str, strike: float) -> None:
        super().__init__(CP, strike)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.CP == "c":
            return np.maximum(x - self.strike, 0.)
        return np.maximum(self.strike - x, 0.)


class ArithmeticBasketPayoff(Payoff):

    def __init__(self, CP: str, strike: float) -> None:
        super().__init__(CP, strike)
        # weights
        self._w1 = 0.5
        self._w2 = 0.5
    
    def __call__(self, x1: float, x2: float) -> np.ndarray:
        if self.CP == "c":
            return np.maximum(self._w1 * x1 + self._w2 * x2 - self.strike, 0.)
        return np.maximum(self.strike - self._w1 * x1 - self._w2 * x2, 0.)


class WorstOfPayoff(Payoff):

    def __init__(self, CP: str, strike: float) -> None:
        super().__init__(CP, strike)
    
    def __call__(self, x1: float, x2: float) -> np.ndarray:
        if self.CP == "c":
            return np.maximum(np.minimum(x1, x2) - self.strike, 0.)
        return np.maximum(self.strike - np.minimum(x1, x2), 0.)

        
        
    
    

