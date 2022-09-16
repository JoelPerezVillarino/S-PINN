import numpy as np


class xVAData(object):

    def __init__(
        self, 
        lamB: float, 
        lamC: float, 
        rB: float, 
        rC: float,
    ) -> None:
        
        self.lamB = lamB
        self.lamC = lamC
        self.rB = rB
        self.rC = rC
        
    def source_function(self, u: np.ndarray, v: np.ndarray):
        hatV_min, hatV_max = np.minimum(u + v, 0.), np.maximum(u + v, 0.)
        val = (1. - self.rB) * self.lamB * hatV_min + (1. - self.rB) * self.lamB * hatV_max \
            + (1. - self.rC) * self.lamC * hatV_max
        return - val
    
    def xva_correction(self, tau: float):
        term = self.lamB * (1. - self.rB) + self.lamC * (1. - self.rC)
        return np.exp(- tau * term)
