from typing import Tuple
import numpy as np
from scipy import sparse

from libs.stochastic_processes import LognormalProcess2D
from . import Parabolic2D


class FdBlackScholes2D(Parabolic2D):

    def __init__(
        self,
        process: LognormalProcess2D, 
        xGrid_data: Tuple[float, float, float], 
        yGrid_data: Tuple[float, float, float], 
        tauGrid_data: Tuple[float, float, float]
    ) -> None:
        super().__init__(xGrid_data, yGrid_data, tauGrid_data)
        self.process = process
    
    def solve(self):
        super().solve()
    
    def _build_matrix(self):
        # Allocate sparse matrix, A, and fill it
        A = sparse.lil_matrix((self.nx * self.ny, self.nx * self.ny), )
        # i = 1... nx-2, j = 1... ny - 2
        for i in range(1, self.nx - 1):
            ax, bx = self._ax_term(i), self._bx_term(i)
            for j in range(1, self.ny - 1):
                ay, by = self._ay_term(j), self._by_term(j)
                p = self._p_term(i, j)

                A[i * self.ny + j, (i - 1) * self.ny + j] = ax - bx
                A[i * self.ny + j, (i + 1) * self.ny + j] = ax + bx
                A[i * self.ny + j, i* self.ny + j - 1] = ay - by
                A[i * self.ny + j, i* self.ny + j + 1] = ay + by
                A[i * self.ny + j, i * self.ny + j] = - 2. * ax - 2. * ay - self.process.r

                A[i * self.ny + j, (i + 1) * self.ny + j + 1] = p
                A[i * self.ny + j, (i - 1) * self.ny + j + 1] = - p
                A[i * self.ny + j, (i + 1) * self.ny + j - 1] = -p
                A[i * self.ny + j, (i - 1) * self.ny + j - 1] = p
        
        return A
    
    def _build_boundary_conditions(
        self, 
        A: sparse.lil_matrix):
        super()._build_boundary_conditions(A)
    
    def _build_right_hand(
        self, 
        v: np.ndarray, 
        tau: float
    ):
        super()._build_right_hand(v, tau)
    
    def _ax_term(self, i):
        return np.power(self.xGrid[i], 2) * np.power(self.process.sigma[0], 2) \
            / (2. * np.power(self.hx, 2))
    
    def _ay_term(self, j):
        return np.power(self.yGrid[j], 2) * np.power(self.process.sigma[1], 2) \
            / (2. * np.power(self.hy, 2))
    
    def _bx_term(self, i):
        return self.process.rR[0] * self.xGrid[i] / (2. * self.hx)
    
    def _by_term(self, j):
        return self.process.rR[1] * self.yGrid[j] / (2. * self.hy)
    
    def _p_term(self, i, j):
        return self.process.rho * self.process.sigma[0] * self.process.sigma[1] \
            * self.xGrid[i] * self.yGrid[j] / (4. * self.hx * self.hy)