from typing import Union, Tuple
import numpy as np
from scipy import sparse

import libs.black_scholes as bs
from libs.stochastic_processes.processes2D import HestonProcess
from ..utils.payoffs import VanillaPayoff
from ..utils import xVAData
from .fd_parabolic_2d import Parabolic2D
from ..time_engines.crank_nicolson import CrankNicolson, FixedPointCrankNicolson


class FdVanillaHeston(Parabolic2D):

    def __init__(
        self,
        CP: str,
        strike: float, 
        process: HestonProcess,
        xGrid_data: Tuple[float, float, float], 
        yGrid_data: Tuple[float, float, float], 
        tauGrid_data: Tuple[float, float, float], 
        method=CrankNicolson()
    ) -> None:

        super().__init__(xGrid_data, yGrid_data, tauGrid_data)

        if not isinstance(process, HestonProcess):
            raise ValueError("process must be a instance of HestonProcess")
        print(f"Is Feller condition satisfied? {process.checkFeller()}")

        self.process = process
        self.strike = strike
        self.payoff = VanillaPayoff(CP, self.strike)
        self.method = method
    

    def solve(self, ):
        # Payoff values
        for i in range(self.nx):
            self.uSol[i * self.ny: (i + 1) * self.ny, 0] = self.payoff(self.xGrid[i])
        # Applying fd scheme
        self.method(self)
        return

    def _build_matrix(self, ):
        # Allocate sparse matrix, A, and fill it
        A = sparse.lil_matrix((self.nx * self.ny, self.nx * self.ny), )

        indy = self.ny // 2 
        # i = 1 ... nx-2, j = 1 ... ny - 2
        for i in range(1, self.nx - 1):
            bx = self._bx_term(i)
            for j in range(1, self.ny-1):
                ax = self._ax_term(i, j)
                ay = self._ay_term(j)
                by = self._by_term(j)
                p = self._pxy_term(i, j)

                A[i * self.ny + j, (i - 1) * self.ny + j] = ax - bx
                A[i * self.ny + j, (i + 1) * self.ny + j] = ax + bx
                A[i * self.ny + j, i * self.ny + j - 1] = ay - by
                A[i * self.ny + j, i * self.ny + j + 1] = ay + by
                A[i * self.ny + j, i * self.ny + j] = - (2. * ax + 2. * ay 
                                                        + self.process.r)

                A[i * self.ny + j, (i + 1) * self.ny + j + 1] = p
                A[i * self.ny + j, (i + 1) * self.ny + j - 1] = - p
                A[i * self.ny + j, (i - 1) * self.ny + j + 1] = - p
                A[i * self.ny + j, (i - 1) * self.ny + j - 1] = p
            """
            for j in range(indy, self.ny - 1):
                ax = self._ax_term(i, j)
                ay = self._ay_term(j)
                by = self._by_term(j)
                p = self._pxy_term(i, j)

                A[i * self.ny + j, (i - 1) * self.ny + j] = ax - bx
                A[i * self.ny + j, (i + 1) * self.ny + j] = ax + bx
                A[i * self.ny + j, i * self.ny + j - 2] = by
                A[i * self.ny + j, i * self.ny + j - 1] = ay - 4. * by
                A[i * self.ny + j, i * self.ny + j] = - 2. * ax - 2. * ay + 3. * by \
                                                     - self.process.r

                A[i * self.ny + j, (i + 1) * self.ny + j + 1] = p
                A[i * self.ny + j, (i + 1) * self.ny + j - 1] = - p
                A[i * self.ny + j, (i - 1) * self.ny + j + 1] = - p
                A[i * self.ny + j, (i - 1) * self.ny + j - 1] = p
            """
        # Boundary S = 0 (i = 0, j = 1 ... ny-1)
        i = 0
        for j in range(1, self.ny-1):
            ay = self._ay_term(j)
            by = self._by_term(j)

            A[i * self.ny + j, i * self.ny + j - 1] = ay - by
            A[i * self.ny + j, i * self.ny + j + 1] = ay + by
            A[i * self.ny + j, i * self.ny + j] = - (2. * ay + self.process.r)

        
        # Boundary V = 0 (i = 1 ... nx - 2, j = 0)
        j = 0
        by = self._by_term(j)
        for i in range(1, self.nx-1):
            bx = self._bx_term(i)

            A[i * self.ny + j, (i - 1) * self.ny + j] = - bx
            A[i * self.ny + j, (i + 1) * self.ny + j] = bx
            A[i * self.ny + j, i * self.ny + j + 2] = - by
            A[i * self.ny + j, i * self.ny + j + 1] = 4. * by
            A[i * self.ny + j, i * self.ny + j] = - (self.process.r + 3. * by)
        
        return A
    
    def _build_boundary_conditions(self, A: sparse.lil_matrix, ):
        # Boundary S = 0, V = 0 (i = 0, j = 0)
        i, j = 0, 0
        A[i * self.ny + j, i * self.ny + j] = 1.
        """
        # Boundary V = 0 (i = 1 ... nx - 1, j = 0)
        j = 0
        for i in range(1, self.nx-1):
            A[i * self.ny + j, i * self.ny + j] = 1.
            A[i * self.ny + j, i * self.ny + j + 1] = -1.
        """
        # Boundary S = S_max (i = nx - 1, j = 0 ... ny - 2)
        i = self.nx - 1
        for j in range(self.ny - 1):
            A[i * self.ny + j, i * self.ny + j] = 1.
            A[i * self.ny + j, (i - 1) * self.ny + j] = -1.
           
        # Boundary V = V_max (i = 1 .. nx - 1, j = ny - 1)
        j = self.ny - 1
        for i in range(self.nx):
            A[i * self.ny + j, i * self.ny + j] = 1.
            A[i * self.ny + j, i * self.ny + j - 1] = -1.
    
    def _build_right_hand(
        self, 
        v: np.ndarray,
        tau: float, 
    ):
        # Rewrite v array with the boundary conditions
        # Boundary S = 0, V = 0
        i, j = 0, 0
        if self.payoff.CP == "c":
            v[i * self.ny + j] = 0.
        else:
            v[i * self.ny + j] = self.payoff.strike * np.exp(- tau * self.process.r)
        """
        # Boundary V = 0 (i = 1 ... nx - 1, j = 0)
        j = 0
        for i in range(1, self.nx-1):
            v[i * self.ny + j] = 0.
        """
        # Boundary S = S_max (i = nx - 1, j = 0 ... ny - 2)
        i = self.nx - 1
        for j in range(self.ny - 1):
            v[i * self.ny + j] = self.hx * bs.deltaBlackScholesMerton(
                self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[j]),
                 self.process.q, self.payoff.CP
            )
        # Boundary V = V_max (i = 0 .. nx - 1, j = ny - 1)
        j = self.ny - 1
        for i in range(self.nx):
            v[i * self.ny + j] = 0.
            #slope = bs.vegaBlackScholesMerton(
            #    self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[j]), self.process.q
            #)
            #v[i * self.ny + j] = self.hy * slope


    def _ax_term(self, i, j):
        term = np.power(self.xGrid[i], 2) * self.yGrid[j] / (2. * np.power(self.hx, 2))
        return term
    
    def _ay_term(self, j):
        term = np.power(self.process.sigma, 2) * self.yGrid[j] \
             / (2. * np.power(self.hy, 2))
        return term
    
    def _pxy_term(self, i, j):
        term = self.process.rho * self.process.sigma * self.xGrid[i] * self.yGrid[j] \
             / (4. * self.hx * self.hy)
        return term
    
    def _bx_term(self, i):
        term = (self.process.r - self.process.q) * self.xGrid[i] / (2. * self.hx)
        return term
    
    def _by_term(self, j):
        term = self.process.kappa * (self.process.eta - self.yGrid[j]) / (2. * self.hy)
        return term


class FdxVAHeston(FdVanillaHeston):

    def __init__(
        self, 
        CP: str, 
        strike: float, 
        process: HestonProcess,
        xva_data: xVAData,
        xGrid_data: Tuple[float, float, float], 
        yGrid_data: Tuple[float, float, float], 
        tauGrid_data: Tuple[float, float, float], 
        method=FixedPointCrankNicolson()
    ) -> None:
        super().__init__(CP, strike, process, xGrid_data, yGrid_data, tauGrid_data, method)
        self.xva_data = xva_data
        self.vSol = None

    def solve(self, vSol: np.ndarray):
        self.vSol = vSol
        self.method(self, vSol)
    
    def get_solution(self, index=None):
        return self.uSol[:, index] + self.vSol[:, index]
    
    def get_xva(self, index=None):
        return self.uSol[:, index]
    
    def _build_matrix(self):
        return super()._build_matrix()
    
    def _build_boundary_conditions(self, A: sparse.lil_matrix):
        super()._build_boundary_conditions(A)
    
    def _build_right_hand(self, v: np.ndarray, tau: float):
        # Compute xVA correction
        xva_correction = self.xva_data.xva_correction(tau)
        # Boundary S = 0, V = 0
        i, j = 0, 0
        if self.payoff.CP == "c":
            v[i * self.ny + j] = 0.
        else:
            aux_v = self.payoff.strike * np.exp(- tau * self.process.r)
            v[i * self.ny + j] = - aux_v + aux_v * xva_correction
        """
        # Boundary V = 0 (i = 1 ... nx - 2, j = 0)
        j = 0
        for i in range(1, self.nx - 1):
            v[i * self.ny + j] = 0.
        """
        # Boundary S = S_max (i = nx - 1, j = 0 ... ny - 2)
        i = self.nx - 1
        for j in range(0, self.ny - 1):
            slope = (-1. + xva_correction) * bs.deltaBlackScholesMerton(
                self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[j]),
                 self.process.q, self.payoff.CP
            )
            v[i * self.ny + j] = self.hx * slope
        # Boundary V = V_max (i = 0 .. nx - 1, j = ny - 1)
        j = self.ny - 1
        for i in range(1, self.nx):
            v[i * self.ny + j] = 0.