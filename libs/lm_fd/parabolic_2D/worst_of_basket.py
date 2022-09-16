from copy import deepcopy
from typing import Tuple
import numpy as np
from scipy import sparse

from libs.stochastic_processes import LognormalProcess2D
from libs import black_scholes as bs
from . import FdBlackScholes2D
from ..utils import WorstOfPayoff
from ..utils import xVAData
from ..time_engines import ImplicitEuler, CrankNicolson
from ..time_engines import FixedPointImplicitEuler, FixedPointCrankNicolson


class FdWorstOf2D(FdBlackScholes2D):

    def __init__(
        self,
        CP: str,
        strike: float, 
        process: LognormalProcess2D, 
        xGrid_data: Tuple[float, float, float], 
        yGrid_data: Tuple[float, float, float], 
        tauGrid_data: Tuple[float, float, float], 
        method="CrankNicolson"
    ) -> None:
        super().__init__(process, xGrid_data, yGrid_data, tauGrid_data)
        self.process = process
        self.strike = strike
        self.payoff = WorstOfPayoff(CP, strike)
        self.method = method
    
    def solve(self):
        # Payoff
        for i in range(self.nx):
            self.vSol[i * self.ny: (i + 1) * self.ny, :] = self.payoff(self.xGrid[i], self.yGrid)[:, None]
        # Load matrix A
        A = self._build_matrix()
        # First stage: Euler implicit (2 X htau/2 step)
        method = ImplicitEuler(
            0.5 * self.htau, self.m_dim, deepcopy(A), self._build_boundary_conditions,
            self._build_right_hand
        )
        for k in range(2):
            tau = (k + 1) * 0.5 * self.htau
            method(self.vSol, tau)
        
        # Second stage: Crank Nicolson
        method = CrankNicolson(
            self.htau, self.m_dim, A, self._build_boundary_conditions, self._build_right_hand
        )
        for k in range(2, self.ntau):
            tau = self.tauGrid[k]
            method(self.vSol, tau)

    def _build_matrix(self):
        return super()._build_matrix()
    
    def _build_boundary_conditions(self, A: sparse.lil_matrix) -> None:
        # Call option
        if self.payoff.CP == "c":
            # Boundary S1 = 0 (i = 0, j = 0 ... ny - 1)
            i = 0
            for j in range(self.ny):
                A[i * self.ny + j, i * self.ny + j] = 1.
            # Boundary S2 = 0 (i = 1 ... nx, j = 0)
            j = 0
            for i in range(1, self.nx):
                A[i * self.ny + j, i * self.ny + j] = 1.
            # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
            i = self.nx - 1
            for j in range(1, self.ny):
                A[i * self.ny +j, i * self.ny + j] = 1.
                A[i * self.ny + j, (i - 1) * self.ny + j] = -2.
                A[i * self.ny + j, (i - 2) * self.ny + j] = 1.
            # Boundary S2 = S2_max(i = 1 ... nx - 2, j = ny - 1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                A[i * self.ny + j, i * self.ny + j] = 1
                A[i * self.ny + j, i * self.ny + j - 1] = -2.
                A[i * self.ny + j, i * self.ny + j - 2] = 1.
        # Put option
        else:
            # Boundary S1 = 0 (i = 0, j = 0 ... ny - 1)
            i = 0
            for j in range(self.ny):
                A[i * self.ny + j, i * self.ny + j] = 1.
            # Boundary S2 = 0 (i = 1 ... nx, j = 0)
            j = 0
            for i in range(1, self.nx):
                A[i * self.ny + j, i * self.ny + j] = 1.
            # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
            i = self.nx - 1
            for j in range(1, self.ny):
                A[i * self.ny +j, i * self.ny + j] = 1.
                A[i * self.ny + j, (i - 1) * self.ny + j] = -1.
            # Boundary S2 = S2_max(i = 1 ... nx - 2, j = ny - 1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                A[i * self.ny + j, i * self.ny + j] = 1
                A[i * self.ny + j, i * self.ny + j - 1] = -1.
    
    def _build_right_hand(
        self, 
        v: np.ndarray, 
        tau: float
    ) -> None:
        # Rewrite array with the proper boundary conditions
        # Call option
        if self.payoff.CP == "c":
            # Boundary S1 = 0 (i = 0, j= 0 ... ny - 1)
            i = 0
            for j in range(self.ny):
                v[i * self.ny + j, :] = 0.
            # Boundary S2 = 0 (i = 1 ... nx, j = 0)
            j = 0
            for i in range(1, self.nx):
                v[i * self.ny + j, :] = 0.
            # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
            i = self.nx - 1
            for j in range(1, self.ny):
                v[i * self.ny + j, :] = 0.
            # Boundary S2 = S2_max(i = 1 ... nx - 2, j = ny - 1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                v[i * self.ny + j, :] = 0.
            
        # Put option
        else:
            i = 0
            for j in range(self.ny):
                v[i * self.ny + j, :] = self.strike * np.exp(- self.process.r * tau)
            # Boundary S2 = 0 (i = 1 ... nx, j = 0)
            j = 0
            for i in range(1, self.nx):
                v[i * self.ny + j, :] = self.strike * np.exp(- self.process.r * tau)

            # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
            i = self.nx - 1
            for j in range(1, self.ny):
                v[i * self.ny + j, :] = self.hx * bs.deltaBlackScholesMerton(
                    self.xGrid[i], self.strike, tau, self.process.r, self.process.sigma[0], 
                    self.process.q[0], self.payoff.CP
                )
            # Boundary S2 = S2_max(i = 1 ... nx - 2, j = ny - 1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                v[i * self.ny + j, :] = self.hy * bs.deltaBlackScholesMerton(
                    self.yGrid[j], self.strike, tau, self.process.r, self.process.sigma[1],
                    self.process.q[1], self.payoff.CP
                )


class FdxVAWorstOf2D(FdWorstOf2D):

    def __init__(
        self, 
        CP: str, 
        strike: float, 
        process: LognormalProcess2D, 
        xva_data: xVAData,
        xGrid_data: Tuple[float, float, float], 
        yGrid_data: Tuple[float, float, float], 
        tauGrid_data: Tuple[float, float, float], 
        method="FixedPointCrankNicolson"
    ) -> None:
        super().__init__(CP, strike, process, xGrid_data, yGrid_data, tauGrid_data, method)
        self.xva_data = xva_data
        # uSol -> XVA sol \\ vSol -> risk-free Sol
        self.uSol = np.zeros((self.m_dim, 1))
    
    def solve(self):
        # Payoff
        for i in range(self.nx):
            self.vSol[i * self.ny: (i + 1) * self.ny, :] = self.payoff(self.xGrid[i], self.yGrid)[:, None]
        # Load matrix A
        A = self._build_matrix()
        # First stage: Euler implicit (2 X htau/2 step)
        method = FixedPointImplicitEuler(
            0.5 * self.htau, self.m_dim, deepcopy(A), self._build_boundary_conditions,
            self._build_right_hand, self._build_nl_right_hand, self.xva_data.source_function
        )
        for k in range(2):
            tau = (k + 1) * 0.5 * self.htau
            method(self.uSol, self.vSol, tau)
        
        # Second stage: Crank Nicolson
        method = FixedPointCrankNicolson(
            self.htau, self.m_dim, A, self._build_boundary_conditions, self._build_right_hand,
            self._build_nl_right_hand, self.xva_data.source_function
        )
        for k in range(2, self.ntau):
            tau = self.tauGrid[k]
            method(self.uSol, self.vSol, tau)

    def get_solution(self, grid=False):
        if grid:
            X, _ = self.get_coordinates(grid=True)
            v =  self.uSol + self.vSol
            v =  v.flatten().reshape(X.shape)
            return v
        return self.uSol + self.vSol
    
    def get_xva(self, grid=False):
        if grid:
            X, _ = self.get_coordinates(grid=True)
            v =  self.uSol.flatten().reshape(X.shape)
            return v
        return self.uSol
    
    def get_risk_free(self, grid=False):
        if grid:
            X, _ = self.get_coordinates(grid=True)
            v =  self.vSol.flatten().reshape(X.shape)
            return v
        return self.vSol
    
    def _build_matrix(self):
        return super()._build_matrix()
    
    def _build_boundary_conditions(self, A: sparse.lil_matrix):
        super()._build_boundary_conditions(A)
    
    def _build_right_hand(self, v: np.ndarray, tau: float):
        super()._build_right_hand(v, tau)
    
    def _build_nl_right_hand(self, v: np.ndarray, tau: float) -> None:
        # Rewrite array with the proper boundary conditions
        # Call option
        if self.payoff.CP == "c":
            # Boundary S1 = 0 (i = 0, j= 0 ... ny - 1)
            i = 0
            for j in range(self.ny):
                v[i * self.ny + j] = 0.
            # Boundary S2 = 0 (i = 1 ... nx, j = 0)
            j = 0
            for i in range(1, self.nx):
                v[i * self.ny + j] = 0.
            # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
            i = self.nx - 1
            for j in range(1, self.ny):
                v[i * self.ny + j] = 0.
            # Boundary S2 = S2_max(i = 1 ... nx - 2, j = ny - 1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                v[i * self.ny + j] = 0.
            
        # Put option
        else:
            adjustment = self.xva_data.xva_correction(tau)
            i = 0
            for j in range(self.ny):
                aux_v = self.strike * np.exp(- self.process.r * tau)
                v[i * self.ny + j] = (-1. + adjustment) * aux_v
        
            # Boundary S2 = 0 (i = 1 ... nx, j = 0)
            j = 0
            for i in range(1, self.nx):
                aux_v = self.strike * np.exp(- self.process.r * tau)
                v[i * self.ny + j] = (-1. + adjustment) * aux_v

            # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
            i = self.nx - 1
            for j in range(1, self.ny):
                slope = (-1. + adjustment) * bs.deltaBlackScholesMerton(
                    self.xGrid[i], self.strike, tau, self.process.r, self.process.sigma[0], 
                    self.process.q[0], self.payoff.CP
                ) 
                v[i * self.ny + j] = self.hx * slope
            # Boundary S2 = S2_max(i = 1 ... nx - 2, j = ny - 1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                slope = (-1. + adjustment) * bs.deltaBlackScholesMerton(
                    self.yGrid[j], self.strike, tau, self.process.r, self.process.sigma[1],
                    self.process.q[1], self.payoff.CP
                )
                v[i * self.ny + j] = self.hy * slope
    
