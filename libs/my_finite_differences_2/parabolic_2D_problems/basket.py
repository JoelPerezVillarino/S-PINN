from typing import Tuple, Union
import numpy as np
from scipy import sparse

from libs.stochastic_processes import LognormalProcess2D
from .black_scholes2 import FdBlackScholes2D
from ..utils import ArithmeticBasketPayoff, WorstOfPayoff
from ..utils import xVAData
from ..time_engines import CrankNicolson, FixedPointCrankNicolson
import libs.black_scholes as bs


class FdArithmeticBasket2D(FdBlackScholes2D):

    def __init__(
        self,
        CP: str, 
        strike: float, 
        process: LognormalProcess2D, 
        xGrid_data: Tuple[float, float, float], 
        yGrid_data: Tuple[float, float, float], 
        tauGrid_data: Tuple[float, float, float],
        method=CrankNicolson()
    ) -> None:

        super().__init__(process, xGrid_data, yGrid_data, tauGrid_data)
        self.process = process
        self.strike = strike
        self.payoff = ArithmeticBasketPayoff(CP, strike)
        self.method = method
    
    def solve(self):
        # Payoff
        for i in range(self.nx):
            #for j in range(self.ny):
            #    self.uSol[i * self.ny + j, 0] = self.payoff(self.xGrid[i], self.yGrid[j])
            self.uSol[i * self.ny: (i + 1) * self.ny, 0] = self.payoff(self.xGrid[i], self.yGrid)
        self.method(self)

    def _build_matrix(self):
        return super()._build_matrix()
    
    def _build_boundary_conditions(self, A: sparse.lil_matrix):
        # i = 0, j= 0
        i, j = 0, 0
        A[i * self.ny + j, i * self.ny + j] = 1.
        # Boundary S1 = 0 (i = 0, j= 1 ... ny - 1)
        i = 0
        for j in range(self.ny):
            A[i * self.ny + j, i * self.ny + j] = 1.
        # Boundary S2 = 0 (i = 1 ... nx-1, j = 0)
        j = 0
        for i in range(1, self.nx):
            A[i * self.ny + j, i * self.ny + j] = 1.
        # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
        i = self.nx - 1
        for j in range(1, self.ny):
            A[i * self.ny + j, (i - 2) * self.ny + j] = 1
            A[i * self.ny + j, (i - 1) * self.ny + j] = -2.
            A[i * self.ny + j, i * self.ny + j] = 1.
        # Boundary S2 = S2_max (i = 1 ... nx - 2, j = ny - 1)
        j = self.ny - 1
        for i in range(1, self.nx - 1):
            A[i * self.ny + j, i * self.ny + j - 2] = 1.
            A[i * self.ny + j, i * self.ny + j - 1] = -2.
            A[i * self.ny + j, i * self.ny + j] = 1.

    def _build_right_hand(
        self, 
        v: np.ndarray, 
        tau: float
    ):
        # Rewrite array with the proper boundary conditions
        # i = 0, j = 0
        i, j = 0, 0
        if self.payoff.CP == "c":
            v[i * self.ny + j, 0] = 0.
        else:
            v[i * self.ny + j, 0] = self.strike * np.exp(- self.process.r * tau)
        # Boundary S1 = 0 (i = 0, j= 0 ... ny - 1)
        i = 0
        for j in range(1, self.ny):
            v[i * self.ny + j, 0] = bs.blackScholesMerton(
                self.payoff._w2 * self.yGrid[j], self.strike, tau, self.process.r,
                self.process.sigma[1], self.process.q[1], self.payoff.CP
            )
        # Boundary S2 = 0 (i = 1 ... nx-1, j = 0)
        j = 0
        for i in range(1, self.nx):
            v[i * self.ny + j, 0] = bs.blackScholesMerton(
                self.payoff._w1 * self.xGrid[i], self.strike, tau, self.process.r,
                self.process.sigma[0], self.process.q[0], self.payoff.CP
            )
        # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
        i = self.nx - 1
        for j in range(1, self.ny):
            v[i * self.ny + j, 0] = 0.
        # Boundary S2 = S2_max (i = 1 ... nx - 2, j = ny - 1)
        j = self.ny - 1
        for i in range(1, self.nx - 1):
            v[i * self.ny + j, 0] = 0.


class FdWorstOf2D(FdBlackScholes2D):

    def __init__(
        self,
        CP: str,
        strike: float, 
        process: LognormalProcess2D, 
        xGrid_data: Tuple[float, float, float], 
        yGrid_data: Tuple[float, float, float], 
        tauGrid_data: Tuple[float, float, float], 
        method=CrankNicolson()
    ) -> None:
        super().__init__(process, xGrid_data, yGrid_data, tauGrid_data)
        self.process = process
        self.strike = strike
        self.payoff = WorstOfPayoff(CP, strike)
        self.method = method

    def solve(self):
        # Payoff
        for i in range(self.nx):
            self.uSol[i * self.ny: (i + 1) * self.ny, 0] = self.payoff(self.xGrid[i], self.yGrid)
        self.method(self)
    
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
            i = 0
            for j in range(self.ny):
                v[i * self.ny + j] = self.strike * np.exp(- self.process.r * tau)
            # Boundary S2 = 0 (i = 1 ... nx, j = 0)
            j = 0
            for i in range(1, self.nx):
                v[i * self.ny + j] = self.strike * np.exp(- self.process.r * tau)

            # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
            i = self.nx - 1
            for j in range(1, self.ny):
                v[i * self.ny + j] = self.hx * bs.deltaBlackScholesMerton(
                    self.xGrid[i], self.strike, tau, self.process.r, self.process.sigma[0], 
                    self.process.q[0], self.payoff.CP
                )
            # Boundary S2 = S2_max(i = 1 ... nx - 2, j = ny - 1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                v[i * self.ny + j] = self.hy * bs.deltaBlackScholesMerton(
                    self.yGrid[j], self.strike, tau, self.process.r, self.process.sigma[1],
                    self.process.q[1], self.payoff.CP
                )


class FdxVAArithmeticBasket2D(FdArithmeticBasket2D):

    def __init__(
        self, 
        CP: str, 
        strike: float, 
        process: LognormalProcess2D,
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
        return super()._build_boundary_conditions(A)
    
    def _build_right_hand(self, v: np.ndarray, tau: float):
        # Rewrite array with the proper boundary conditions
        # i = 0, j = 0
        i, j = 0, 0
        if self.payoff.CP == "c":
            v[i * self.ny + j, 0] = 0.0
        else:
            adjustment = self.xva_data.xva_correction(tau)
            v[i * self.ny + j, 0] = \
                -(1. - adjustment) * self.strike * np.exp(- self.process.r * tau)
        # Boundary S1 = 0 (i = 0, j= 0 ... ny - 1)
        i = 0
        for j in range(1, self.ny):
            v[i * self.ny + j, 0] = bs.xVABlackScholesMerton(
                self.payoff._w2 * self.yGrid[j], self.strike, tau, self.process.r,
                self.process.sigma[1], self.process.q[1], self.xva_data.lamB,
                self.xva_data.lamC, self.xva_data.rB, self.xva_data.rC, self.payoff.CP
            )
        # Boundary S2 = 0 (i = 1 ... nx-1, j = 0)
        j = 0
        for i in range(1, self.nx):
            v[i * self.ny + j, 0] = bs.xVABlackScholesMerton(
                self.payoff._w1 * self.xGrid[i], self.strike, tau, self.process.r,
                self.process.sigma[0], self.process.q[0], self.xva_data.lamB,
                self.xva_data.lamC, self.xva_data.rB, self.xva_data.rC, self.payoff.CP
            )
        # Boundary S1 = S1_max (i = nx - 1, j = 1 ... ny - 1)
        i = self.nx - 1
        for j in range(1, self.ny):
            v[i * self.ny + j, 0] = 0.
        # Boundary S2 = S2_max (i = 1 ... nx - 2, j = ny - 1)
        j = self.ny - 1
        for i in range(1, self.nx - 1):
            v[i * self.ny + j, 0] = 0.


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
        method=FixedPointCrankNicolson()
    ) -> None:
        super().__init__(CP, strike, process, xGrid_data, yGrid_data, tauGrid_data, method)
        self.xva_data = xva_data
        self.vSol = None
    
    def solve(self, vSol):
        self.vSol = vSol
        self.method(self, vSol)
    
    def get_solution(self, index=None):
        return self.uSol[:, index] + self.vSol[:, index]
    
    def get_xva(self, index=None):
        return self.uSol[:, index]
    
    def _build_matrix(self):
        return super()._build_matrix()
    
    def _build_boundary_conditions(self, A: sparse.lil_matrix) -> None:
        return super()._build_boundary_conditions(A)
    
    def _build_right_hand(self, v: np.ndarray, tau: float) -> None:
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

        
    


    