from copy import deepcopy
from typing import Tuple
import numpy as np
from scipy import sparse

import libs.black_scholes as bs
from libs.stochastic_processes.processes2D import HestonProcess
from ..utils.payoffs import VanillaPayoff
from ..utils import xVAData
from .parabolic_2D import Parabolic2D
from ..time_engines import ImplicitEuler, CrankNicolson
from ..time_engines import FixedPointImplicitEuler, FixedPointCrankNicolson


class FdVanillaHeston(Parabolic2D):

    def __init__(
        self,
        CP: str,
        strike: float, 
        process: HestonProcess,
        xGrid_data: Tuple[float, float, float], 
        yGrid_data: Tuple[float, float, float], 
        tauGrid_data: Tuple[float, float, float], 
        method="CrankNicolson"
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
            self.vSol[i * self.ny: (i + 1) * self.ny, :] = self.payoff(self.xGrid[i])
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

    def _build_matrix(self, ):
        # Allocate sparse matrix, A, and fill it
        A = sparse.lil_matrix((self.nx * self.ny, self.nx * self.ny), )

        #indy = self.ny // 2
        #condition = np.isclose(self.yGrid, 1., atol=self.hy * 0.5) 
        #indy, = np.where(condition)[0]
        
        # i = 1 ... nx-2, j = 1 ... ny - 2
        for i in range(1, self.nx - 1):
        #for i in range(1, indy):
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
        
        """
        # Boundary S = 0 (i = 0, j = 1 ... ny-1)
        i = 0
        for j in range(1, self.ny-1):
            ay = self._ay_term(j)
            by = self._by_term(j)

            A[i * self.ny + j, i * self.ny + j - 1] = ay - by
            A[i * self.ny + j, i * self.ny + j + 1] = ay + by
            A[i * self.ny + j, i * self.ny + j] = - (2. * ay + self.process.r)
        """
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
        # Boundary S = 0, V >= 0 (i=0, j=0...ny)
        i = 0
        for j in range(self.ny):
            A[i * self.ny + j, i * self.ny + j] = 1.

        # Boundary S = S_max (i = nx - 1, j = 0 ... ny - 1)
        i = self.nx - 1
        for j in range(self.ny):
            A[i * self.ny + j, i * self.ny + j] = 1.
            #A[i * self.ny + j, (i - 1) * self.ny + j] = -1.
        # Boundary V = V_max (i = 1 .. nx - 2, j = ny - 1)
        j = self.ny - 1
        for i in range(1, self.nx - 1):
            A[i * self.ny + j, i * self.ny + j] = 1.

    
    def _build_right_hand(
        self, 
        v: np.ndarray,
        tau: float, 
    ):
        # Rewrite v array with the boundary conditions
        if self.payoff.CP == "c":
            # Boundary S = 0, V >= 0
            i = 0
            #for j in range(self.ny):
            #v[i * self.ny + j] = 0.
            v[i * self.ny: (i + 1) * self.ny] = 0.
            
            # Boundary S = S_max (i = nx - 1, j = 0 ... ny - 1)
            i = self.nx - 1
            #v[i * self.ny: (i + 1) * self.ny] = self.hx * np.exp(- self.process.q * tau)
            v[i * self.ny: (i + 1) * self.ny] = bs.blackScholesMerton(
                self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[:, None]),
                    self.process.q, self.payoff.CP
            )
            # Boundary V = V_max (i = 1 ... nx - 2, j = ny-1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                #v[i * self.ny + j] = self.xGrid[i] * np.exp(- self.process.q * tau)
                v[i * self.ny + j] = bs.blackScholesMerton(
                    self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[j]),
                    self.process.q, self.payoff.CP
                )
        
        else:
            # Boundary S = 0, V >= 0
            i = 0
            #for j in range(self.ny):
            #    v[i * self.ny + j] = self.payoff.strike * np.exp(- tau * self.process.r)
            v[i * self.ny: (i + 1) * self.ny] = self.payoff.strike * np.exp(- tau * self.process.r)
            
            # Boundary S = S_max (i = nx - 1, j = 0 ... ny - 1)
            i = self.nx - 1
            #v[i * self.ny: (i + 1) * self.ny] = 0.
            v[i * self.ny: (i + 1) * self.ny] = bs.blackScholesMerton(
                self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[:, None]),
                    self.process.q, self.payoff.CP
                )

            # Boundary V = V_max (i = 1 ... nx - 2, j = ny-1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                #v[i * self.ny + j] = self.payoff.strike * np.exp(- tau * self.process.r)
                v[i * self.ny + j] = bs.blackScholesMerton(
                    self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[j]),
                    self.process.q, self.payoff.CP
                )

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
        method="FixedPointCrankNicolson",
    ) -> None:
        super().__init__(CP, strike, process, xGrid_data, yGrid_data, tauGrid_data, method)
        self.xva_data = xva_data
        # uSol -> XVA sol \\ vSol -> risk-free Sol
        self.uSol = np.zeros((self.m_dim, 1))
    
    def solve(self, maxiter=200, tol=1.e-10):
        # Payoff values
        for i in range(self.nx):
            self.vSol[i * self.ny: (i + 1) * self.ny, :] = self.payoff(self.xGrid[i])
        # Load matrix A
        A = self._build_matrix()
        # First stage: Euler implicit (2 X htau/2 step)
        method = FixedPointImplicitEuler(
            0.5 * self.htau, self.m_dim, deepcopy(A), self._build_boundary_conditions,
            self._build_right_hand, self._build_nl_right_hand, self.xva_data.source_function
        )
        method.set_tol(tol)
        method.set_maxiter(maxiter)
        for k in range(2):
            tau = (k + 1) * 0.5 * self.htau
            method(self.uSol, self.vSol, tau)
        
        # Second stage: Crank Nicolson
        method = FixedPointCrankNicolson(
            self.htau, self.m_dim, A, self._build_boundary_conditions, self._build_right_hand,
            self._build_nl_right_hand, self.xva_data.source_function
        )
        method.set_tol(tol)
        method.set_maxiter(maxiter)
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
    
    def _build_nl_right_hand(self, v: np.ndarray, tau: float):
        # Compute xVA correction
        xva_correction = self.xva_data.xva_correction(tau)
        # Rewrite v array with the boundary conditions
        if self.payoff.CP == "c":
            # Boundary S = 0, V >= 0
            i = 0
            v[i * self.ny: (i + 1) * self.ny] = 0.
            
            # Boundary S = S_max (i = nx - 1, j = 0 ... ny - 1)
            i = self.nx - 1
            bs_term = bs.blackScholesMerton(
                self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[:, None]),
                    self.process.q, self.payoff.CP
            )
            v[i * self.ny: (i + 1) * self.ny] = - (1. - xva_correction) * bs_term 
            # Boundary V = V_max (i = 1 ... nx - 2, j = ny-1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                bs_term = bs.blackScholesMerton(
                    self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[j]),
                    self.process.q, self.payoff.CP
                )
                v[i * self.ny + j] = - (1. - xva_correction) * bs_term
        
        else:
            # Boundary S = 0, V >= 0
            i = 0
            term = self.payoff.strike * np.exp(- tau * self.process.r)
            v[i * self.ny: (i + 1) * self.ny] = - (1. - xva_correction) * term
            
            # Boundary S = S_max (i = nx - 1, j = 0 ... ny - 1)
            i = self.nx - 1
            bs_term = bs.blackScholesMerton(
                self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[:, None]),
                    self.process.q, self.payoff.CP
                )
            v[i * self.ny: (i + 1) * self.ny] = - (1. - xva_correction) * bs_term

            # Boundary V = V_max (i = 1 ... nx - 2, j = ny-1)
            j = self.ny - 1
            for i in range(1, self.nx - 1):
                bs_term = bs.blackScholesMerton(
                    self.xGrid[i], self.strike, tau, self.process.r, np.sqrt(self.yGrid[j]),
                    self.process.q, self.payoff.CP
                )
                v[i * self.ny + j] = - (1. - xva_correction) * bs_term