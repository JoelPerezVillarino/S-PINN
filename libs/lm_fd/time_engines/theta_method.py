from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized


class ThetaMethod(object):

    def __init__(
        self,
        theta: float,
        htau: float,
        dim: int, 
        A: sparse.lil_matrix, 
        build_boundary_conditions: callable,
        build_right_hand: callable,
    ) -> None:
        self._theta = theta
        self._htau = htau
        self._dim = dim
        # Save right hand method 
        self._build_right_hand = build_right_hand

        # Processing A matrix (implicit and explicit forms)
        I = sparse.identity(self._dim, format="lil")
        implicitMatrix = I - A.multiply(self._theta * self._htau)
        build_boundary_conditions(implicitMatrix)
        implicitMatrix = implicitMatrix.tocsc()

        self._LU_factor = factorized(implicitMatrix) 
        explicitMatrix = I + A.multiply((1. - self._theta) * self._htau)
        self._explicitMatrix = explicitMatrix.tocsc()
        
    def __call__(
        self,
        vSol: np.ndarray,
        tau: float 
    ) -> np.ndarray:
        """ One step theta method """
        b = self._explicitMatrix.dot(vSol[:, 0: 1])
        self._build_right_hand(b, tau)
        vSol[:, 0: 1] = self._LU_factor(b) 
        

class FixedPointThetaMethod(ThetaMethod):

    def __init__(
        self,
        theta: float, 
        htau: float,
        dim: int,
        A: sparse.lil_matrix,
        build_boundary_conditions: callable,
        build_right_hand: callable,
        build_nl_right_hand: callable,
        source: callable,
    ) -> None:
        super().__init__(theta, htau, dim, A, build_boundary_conditions, build_right_hand)
        # Fixed point params
        self._maxiter = 200
        self._tol = 1.e-10
        # Allocate NL right hand function
        self._build_nl_right_hand = build_nl_right_hand
        # Allocate source function
        self._source = source

    def __call__(self, uSol: np.ndarray, vSol: np.ndarray, tau: float) -> np.ndarray:
        """ One step thetaMethod with fixed point algorithm """
        # Computing function value depending on vSol_km1
        u0 = deepcopy(uSol)
        f_km1 = self._source(u0, vSol)
        # Theta method: vSol_km1 -> vSol_k
        super().__call__(vSol, tau)
        # Fixed Point algorithm
        g_k = self._explicitMatrix.dot(u0) + (1. - self._theta) * self._htau * f_km1
        # Loop
        for _ in range(self._maxiter):
            f_k = self._source(u0, vSol)
            b = g_k + self._theta * self._htau * f_k
            self._build_nl_right_hand(b, tau)
            uSol[:, 0: 1] = self._LU_factor(b)
            # Stopping criteria
            if self._stopping_criteria(uSol[:, 0:1], u0):
                break
            # Updating u0 for the next iteration if criteria > tol
            u0 = deepcopy(uSol[:, 0:1])
        else:
            print(f"Convergence of the FixedPoint algorithm fails at time {tau:.2e},")
            print(f"for {self._maxiter} iterations and {self._tol:.2e} tol.")
            print(f"tol achieved: {self.test_tol}")    


    def set_maxiter(self, new_maxiter):
        self._maxiter = new_maxiter
    
    def set_tol(self, new_tol):
        self._tol = new_tol
    
    def _stopping_criteria(self, u_k, u_km1):
        term = np.abs(u_k - u_km1) / np.maximum(1., u_k)
        self.test_tol = np.linalg.norm(term, ord=np.inf)
        return self.test_tol <= self._tol

    def _tol_criteria(self, u_k, u_km1, tol_vector, it):
        term = np.abs(u_k - u_km1) / np.maximum(1., u_k)
        boolean = np.less_equal(term, self._tol)
        for i, val in enumerate(boolean):
            if val and tol_vector[i] == 0:
                tol_vector[i] = it
        

    


