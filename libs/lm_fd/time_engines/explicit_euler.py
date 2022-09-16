import numpy as np
from scipy import sparse

from .theta_method import ThetaMethod, FixedPointThetaMethod


class ExplicitEuler(ThetaMethod):

    def __init__(
        self,
        htau: float,
        dim: int,
        A: sparse.lil_matrix,
        build_boundary_conditions: callable,
        build_right_hand: callable, 
    ) -> None:
        super().__init__(0., htau, dim, A, build_boundary_conditions, build_right_hand)
        
    def __call__(self, uSol: np.ndarray, tau: float) -> np.ndarray:
        return super().__call__(uSol, tau)


class FixedPointExplicitEuler(FixedPointThetaMethod):

    def __init__(
        self,
        htau: float,
        dim: int,
        A: sparse.lil_matrix, 
        build_boundary_conditions: callable, 
        build_right_hand: callable, 
        build_nl_right_hand: callable, 
        source: callable
    ) -> None:
        super().__init__(
            0., htau, dim, A, build_boundary_conditions, build_right_hand, build_nl_right_hand, source
        )

    def __call__(self, uSol: np.ndarray, vSol: np.ndarray, tau: float) -> np.ndarray:
        return super().__call__(uSol, vSol, tau)
    
    def set_maxiter(self, new_maxiter):
        super().set_maxiter(new_maxiter)
    
    def set_tol(self, new_tol):
        super().set_tol(new_tol)
    
    def _stopping_criteria(self, u_k, u_km1):
        return super()._stopping_criteria(u_k, u_km1)