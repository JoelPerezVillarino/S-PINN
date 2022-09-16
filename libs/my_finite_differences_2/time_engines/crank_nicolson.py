from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized


class CrankNicolson(object):

    def __init__(
        self, 
        
    ) -> None:
        self.__theta = 0.5
        
    def __call__(
        self,
        model, ) -> np.ndarray:
        identity = sparse.identity((model.m_dim), format="lil")
        A = model._build_matrix()
        # First stage: fully implicty for capture the strike behaviour
        # Two semi steps
        Aimplicit = identity - A.multiply(model.htau / 2.)
        model._build_boundary_conditions(Aimplicit)
        Aimplicit = Aimplicit.tocsc()
        solveLU = factorized(Aimplicit)
        # First semi step
        b_term = identity.dot(model.uSol[:, 0:1])
        model._build_right_hand(b_term, model.htau / 2.)
        u_h2 = solveLU(b_term)
        # Second semi step
        b_term = identity.dot(u_h2[:, 0:1])
        model._build_right_hand(b_term, model.htau)
        model.uSol[:, 1:2] = solveLU(b_term)
        
        # Second stage: CrankNicolson
        Aexplicit = identity + A.multiply(self.__theta * model.htau)
        Aimplicit = identity - A.multiply(self.__theta * model.htau)
        model._build_boundary_conditions(Aimplicit)
        Aexplicit = Aexplicit.tocsc()
        Aimplicit = Aimplicit.tocsc()
        solveLU = factorized(Aimplicit)

        for k in range(2, model.ntau):
            tau = model.tauGrid[k]
            b_term = Aexplicit.dot(model.uSol[:, k-1: k])
            model._build_right_hand(b_term, tau)
            model.uSol[:, k: k+1] = solveLU(b_term) 
 
        return


class FixedPointCrankNicolson(object):

    def __init__(self) -> None:
        self.__theta = 0.5
        self.__maxiter = 200
        self.__tol = 1.e-10

    def __call__(self, model, vSol: np.ndarray) -> np.ndarray:
        if model.uSol.shape != vSol.shape:
            raise ValueError("vSol and uSol must have the same dimension!")
        identity = sparse.identity((model.m_dim), format="lil")
        A = model._build_matrix()
        # Defining implicit and explicit matrix
        Aexplicit = identity + A.multiply(self.__theta * model.htau)
        Aexplicit = Aexplicit.tocsc()

        Aimplicit = identity - A.multiply(self.__theta * model.htau)
        # Build boundary conditions
        model._build_boundary_conditions(Aimplicit)
        Aimplicit = Aimplicit.tocsc()
        # Lu factorization
        solveLU = factorized(Aimplicit)

        # Loop
        for k in range(1, model.ntau):
            tau = model.tauGrid[k]
            u0 = deepcopy(model.uSol[:, k-1: k])
            f_km1 = model.xva_data.source_function(u0, vSol[:, k-1: k])
            g_k = Aexplicit.dot(u0) + self.__theta * model.htau * f_km1
            # Fixed point algorithm
            for _ in range(self.__maxiter):
                f_k = model.xva_data.source_function(u0, vSol[:, k: k+1])
                b = g_k + self.__theta * model.htau * f_k
                model._build_right_hand(b, tau)
                model.uSol[:, k: k+1] = solveLU(b)
                # Stopping criteria
                if self.__stopping_criteria(model.uSol[:, k: k+1], u0):
                    break
                # Updating u0 for the next iteration if criteria > tol
                u0 = deepcopy(model.uSol[:, k: k+1])
            else:
                print(f"Convergence of the FixedPoint algorithm fails at the step {k},")
                print(f"for {self.__maxiter} iterations and {self.__tol:.2e} tol.")


    def set_maxiter(self, new_maxiter):
        self.__maxiter = new_maxiter
    
    def set_tol(self, new_tol):
        self.__tol = new_tol
    
    def __stopping_criteria(self, u_k, u_km1):
        term = np.abs(u_k - u_km1) / np.maximum(1., u_k)
        term = np.linalg.norm(term, ord=np.inf)
        return term <= self.__tol
