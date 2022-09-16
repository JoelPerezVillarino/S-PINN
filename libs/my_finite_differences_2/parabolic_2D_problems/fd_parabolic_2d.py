from pathlib import Path
from typing import Tuple, Union
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 


class Parabolic2D(object):

    def __init__(
        self, 
        xGrid_data: Tuple[float, float, float], 
        yGrid_data: Tuple[float, float, float], 
        tauGrid_data: Tuple[float, float, float], 
    ) -> None:
        
        self.process = None
        self.strike = None
        self.payoff = None
        self.method = None

        # x grid
        self.hx = xGrid_data[2]
        self.nx = int((xGrid_data[1] - xGrid_data[0]) / self.hx)
        self.xGrid = np.linspace(xGrid_data[0], xGrid_data[1], self.nx)
        # y grid
        self.hy = yGrid_data[2]
        self.ny = int((yGrid_data[1] - yGrid_data[0]) / self.hy)
        self.yGrid = np.linspace(yGrid_data[0], yGrid_data[1], self.ny)
        # tau Grid
        self.htau = tauGrid_data[2]
        self.ntau = int((tauGrid_data[1] - tauGrid_data[0]) / self.htau)
        self.tauGrid = np.linspace(tauGrid_data[0], tauGrid_data[1], self.ntau)

        print("Grid info:")
        print(f"    dx, dy, dtau: {self.hx:.2f}, {self.hy:.2f}, {self.htau:.2f};")
        print(f"    nx, ny, ntau: {self.nx:.2f}, {self.ny:.2f}, {self.ntau:.2f}.")

        self.m_dim = self.nx * self.ny

        # Allocate solution
        self.uSol = np.zeros((self.m_dim, self.ntau))
    
    def solve(self, ):
        raise NotImplementedError("Implemented in subclasses")
    
    def get_solution(self, index=None):
        if index is None:
            return self.uSol
        elif isinstance(index, int):
            return self.uSol[:, index]
    
    def save_solution(self, path: Path, index=-1, mod=None):
        if mod is None:
            if index is None:
                X, Y, T = np.meshgrid(self.xGrid, self.yGrid, self.tauGrid, indexing="ij")
                x = np.vstack((np.ravel(X), np.ravel(Y), np.ravel(T))).T
                u = self.get_solution(index).flatten()[:, None]
                shape = X.shape
            else:
                X, Y, T = np.meshgrid(self.xGrid, self.yGrid, self.tauGrid[index], indexing="ij")
                x = np.vstack((np.ravel(X), np.ravel(Y), np.ravel(T))).T
                u = self.get_solution(index).flatten()[:, None]
                shape = X.shape

            np.savez(str(path), points=x, u=u, shape=shape)
            print(f"Successfully saved!!")

        elif isinstance(mod, int) and self.nx % mod == 0 and self.ny % mod == 0:
            if index is None:
                raise NotImplementedError(
                    "Not implemented yet the case where index=None and modular stuff is required!"
                )

            new_nx, new_ny = self.nx // mod, self.ny // mod
            X, Y = np.meshgrid(self.xGrid, self.yGrid, indexing="ij")
            u = self.get_solution(index).reshape(X.shape)
            new_u = np.zeros((new_nx * new_ny, 1))
            points = np.zeros((new_nx * new_ny, 3))
            for i in range(new_nx):
                for j in range(new_ny):
                    new_u[i * new_ny + j] = u[i * mod, j * mod]
                    points[i * new_ny + j, :] = \
                        np.array([self.xGrid[i * mod], self.yGrid[j * mod], self.tauGrid[index]])
            shape = (new_nx, new_ny)
            np.savez(str(path), points=points, u=new_u, shape=shape)
            print(f"Successfully saved!!")
        
        elif isinstance(mod, tuple) and len(mod) == 2 and self.nx % mod[0] == 0 and self.ny % mod[1] == 0:
            new_nx, new_ny = self.nx // mod[0], self.ny // mod[1]
            X, Y = np.meshgrid(self.xGrid, self.yGrid, indexing="ij")
            u = self.get_solution(index).reshape(X.shape)
            new_u = np.zeros((new_nx * new_ny, 1))
            points = np.zeros((new_nx * new_ny, 3))
            for i in range(new_nx):
                for j in range(new_ny):
                    new_u[i * new_ny + j, 0] = u[i * mod[0], j * mod[1]]
                    points[i * new_ny + j, :] = \
                        np.array([self.xGrid[i * mod[0]], self.yGrid[j * mod[1]], self.tauGrid[index]])
            shape = (new_nx, new_ny)
            np.savez(str(path), points=points, u=new_u, shape=shape)
            print(f"Successfully saved!!")
        else:
            raise ValueError("mod must be None or divisor of nx and ny or tuple!!")
    
    def get_gradient(self, var=None, index=-1):
        X, Y = np.meshgrid(self.xGrid, self.yGrid, indexing="ij")
        u = self.get_solution(index=index).reshape(X.shape)
        grad_u = np.gradient(u, self.xGrid, self.yGrid, )
        if var == "x":
             return grad_u[0]
        elif var == "y":
            return grad_u[1]
        else:
            return grad_u
    
    def get_hessian(self, var=None, index=-1):
        grad_u = self.get_gradient(index=index)
        if var == "x":
            grad_u_x = grad_u[0]
            hess_u = np.gradient(grad_u_x, self.hx, axis=0)
        elif var == "y":
            grad_u_y = grad_u[1]
            hess_u = np.gradient(grad_u_y, self.hy, axis=1)
        elif var == "xy":
            grad_u_x = grad_u[0]
            hess_u = np.gradient(grad_u_x, self.hy, axis=1)
        else:
            hess_u = np.gradient(grad_u[0], self.xGrid, self.yGrid,)
            hess_u.append(np.gradient(grad_u[1], self.hy, axis=1))
        return hess_u
    
    def save_gradient(self, path: Path, index=-1, mod=None):
        if mod is None:
            X, Y, T = np.meshgrid(self.xGrid, self.yGrid, self.tauGrid[index], indexing="ij")
            x = np.vstack((np.ravel(X), np.ravel(Y), np.ravel(T))).T
            grad_u = self.get_gradient(index=index)
            u_x, u_y = grad_u[0].flatten()[:, None], grad_u[1].flatten()[:, None]
            np.savez(str(path), points=x, u_x=u_x, u_y=u_y)
            print("Gradient successfully saved!")

        elif isinstance(mod, int) and self.nx % mod == 0 and self.ny % mod == 0:
            if index is None:
                raise NotImplementedError(
                    "Not implemented yet the case where index=None and modular stuff is required!"
                )
            new_nx, new_ny = self.nx // mod, self.ny // mod
            grad_u = self.get_gradient(index=index)
            u_x, u_y = grad_u[0], grad_u[1]
            new_u_x = np.zeros((new_nx * new_ny, 1))
            new_u_y = np.zeros((new_nx * new_ny, 1))
            points = np.zeros((new_nx * new_ny, 3))
            for i in range(new_nx):
                for j in range(new_ny):
                    new_u_x[i * new_ny + j, 0] = u_x[i * mod, j * mod]
                    new_u_y[i * new_ny + j, 0] = u_y[i * mod, j * mod]
                    points[i * new_ny + j, :] = \
                        np.array([self.xGrid[i * mod], self.yGrid[j * mod], self.tauGrid[index]])
            shape = (new_nx, new_ny)
            np.savez(str(path), points=points, u_x=new_u_x, u_y=new_u_y, shape=shape)
            print("Gradient successfully saved!")
        
        elif isinstance(mod, tuple) and len(mod) == 2 and self.nx % mod[0] == 0 and self.ny % mod[1] == 0:
            if index is None:
                raise NotImplementedError(
                    "Not implemented yet the case where index=None and modular stuff is required!"
                )
            new_nx, new_ny = self.nx // mod[0], self.ny // mod[1]
            grad_u = self.get_gradient(index=index)
            u_x, u_y = grad_u[0], grad_u[1]
            new_u_x = np.zeros((new_nx * new_ny, 1))
            new_u_y = np.zeros((new_nx * new_ny, 1))
            points = np.zeros((new_nx * new_ny, 3))
            for i in range(new_nx):
                for j in range(new_ny):
                    new_u_x[i * new_ny + j, 0] = u_x[i * mod[0], j * mod[1]]
                    new_u_y[i * new_ny + j, 0] = u_y[i * mod[0], j * mod[1]]
                    points[i * new_ny + j, :] = \
                        np.array([self.xGrid[i * mod[0]], self.yGrid[j * mod[1]], self.tauGrid[index]])
            shape = (new_nx, new_ny)
            np.savez(str(path), points=points, u_x=new_u_x, u_y=new_u_y, shape=shape)
            print("Gradient successfully saved!")
        
        else:
            raise ValueError("mod must be None or divisor of nx and ny or tuple!!")
    
    def save_hessian(self, path: Path, index=-1, mod=None):
        if mod is None:
            X, Y, T = np.meshgrid(self.xGrid, self.yGrid, self.tauGrid[index], indexing="ij")
            x = np.vstack((np.ravel(X), np.ravel(Y), np.ravel(T))).T
            hess_u = self.get_hessian(index=index)
            u_xx = hess_u[0].flatten()[:, None]
            u_xy = hess_u[1].flatten()[:, None]
            u_yy = hess_u[2].flatten()[:, None]
            np.savez(str(path), points=x, u_xx=u_xx, u_xy=u_xy, u_yy=u_yy)
            print("Hessian successfully saved!")
        
        elif isinstance(mod, int) and self.nx % mod == 0 and self.ny % mod == 0:
            if index is None:
                raise NotImplementedError(
                    "Not implemented yet the case where index=None and modular stuff is required!"
                )
            new_nx, new_ny = self.nx // mod, self.ny // mod
            hess_u = self.get_hessian(index=index)
            u_xx, u_xy, u_yy = hess_u[0], hess_u[1], hess_u[2]
            points = np.zeros((new_nx * new_ny, 3))
            new_u_xx = np.zeros((new_nx * new_ny, 1))
            new_u_yy = np.zeros((new_nx * new_ny, 1))
            new_u_xy = np.zeros((new_nx * new_ny, 1))
            for i in range(new_nx):
                for j in range(new_ny):
                    new_u_xx[i * new_ny + j, 0] = u_xx[i * mod, j * mod]
                    new_u_xy[i * new_ny + j, 0] = u_xy[i * mod, j * mod]
                    new_u_yy[i * new_ny + j, 0] = u_yy[i * mod, j * mod]
                    points[i * new_ny + j, :] = \
                        np.array([self.xGrid[i * mod], self.yGrid[j * mod], self.tauGrid[index]])
            shape = (new_nx, new_ny)
            np.savez(str(path), points=points, u_xx=new_u_xx, u_xy=new_u_xy, u_yy=new_u_yy, shape=shape)
            print("Hessian successfully saved!")

        elif isinstance(mod, tuple) and len(mod) == 2 and self.nx % mod[0] == 0 and self.ny % mod[1] == 0:
            if index is None:
                raise NotImplementedError(
                    "Not implemented yet the case where index=None and modular stuff is required!"
                )
            new_nx, new_ny = self.nx // mod[0], self.ny // mod[1]
            hess_u = self.get_hessian(index=index)
            u_xx, u_xy, u_yy = hess_u[0], hess_u[1], hess_u[2]
            points = np.zeros((new_nx * new_ny, 3))
            new_u_xx = np.zeros((new_nx * new_ny, 1))
            new_u_yy = np.zeros((new_nx * new_ny, 1))
            new_u_xy = np.zeros((new_nx * new_ny, 1))
            for i in range(new_nx):
                for j in range(new_ny):
                    new_u_xx[i * new_ny + j, 0] = u_xx[i * mod[0], j * mod[1]]
                    new_u_xy[i * new_ny + j, 0] = u_xy[i * mod[0], j * mod[1]]
                    new_u_yy[i * new_ny + j, 0] = u_yy[i * mod[0], j * mod[1]]
                    points[i * new_ny + j, :] = \
                        np.array([self.xGrid[i * mod[0]], self.yGrid[j * mod[1]], self.tauGrid[index]])
            shape = (new_nx, new_ny)
            np.savez(str(path), points=points, u_xx=new_u_xx, u_xy=new_u_xy, u_yy=new_u_yy, shape=shape)
            print("Gradient successfully saved!")
        
        else:
            raise ValueError("mod must be None or divisor of nx and ny or tuple!!")
        
    
    def plot_gradient(self, var=None, index=-1, xlabel=None, ylabel=None, title=None):
        X, Y = np.meshgrid(self.xGrid, self.yGrid, indexing="ij")
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        if var == "x" or var == "y":
            w = self.get_gradient(var, index=index)
            ax = fig.add_subplot(111, projection="3d")    
            ax.plot_surface(X, Y, w, cmap="coolwarm")
            if xlabel is not None and isinstance(xlabel, str):
                ax.set_xlabel(xlabel)
            if ylabel is not None and isinstance(ylabel, str):
                ax.set_ylabel(ylabel)
            if title is not None and isinstance(title, str):
                ax.set_title(title)
        else:
            W = self.get_gradient(var, index=index)
            wx, wy = W[0], W[1]
            w = np.sqrt(np.power(wx, 2) + np.power(wy, 2))
            ax = fig.add_subplot(111)
            pcm = ax.pcolor(X, Y, w, cmap="coolwarm")
            fig.colorbar(pcm, ax=ax, extend="max")
            if xlabel is not None and isinstance(xlabel, str):
                ax.set_xlabel(xlabel)
            if ylabel is not None and isinstance(ylabel, str):
                ax.set_ylabel(ylabel)
            if title is not None and isinstance(title, str):
                ax.set_title(title)
            ax.legend()
    
    def plot_hessian(self, var=None, index=-1, xlabel=None, ylabel=None, title=None):
        X, Y = np.meshgrid(self.xGrid, self.yGrid, indexing="ij")
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        if var == "x" or var == "y" or var == "xy":
            w = self.get_hessian(var, index=index)
            ax = fig.add_subplot(111, projection="3d")    
            ax.plot_surface(X, Y, w, cmap="coolwarm")
            if xlabel is not None and isinstance(xlabel, str):
                ax.set_xlabel(xlabel)
            if ylabel is not None and isinstance(ylabel, str):
                ax.set_ylabel(ylabel)
            if title is not None and isinstance(title, str):
                ax.set_title(title)

    def plot_solution(self, index=-1, xlabel=None, ylabel=None, title=None):
        X, Y = np.meshgrid(self.xGrid, self.yGrid, indexing="ij")
        W = np.reshape(self.get_solution(index), X.shape)

        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, W, cmap="coolwarm")

        if xlabel is not None and isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        if ylabel is not None and isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        if title is not None and isinstance(title, str):
            ax.set_title(title)

    def _build_matrix(self, ):
        raise NotImplementedError("Implemented in subclasses")
    
    def _build_boundary_conditions(self, A: sparse.lil_matrix, ):
        raise NotImplementedError("Implemented in subclasses")

    def _build_right_hand(
        self, 
        v: np.ndarray, 
        tau: float
    ):
        raise NotImplementedError("Implemented in subclasses")
