from pathlib import Path
from typing import Tuple
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
        print(f"    dx, dy, dtau: {self.hx:.3f}, {self.hy:.3f}, {self.htau:.3f};")
        print(f"    nx, ny, ntau: {self.nx:.3f}, {self.ny:.3f}, {self.ntau:.3f}.")

        self.m_dim = self.nx * self.ny

        # Allocate solution
        self.vSol = np.zeros((self.m_dim, 1))

    def solve(self, ):
        raise NotImplementedError("Implemented in subclasses")

    def get_coordinates(self, grid=False):
        if grid:
            X, Y = np.meshgrid(self.xGrid, self.yGrid, indexing="ij")
            return X, Y
        X, Y, T = np.meshgrid(self.xGrid, self.yGrid, self.tauGrid[-1], indexing="ij")
        points =  np.vstack((np.ravel(X), np.ravel(Y), np.ravel(T))).T
        return points
    
    def get_solution(self, grid=False):
        if grid:
            X, _ = self.get_coordinates(grid=True)
            u = self.vSol.flatten().reshape(X.shape)
            return u
        return self.vSol
    
    def get_gradient(self, var=None, grid=False):
        u = self.get_solution(grid=True)
        grad_u = np.gradient(u, self.xGrid, self.yGrid, )
        if not grid:
            grad_u_x = grad_u[0].flatten()[:, None]
            grad_u_y = grad_u[1].flatten()[:, None]
            grad_u = (grad_u_x, grad_u_y)
        if var == "x":
            return grad_u[0]
        elif var == "y":     
            return grad_u[1]
        else:
            return grad_u
    
    def get_hessian(self, var=None, grid=False):
        grad_u = self.get_gradient(grid=True)
        hess_u = np.gradient(grad_u[0], self.xGrid, self.yGrid, )
        hess_u.append(np.gradient(grad_u[1], self.hy, axis=1))
        if not grid:
            hess_u_xx = hess_u[0].flatten()[:, None]
            hess_u_xy = hess_u[1].flatten()[:, None]
            hess_u_yy = hess_u[2].flatten()[:, None]
            hess_u = (hess_u_xx, hess_u_xy, hess_u_yy)
        if var == "x":
            return hess_u[0]
        elif var == "y":
            return hess_u[2]
        elif var == "xy":
            return hess_u[1]
        else:
            return hess_u
    
    def save_solution(self, fname_path: Path, mod=None):
        if mod is None:
            x = self.get_coordinates()
            u = self.get_solution().flatten()[:, None]
            shape = (self.nx, self.ny)
            np.savez(str(fname_path), points=x, u=u, shape=shape)
            print(f"Sol. succesfully saved in {str(fname_path)}!")

        elif isinstance(mod, int) and self.nx % mod == 0 and self.ny % mod == 0:
            nx, ny = self.nx // mod, self.ny // mod
            u = self.get_solution(grid=True)
            mod_u = np.zeros((nx * ny, 1))
            points = np.zeros((nx * ny, 3))
            for i in range(nx):
                for j in range(ny):
                    mod_u[i * ny + j, :] = u[i * mod, j * mod]
                    points[i * ny + j, :] = \
                        np.array([self.xGrid[i * mod], self.yGrid[j * mod], self.tauGrid[-1]])
            shape = (nx, ny)
            np.savez(str(fname_path), points=points, u=mod_u, shape=shape)
            print(f"Sol. succesfully saved in {str(fname_path)}!")

        elif isinstance(mod, tuple) and len(mod) == 2 and self.nx % mod[0] == 0 and self.ny % mod[1] == 0:
            nx, ny = self.nx // mod[0], self.ny // mod[1]
            u = self.get_solution(grid=True)
            mod_u = np.zeros((nx * ny, 1))
            points = np.zeros((nx * ny, 3))
            for i in range(nx):
                for j in range(ny):
                    mod_u[i * ny + j, :] = u[i * mod[0], j * mod[1]]
                    points[i * ny + j, :] = \
                        np.array([self.xGrid[i * mod[0]], self.yGrid[j * mod[1]], self.tauGrid[-1]])
            shape = (nx, ny)
            np.savez(str(fname_path), points=points, u=mod_u, shape=shape)
            print(f"Sol. succesfully saved in {str(fname_path)}!")
        else:
            raise ValueError("mod must be None or divisor of nx and ny or tuple!!")
    
    def save_gradient(self, fname_path: Path, mod=None):
        if mod is None:
            x = self.get_coordinates()
            grad_u = self.get_gradient()
            shape = (self.nx, self.ny)
            np.savez(str(fname_path), points=x, u_x=grad_u[0], u_y=grad_u[1], shape=shape)
            print(f"Grad. succesfully saved in {str(fname_path)}!")

        elif isinstance(mod, int) and self.nx % mod == 0 and self.ny % mod == 0:
            nx, ny = self.nx // mod, self.ny // mod
            grad_u = self.get_gradient(grid=True)
            u_x = np.zeros((nx * ny, 1))
            u_y = np.zeros_like(u_x)
            points = np.zeros((nx * ny, 3))
            for i in range(nx):
                for j in range(ny):
                    u_x[i * ny + j, :] = grad_u[0][i * mod, j * mod]
                    u_y[i * ny + j, :] = grad_u[1][i * mod, j * mod]
                    points[i * ny + j, :] = \
                        np.array([self.xGrid[i * mod], self.yGrid[j * mod], self.tauGrid[-1]])
            shape = (nx, ny)
            np.savez(str(fname_path), points=points, u_x=u_x, u_y=u_y, shape=shape)
            print(f"Grad. succesfully saved in {str(fname_path)}!")
        
        elif isinstance(mod, tuple) and len(mod) == 2 and self.nx % mod[0] == 0 and self.ny % mod[1] == 0:
            nx, ny = self.nx // mod[0], self.ny // mod[1]
            grad_u = self.get_gradient(grid=True)
            u_x = np.zeros((nx * ny, 1))
            u_y = np.zeros_like(u_x)
            points = np.zeros((nx * ny, 3))
            for i in range(nx):
                for j in range(ny):
                    u_x[i * ny + j, :] = grad_u[0][i * mod[0], j * mod[1]]
                    u_y[i * ny + j, :] = grad_u[1][i * mod[0], j * mod[1]]
                    points[i * ny + j, :] = \
                        np.array([self.xGrid[i * mod[0]], self.yGrid[j * mod[1]], self.tauGrid[-1]])
            shape = (nx, ny)
            np.savez(str(fname_path), points=points, u_x=u_x, u_y=u_y, shape=shape)
            print(f"Grad. succesfully saved in {str(fname_path)}!")
        else:
            raise ValueError("mod must be None or divisor of nx and ny or tuple!!")
    
    def save_hessian(self, fname_path: Path, mod=None):
        if mod is None:
            x = self.get_coordinates()
            hess_u = self.get_hessian(grid=False)
            shape = (self.nx, self.ny)
            np.savez(str(fname_path), points=x, u_xx=hess_u[0], u_xy=hess_u[1], u_yy=hess_u[2], shape=shape)
            print(f"Hess. succesfully saved in {str(fname_path)}!")
        
        elif isinstance(mod, int) and self.nx % mod == 0 and self.ny % mod == 0:
            nx, ny = self.nx // mod, self.ny // mod
            hess_u = self.get_hessian(grid=True)
            points = np.zeros((nx * ny, 3))
            u_xx = np.zeros((nx * ny, 1))
            u_xy = np.zeros_like(u_xx)
            u_yy = np.zeros_like(u_xx)
            for i in range(nx):
                for j in range(ny):
                    u_xx[i * ny + j, :] = hess_u[0][i * mod, j * mod]
                    u_xy[i * ny + j, :] = hess_u[1][i * mod, j * mod]
                    u_yy[i * ny + j, :] = hess_u[2][i * mod, j * mod]
                    points[i * ny + j, :] = \
                        np.array([self.xGrid[i * mod], self.yGrid[j * mod], self.tauGrid[-1]])
            shape = (nx, ny)
            np.savez(str(fname_path), points=points, u_xx=u_xx, u_yy=u_yy, u_xy=u_xy, shape=shape)
            print(f"Hess. succesfully saved in {str(fname_path)}!")
        
        elif isinstance(mod, tuple) and len(mod) == 2 and self.nx % mod[0] == 0 and self.ny % mod[1] == 0:
            nx, ny = self.nx // mod[0], self.ny // mod[1]
            hess_u = self.get_hessian(grid=True)
            points = np.zeros((nx * ny, 3))
            u_xx = np.zeros((nx * ny, 1))
            u_xy = np.zeros_like(u_xx)
            u_yy = np.zeros_like(u_xx)
            for i in range(nx):
                for j in range(ny):
                    u_xx[i * ny + j, :] = hess_u[0][i * mod[0], j * mod[1]]
                    u_xy[i * ny + j, :] = hess_u[1][i * mod[0], j * mod[1]]
                    u_yy[i * ny + j, :] = hess_u[2][i * mod[0], j * mod[1]]
                    points[i * ny + j, :] = \
                        np.array([self.xGrid[i * mod[0]], self.yGrid[j * mod[1]], self.tauGrid[-1]])
            shape = (nx, ny)
            np.savez(str(fname_path), points=points, u_xx=u_xx, u_yy=u_yy, u_xy=u_xy, shape=shape)
            print(f"Hess. succesfully saved in {str(fname_path)}!")
    
    def plot_solution(self, xlabel=None, ylabel=None, title=None):
        X, Y = self.get_coordinates(grid=True)
        u = self.get_solution(grid=True)
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, u, cmap="coolwarm")
        ax.legend()
        if xlabel is not None and isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        if ylabel is not None and isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        if title is not None and isinstance(title, str):
            ax.set_title(title)
    
    def plot_gradient(self, var=None, xlabel=None, ylabel=None, title=None):
        X, Y = self.get_coordinates(grid=True)
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        if var == "x" or var == "y":
            w = self.get_gradient(var, grid=True)
            ax = fig.add_subplot(111, projection="3d")    
            ax.plot_surface(X, Y, w, cmap="coolwarm")
        else:
            w = self.get_gradient(var, grid=True)
            wx, wy = w[0], w[1]
            w = np.sqrt(np.power(wx, 2) + np.power(wy, 2))
            ax = fig.add_subplot(111)
            pcm = ax.pcolor(X, Y, w, cmap="coolwarm")
            fig.colorbar(pcm, ax=ax, extend="max")
        ax.legend()
        if xlabel is not None and isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        if ylabel is not None and isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        if title is not None and isinstance(title, str):
            ax.set_title(title)
    
    def plot_hessian(self, var=None, xlabel=None, ylabel=None, title=None):
        X, Y = self.get_coordinates(grid=True)
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        if var == "x" or var == "y" or var =="xy":
            w = self.get_hessian(var, grid=True)
            ax = fig.add_subplot(111, projection="3d")    
            ax.plot_surface(X, Y, w, cmap="coolwarm")
            if xlabel is not None and isinstance(xlabel, str):
                ax.set_xlabel(xlabel)
            if ylabel is not None and isinstance(ylabel, str):
                ax.set_ylabel(ylabel)
            if title is not None and isinstance(title, str):
                ax.set_title(title)
            ax.legend()

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