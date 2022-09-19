from typing import Tuple
import numpy as np

from ..geometry.aux_functions import createLatinHypercubeMesh, createUniformMesh3D


class Domain3D(object):

    def __init__(self , precision="float64") -> None:
        self.int = None
        self.ic = None
        self.s_bot = None
        self.v_bot = None
        self.s_top = None
        self.v_top = None
        
        self.n_int = None
        self.n_ic = None
        self.n_s_bot = None
        self.n_v_bot = None
        self.n_s_top = None
        self.n_v_top = None

        self.shape_int = None
        self.shape_ic = None
        self.shape_s_bot = None
        self.shape_v_bot = None
        self.shape_s_top = None
        self.shape_v_top = None

        self.ds_int = None
        self.dv_int = None
        self.dtau_int = None

        self.ds_ic = None
        self.dv_ic = None
        
        self.ds_vBot = None
        self.dtau_vBot = None
        
        self.ds_vTop = None
        self.dtau_vTop = None

        self.dv_sBot = None
        self.dtau_sBot = None
        
        self.dv_sTop = None
        self.dtau_sTop = None

        self.label = None
        if precision != "float64" and precision != "float32":
            raise ValueError("float32 or float64 precision is needed!")
        self.__precision = precision
    
    def loadUniformMesh(
        self, 
        s_max: float, 
        v_max: float, 
        tau_max: float, 
        n_int: Tuple[int, int, int],
        n_ic: Tuple[int, int],
        n_bound_s: Tuple[int, int],
        n_bound_v: Tuple[int, int]
    ):
        # Int domain
        ns_int, nv_int, ntau_int = n_int[0], n_int[1], n_int[2]
        sGrid = np.linspace(0., s_max, ns_int+2)[1:-1]
        vGrid = np.linspace(0., v_max, nv_int+2)[1:-1]
        tauGrid = np.linspace(0., tau_max, ntau_int+1)[1:]
        self.int, self.n_int, self.shape_int = createUniformMesh3D(
            sGrid, vGrid, tauGrid, self.__precision
        )
        self.ds_int = sGrid[1] - sGrid[0]
        self.dv_int = vGrid[1] - vGrid[0]
        self.dtau_int = tauGrid[1] - tauGrid[0]
        # IC
        ns_ic, nv_ic = n_ic[0], n_ic[1]
        sGrid = np.linspace(0., s_max, ns_ic)
        vGrid = np.linspace(0., v_max, nv_ic)
        self.ic, self.n_ic, self.shape_ic = createUniformMesh3D(
            sGrid, vGrid, np.array([0.]), self.__precision
        )
        self.ds_ic = sGrid[1] - sGrid[0]
        self.dv_ic = vGrid[1] - vGrid[0]
        # V Bot
        ns_bound, ntau_bound = n_bound_v[0], n_bound_v[1]
        sGrid = np.linspace(0., s_max, ns_bound+1)[1:]
        tauGrid = np.linspace(0., tau_max, ntau_bound+1)[1:]
        self.v_bot, self.n_v_bot, self.shape_v_bot = createUniformMesh3D(
            sGrid, np.array([0.]), tauGrid, self.__precision
        )
        self.ds_vBot = sGrid[1] - sGrid[0]
        self.dtau_vBot = tauGrid[1] - tauGrid[0]
        # V Top
        self.v_top, self.n_v_top, self.shape_v_top = createUniformMesh3D(
            sGrid, np.array([v_max]), tauGrid, self.__precision
        )
        self.ds_vTop = sGrid[1] - sGrid[0]
        self.dtau_vTop = tauGrid[1] - tauGrid[0]
        # S Top
        nv_bound, ntau_bound = n_bound_s[0], n_bound_s[1]
        vGrid = np.linspace(0., v_max, nv_bound+2)[:-1]
        tauGrid = np.linspace(0., tau_max, ntau_bound+1)[1:]
        self.s_top, self.n_s_top, self.shape_s_top = createUniformMesh3D(
            np.array([s_max]), vGrid, tauGrid, self.__precision
        )
        self.dv_sTop = vGrid[1] - vGrid[0]
        self.dtau_sTop = tauGrid[1] - tauGrid[0]
        # S Bot
        vGrid = np.linspace(0., v_max, nv_bound)
        self.s_bot, self.n_s_bot, self.shape_s_bot = createUniformMesh3D(
            np.array([0.]), vGrid, tauGrid, self.__precision
        )
        self.dv_sBot = vGrid[1] - vGrid[0]
        self.dtau_sBot = tauGrid[1] - tauGrid[0]

        nTotal = self.n_int + self.n_ic + self.n_v_bot + self.n_v_top + self.n_s_top + self.n_s_bot
        print("-" * 82)
        print(f"Total No of points:             {nTotal}")
        print(f"No of interior points:          {self.n_int}")
        print(f"No of initial condition points: {self.n_ic}")
        print(f"No of bc bot S points:          {self.n_s_bot}")
        print(f"No of bc bot V points:          {self.n_v_bot}")
        print(f"No of bc top S points:          {self.n_s_top}")
        print(f"No of bc top V points:          {self.n_v_top}")
        print("-" * 82)

        self.label = "uniform"

    def loadLatinHypercubeMesh(
        self,
        s_max: float, 
        v_max: float, 
        tau_max: float, 
        n_int: int, 
        n_ic: int, 
        n_bound_s: int, 
        n_bound_v: int
    ):
        eps = 1e-8
        # Int domain
        int_min = np.array([0. + eps, 0. + eps, 0. + eps])
        int_max = np.array([s_max - eps, v_max - eps, tau_max])
        self.int = createLatinHypercubeMesh(int_min, int_max, n_int, self.__precision)
        self.n_int = n_int
        # IC
        ic_min, ic_max = np.array([0., 0., 0.]), np.array([s_max, v_max, 0.])
        self.ic = createLatinHypercubeMesh(ic_min, ic_max, n_ic, self.__precision)
        self.n_ic = n_ic
        # Bot S
        bots_min, bots_max = np.array([0., 0., 0. + eps]), np.array([0., v_max, tau_max])
        self.s_bot = createLatinHypercubeMesh(bots_min, bots_max, n_bound_s, self.__precision)
        self.n_s_bot = n_bound_s
        # Bot V
        botv_min, botv_max = np.array([0. + eps, 0., 0. + eps]), np.array([s_max, 0., tau_max])
        self.v_bot = createLatinHypercubeMesh(botv_min, botv_max, n_bound_v, self.__precision)
        self.n_v_bot = n_bound_v
        # Top S
        tops_min, tops_max = np.array([s_max, 0. + eps, 0. + eps]), np.array([s_max, v_max-eps, tau_max])
        self.s_top = createLatinHypercubeMesh(tops_min, tops_max, n_bound_s, self.__precision)
        self.n_s_top = n_bound_s
        # Top V
        topv_min, topv_max = np.array([0. + eps, v_max, 0. + eps]), np.array([s_max, v_max, tau_max])
        self.v_top = createLatinHypercubeMesh(topv_min, topv_max, n_bound_v, self.__precision)
        self.n_v_top = n_bound_v

        self.label = "LHS"