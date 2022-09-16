import numpy as np

from ..geometry.aux_functions import createUniformMesh2D, createLatinHypercubeMesh, createSlices2D


class Domain2D(object):

    def __init__(self, precision="float64"):
        self.int = None
        self.ic = None
        self.top = None
        self.bot = None

        self.n_int = None
        self.n_ic = None
        self.n_top = None
        self.n_bot = None

        self.shape_int = None
        self.shape_ic = None
        self.shape_top = None
        self.shape_bot = None

        self.dx = None
        self.dt = None

        self.label = None
        if precision != "float64" and precision != "float32":
            raise ValueError("float32 or float64 precision is needed!")
        self.__precision = precision

    def loadUniformMesh(self, x_max: float, tau_max: float, n_int: tuple, n_ic: int, n_bound: int):
        # Int domain
        nx_int, ntau_int = n_int[0], n_int[1]
        xGrid = np.linspace(0., x_max, nx_int+1)[1:-1]
        tauGrid = np.linspace(0., tau_max, ntau_int+1)[1:]
        self.int, self.n_int, self.shape_int = createSlices2D(xGrid, tauGrid, self.__precision)
        # Ic domain
        xGrid = np.linspace(0., x_max, n_ic)
        self.ic, self.n_ic, self.shape_ic = createUniformMesh2D(xGrid, np.array([0.]), self.__precision)
        # Top domain
        tauGrid = np.linspace(0., tau_max, n_bound + 1)[1:]
        self.top, self.n_top, self.shape_top = createUniformMesh2D(np.array([x_max]), tauGrid, self.__precision)
        # Bot domain
        self.bot, self.n_bot, self.shape_bot = createUniformMesh2D(np.array([0.]), tauGrid, self.__precision)

        print("-" * 82)
        print(f"Total No of points:             {self.n_int + self.n_ic + self.n_top + self.n_bot}")
        print(f"No of interior points:          {self.n_int}")
        print(f"No of initial condition points: {self.n_ic}")
        print(f"No of bc top points:            {self.n_top}")
        print(f"No of bc bot points:            {self.n_bot}")
        print("-" * 82)

        self.label = "uniform"


class BlackScholesDomain(object):

    def __init__(self, precision="float64"):
        self.int = None
        self.ic = None
        self.top = None
        self.bot = None

        self.n_int = None
        self.n_ic = None
        self.n_top = None
        self.n_bot = None

        self.shape_int = None
        self.shape_ic = None
        self.shape_top = None
        self.shape_bot = None

        self.dx = None
        self.dt = None

        self.label = None
        if precision != "float64" and precision != "float32":
            raise ValueError("float32 or float64 precision is needed!")
        self.__precision = precision

    def loadUniformMesh(self, x_max: float, tau_max: float, n_int: tuple, n_ic: int, n_bounds: int):
        # Int domain
        nx_int, ntau_int = n_int[0], n_int[1]
        xGrid = np.linspace(0., x_max, nx_int+2)[1:-1]
        tauGrid = np.linspace(0., tau_max, ntau_int+1)[1:]
        self.int, self.n_int, self.shape_int = createUniformMesh2D(xGrid, tauGrid, self.__precision)
        # Ic domain
        xGrid = np.linspace(0., x_max, n_ic)
        self.ic, self.n_ic, self.shape_ic = createUniformMesh2D(xGrid, np.array([0.]), self.__precision)
        # Top domain
        tauGrid = np.linspace(0., tau_max, n_bounds + 1)[1:]
        self.top, self.n_top, self.shape_top = createUniformMesh2D(np.array([x_max]), tauGrid, self.__precision)
        # Bot domain
        self.bot, self.n_bot, self.shape_bot = createUniformMesh2D(np.array([0.]), tauGrid, self.__precision)

        print("-" * 82)
        print(f"Total No of points:             {self.n_int + self.n_ic + self.n_top}")
        print(f"No of interior points:          {self.n_int}")
        print(f"No of initial condition points: {self.n_ic}")
        print(f"No of bc bot points:            {self.n_bot}")
        print(f"No of bc top points:            {self.n_top}")
        print("-" * 82)

        self.label = "uniform"

    def loadLatinHypercubeMesh(self, x_max, tau_max, n_int, n_ic, n_bound):
        eps = 1e-8
        # Int domain
        int_min, int_max = np.array([0. + eps, 0. + eps]), np.array([x_max - eps, tau_max])
        self.int = createLatinHypercubeMesh(int_min, int_max, n_int, self.__precision)
        self.n_int = n_int
        # Ic domain
        ic_min, ic_max = np.array([0., 0.]), np.array([x_max, 0.])
        self.ic = createLatinHypercubeMesh(ic_min, ic_max, n_ic, self.__precision)
        self.n_ic = n_ic
        # Top domain
        top_min, top_max = np.array([x_max, 0. + eps]), np.array([x_max, tau_max])
        self.top = createLatinHypercubeMesh(top_min, top_max, n_bound, self.__precision)
        self.n_top = n_bound
        # Bot domain
        bot_min, bot_max = np.array([0., 0. + eps]), np.array([0., tau_max])
        self.bot = createLatinHypercubeMesh(bot_min, bot_max, n_bound, self.__precision)
        self.n_bot = n_bound

        self.label = "LHS"
