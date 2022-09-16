__all__ = ["LBFGS_options", "set_LBFGS_options"]


LBFGS_options = {}


def set_LBFGS_options(
        maxcor=100,
        ftol=0.,
        gtol=1e-8,
        maxiter=1500,
        maxfun=None,
        maxls=50,
):
    """Sets the hyperparameters of L-BFGS.
    Args:
        maxcor (int): The maximum number of variable metric corrections used to
            define the limited memory matrix.
        ftol (float): The iteration stops when
            (f_k - f_{k+1}) / max{|f_k|, |f_{k+1}|, 1} <= ftol
        gtol (float): The iteration will stop when
            max{|proj g_i|, i= 1...n} <= gtol,
            where g_i is the i-th component of the projected gradient
        maxiter (int): Maximum number of iterations.
        maxfun (int): Maximum number of function evaluations
        maxls (int): Maximum number of line search steps (per iteration).
    """

    global LBFGS_options
    LBFGS_options["maxcor"] = maxcor
    LBFGS_options["ftol"] = ftol
    LBFGS_options["gtol"] = gtol
    LBFGS_options["maxiter"] = maxiter
    LBFGS_options["maxfun"] = maxfun if maxfun is not None else int(maxiter * 1.25)
    LBFGS_options["maxls"] = maxls

set_LBFGS_options()
