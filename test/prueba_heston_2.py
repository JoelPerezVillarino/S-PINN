from pathlib import Path
import numpy as np
import tensorflow as tf

import libs.pinns as pn
from libs.stochastic_processes import HestonProcess
from libs.ql_asset_valuator.vanilla_heston import VanillaHestonQL
from libs.pinns.geometry.aux_functions import createLatinHypercubeMesh


def main(ntrials=1):
    # Model parameters
    kappa = 1.5
    eta = 0.04
    sigma = 0.3
    rho = -0.9
    r = 0.025
    rR = 0.025
    # rR = r - q -> q = r - rR in order to obtain the same equation
    q = r - rR

    heston_process = HestonProcess(r, q, kappa, eta, sigma, rho)

    # Option parameters
    strike = 100.
    maturity = 2.
    flag="p"

    # Domain parameters
    s_min, s_max = (0., 4. * strike)
    v_min, v_max = (0., 3.)
    tau_min, tau_max = (0., maturity)

    n_int = int(10e5)
    n_ic = int(10e4)
    n_bound_s = int(10e4)
    n_bound_v = n_bound_s
    
    # Defining NN
    layers = 4
    units = 60
    activation = "tanh"
    kernel_initializer = "glorot_uniform"
    maxVals = tf.constant([s_max, v_max, maturity], dtype=dtype)
    feature_transform = lambda inputs: inputs / maxVals
    scaling_factor = s_max - strike

    # Train features
    metrics = [
        "l1 relative error", "l2 relative error", "linf relative error"
    ]

    loss_weights_spinns = tf.constant(
            [
                1. / (s_max * v_max * maturity), 1. / (s_max * v_max),
                1. / (v_max * maturity), 1. / (s_max * maturity),
                1. / (v_max * maturity), 1. / (s_max * maturity), 1.
            ], dtype=dtype
        )

    loss_weights_spvsd = tf.constant(
            [
                0, 0,
                0, 0,
                0, 0, 1.
            ], dtype=dtype
        )
    epochs = 25000
    decay = ["inverse time", 10000, 0.5]
    

    #fname = Path.cwd().parent.absolute().joinpath("PricesHeston.npz")
    #np.savez(fname, int=geom.int, v=vs)
    #p_data = (geom.int, vs)

    # Load mesh pinns
    fname = Path.cwd().joinpath("geometryHeston.npz")
    dat = np.load(str(fname))
    geom = pn.Domain3D(precision="float64")
    geom.int = tf.constant(dat["int"], dtype=dtype)
    geom.ic = tf.constant(dat["ic"], dtype=dtype)
    geom.s_bot = tf.constant(dat["sbot"], dtype=dtype)
    geom.v_bot = tf.constant(dat["vbot"], dtype=dtype)
    geom.s_top = tf.constant(dat["stop"], dtype=dtype)
    geom.v_top = tf.constant(dat["vtop"], dtype=dtype)

    # test
    fname_test = Path.cwd().parent.absolute().joinpath(
        "datasets_pinns/Heston_Put_Limited/riskfree_solution.npz"
    )
    data_test = np.load(str(fname_test))
    test = (data_test["points"], data_test["u"])

    points_list = np.linspace(1000, 10e5, 12, dtype=np.int)

    for i, n_points in enumerate(points_list[:8]):
        print(82*"--")
        print(f"PASO {i}; NUMERO DE PUNTOS: {n_points}")
        print(82*"--")
        # 1) Space domain
        space_points = createLatinHypercubeMesh(
            np.array([0., 0., 0.]), np.array([s_max, v_max, tau_max]), n_points, "float64")

        # 2) heston prices in space_points
        vs = np.zeros((space_points.shape[0], 1))
        for i in range(space_points.shape[0]):
            vhQL = VanillaHestonQL("p", strike, space_points[i, -1].numpy(), heston_process)
            vs[i, 0] = vhQL.evaluate(space_points[i, :-1].numpy())
        
        fname = Path.cwd().joinpath(f"pricesHeston_n_{n_points}.npz")
        np.savez(str(fname), int=space_points, v=vs)

        # 3) Data 
        data_supvsd = (space_points, vs)

        # Load PDE data instance    
        data = pn.HestonData2(flag, strike, heston_process, data_supvsd, geom, test)

        # Load net
        net = pn.ScaledOutputFNN(
            [3] + [units] * layers + [1],
            activation,
            kernel_initializer, 
            scaling_factor=scaling_factor
        )
        net.apply_feature_transform(feature_transform)

        # load model
        model = pn.Model(data, net)

        # Train spinns
        model.compile("adam", lr=2e-3, metrics=metrics, decay=decay, 
                        loss_weights=loss_weights_spinns)
        loss_history, _ = model.train(epochs, display_every=1000)

        # Save loss history and net model
        fname_loss = save_history_path.joinpath(f"loss_history_spinns_n_{n_points}")
        loss_history.save(str(fname_loss))
        fname_model = save_model_path.joinpath(f"model_spinns_n_{n_points}")
        tf.saved_model.save(net, str(fname_model))

        # Load net
        net = pn.ScaledOutputFNN(
            [3] + [units] * layers + [1],
            activation,
            kernel_initializer, 
            scaling_factor=scaling_factor
        )
        net.apply_feature_transform(feature_transform)

        # load model
        model = pn.Model(data, net)

        # Train spinns
        model.compile("adam", lr=2e-3, metrics=metrics, decay=decay, 
                        loss_weights=loss_weights_spvsd)
        loss_history, _ = model.train(epochs, display_every=1000)

        # Save loss history and net model
        fname_loss = save_history_path.joinpath(f"loss_history_spvsd_n_{n_points}")
        loss_history.save(str(fname_loss))
        fname_model = save_model_path.joinpath(f"model_spvsd_n_{n_points}")
        tf.saved_model.save(net, str(fname_model))

    return


if __name__ == '__main__':
    dtype = "float64"
    tf.keras.backend.set_floatx(dtype)
    # Path managements (save results)
    path = Path.cwd().parent.absolute()
    results_path = path.joinpath("results")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    results_path = results_path.joinpath("PRUEBAS")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    results_path = results_path.joinpath("test_n_points")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    save_history_path = results_path.joinpath("loss_history")
    save_history_path.mkdir(parents=True, exist_ok=True)
    save_model_path = results_path.joinpath("models")
    save_model_path.mkdir(parents=True, exist_ok=True)

    main()
