import tensorflow as tf
import tensorflow_probability as tfp

from libs.stochastic_processes.processes2D import HestonProcess
from ..utils.payoffs import VanillaPayoff
from ..utils.xva import xVAData
from ..geometry.domains_3d import Domain3D


class HestonData2(object):

    def __init__(
        self, 
        flag: str, 
        strike: float, 
        process: HestonProcess,
        data: tuple,
        geom: Domain3D, 
        test: tuple
    ) -> None:
        self.payoff = VanillaPayoff(flag, strike)
        self.process = process
        self.geom = geom
        self.data = data
        self.test = test

        r = tf.constant(self.process.r, dtype="float64")
        rR = tf.constant(self.process.rR, dtype="float64")
        q = tf.constant(self.process.q, dtype="float64")
        kappa = tf.constant(self.process.kappa, dtype="float64")
        eta = tf.constant(self.process.eta, dtype="float64")
        sigma = tf.constant(self.process.sigma, dtype="float64")
        rho = tf.constant(self.process.rho, dtype="float64")
        un_medio = tf.constant(0.5, dtype="float64")

        def pde(u_t, u_ss, u_vv, u_sv, u_s, u_v, u, s, v):
            term = u_t - un_medio * tf.pow(s, 2) * v * u_ss \
                - rho * sigma * s * v * u_sv - un_medio * tf.pow(sigma, 2) * v * u_vv \
                - rR * s * u_s - kappa * (eta - v) * u_v + r * u
            return term
        
        # Allocate functions
        self.pde = pde

        # Loss
        def loss_fn(inputs):
            return tf.reduce_mean(tf.square(inputs))
        
        def int_loss(model, training):
            return loss_fn(self.pde_residual(self.geom.int, model, training))
        
        def ic_loss(model, training):
            return loss_fn(self.ic_residual(self.geom.ic, model, training))
        
        def s_bot_loss(model, training):
            return loss_fn(self.pde_residual(self.geom.s_bot, model, training))
        
        def v_bot_loss(model, training):
            return loss_fn(self.pde_residual(self.geom.v_bot, model, training))
        
        def s_top_loss(model, training):
            return loss_fn(self.neumann_s_residual(self.geom.s_top, model, training))
        
        def v_top_loss(model, training):
            return loss_fn(self.neumann_v_residual(self.geom.v_top, model, training))
        
        def loss_prices(model, training):
            y_pred = model.net(self.data[0], training=training)
            inputs = y_pred - self.data[1]
            return tf.math.sqrt(tf.reduce_sum(tf.square(inputs))) / tf.math.sqrt(tf.cast(inputs.shape[0], dtype=tf.float64))
    
        def boundary_losses(model, training):
            s_bot = s_bot_loss(model, training)
            v_bot = v_bot_loss(model, training)
            s_top = s_top_loss(model, training)
            v_top = v_top_loss(model, training)
            return [s_bot, v_bot, s_top, v_top]
        
        # Allocate loss function
        self.int_loss = int_loss
        self.loss_prices = loss_prices
        self.ic_loss = ic_loss
        self.boundary_losses = boundary_losses
    
    def pde_residual(self, inputs, model, training):
        S, V, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S), tape.watch(V), tape.watch(T)
            inputs = tf.stack((S[:, 0], V[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s = tape.gradient(u, S)
            du_v = tape.gradient(u, V)
        du_t = tape.gradient(u, T)
        # Second order derivatives
        du_ss = tape.gradient(du_s, S)
        du_vv = tape.gradient(du_v, V)
        du_sv = tape.gradient(du_s, V)
        del tape

        res = self.pde(du_t, du_ss, du_vv, du_sv, du_s, du_v, u, S, V)
        return res
    
    def ic_residual(self, inputs, model, training):
        targets = self.payoff(inputs[:, 0:1])
        res = model.net(inputs, training=training) - targets
        return res
    
    def dirichlet_v_residual(self, inputs, model, training):
        targets = self._aux_v_condition(inputs)
        res = model.net(inputs, training=training) - targets
        return res

    def neumann_v_residual(self, inputs, model, training):
        S, V, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S), tape.watch(V), tape.watch(T)
            inputs = tf.stack((S[:, 0], V[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s = tape.gradient(u, S)
            du_v = tape.gradient(u, V)
        du_t = tape.gradient(u, T)
        # Second order derivatives
        du_ss = tape.gradient(du_s, S)
        du_vv = tape.gradient(du_v, V)
        du_sv = tape.gradient(du_s, V)
        del tape
        # Set du_v = 0
        du_v = tf.zeros_like(du_s)
        res = self.pde(du_t, du_ss, du_vv, du_sv, du_s, du_v, u, S, V)
        return res
    
    def neumann_s_residual(self, inputs, model, training):
        S, V, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S), tape.watch(V), tape.watch(T)
            inputs = tf.stack((S[:, 0], V[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s = tape.gradient(u, S)
            du_v = tape.gradient(u, V)
        du_t = tape.gradient(u, T)
        # Second order derivatives
        #du_ss = tape.gradient(du_s, S)
        du_vv = tape.gradient(du_v, V)
        du_sv = tape.gradient(du_s, V)
        del tape
        du_ss = tf.zeros_like(du_s)
        
        #du_s = self._aux_s_condition(inputs)
        res = self.pde(du_t, du_ss, du_vv, du_sv, du_s, du_v, u, S, V)
        return res
    
    def losses(self, model, training):
        int_loss = self.int_loss(model, training)
        ic_loss = self.ic_loss(model, training)
        loss_prices = self.loss_prices(model, training)
        b_losses = self.boundary_losses(model, training)
        losses_list = [
            int_loss, ic_loss, b_losses[0], b_losses[1], b_losses[2], b_losses[3], loss_prices
        ]
        return losses_list


