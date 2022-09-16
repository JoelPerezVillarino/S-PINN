import tensorflow as tf
import tensorflow_probability as tfp

from libs.stochastic_processes.processes2D import HestonProcess
from ..utils.payoffs import VanillaPayoff
from ..utils.xva import xVAData
from ..geometry.domains_3d import Domain3D


class HestonData(object):

    def __init__(
        self, 
        flag: str, 
        strike: float, 
        process: HestonProcess, 
        geom: Domain3D, 
        test: tuple
    ) -> None:
        self.payoff = VanillaPayoff(flag, strike)
        self.process = process
        self.geom = geom
        self.test = test

        r = tf.constant(self.process.r, dtype="float64")
        rR = tf.constant(self.process.rR, dtype="float64")
        q = tf.constant(self.process.q, dtype="float64")
        kappa = tf.constant(self.process.kappa, dtype="float64")
        eta = tf.constant(self.process.eta, dtype="float64")
        sigma = tf.constant(self.process.sigma, dtype="float64")
        rho = tf.constant(self.process.rho, dtype="float64")
        un_medio = tf.constant(0.5, dtype="float64")

        # Load Normal distribution
        dist = tfp.distributions.Normal(
            loc=tf.constant(0., dtype="float64"), scale=tf.constant(1., dtype="float64")
        )

        def pde(u_t, u_ss, u_vv, u_sv, u_s, u_v, u, s, v):
            term = u_t - un_medio * tf.pow(s, 2) * v * u_ss \
                - rho * sigma * s * v * u_sv - un_medio * tf.pow(sigma, 2) * v * u_vv \
                - rR * s * u_s - kappa * (eta - v) * u_v + r * u
            return term
        
        def feller_pde(u_t, u_s, u_v, u, s, ):
            term = u_t - rR * s * u_s - kappa * eta * u_v + r * u
            return term
        # Allocate functions
        self.pde = pde
        self.feller_pde = feller_pde
        
        # Loss
        if self.geom.label == "LHS":

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
        
        elif self.geom.label == "uniform":
            
            def int_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.int, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_int)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_int, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_int, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_int, axis=0)
                return loss
            
            def ic_loss(model, training):
                loss = tf.pow(self.ic_residual(self.geom.ic, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_ic)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_ic, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_ic, axis=0)
                loss = tf.squeeze(loss)
                return loss
            
            def s_bot_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.s_bot, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_s_bot)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_sBot, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_sBot, axis=0)
                return loss
            
            def v_bot_loss(model, training):
                loss = tf.pow(self.feller_residual(self.geom.v_bot, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_v_bot)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_vBot, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_vBot, axis=0)
                return loss
            
            def s_top_loss(model, training):
                loss = tf.pow(self.neumann_s_residual(self.geom.s_top, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_s_top)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_sTop, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_sTop, axis=0)
                return loss
            
            def v_top_loss(model, training):
                loss = tf.pow(self.neumann_v_residual(self.geom.v_top, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_v_top)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_vTop, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_vTop, axis=0)
                return loss
            
        def boundary_losses(model, training):
            s_bot = s_bot_loss(model, training)
            v_bot = v_bot_loss(model, training)
            s_top = s_top_loss(model, training)
            v_top = v_top_loss(model, training)
            return [s_bot, v_bot, s_top, v_top]
        
        # Allocate loss function
        self.int_loss = int_loss
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
    
    def feller_residual(self, inputs, model, training):
        S, V, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S), tape.watch(V), tape.watch(T)
            inputs = tf.stack((S[:, 0], V[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s = tape.gradient(u, S)
            du_v = tape.gradient(u, V)
        du_t = tape.gradient(u, T)
        del tape
        res = self.feller_pde(du_t, du_s, du_v, u, S)
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
        du_vv = tape.gradient(du_v, V)
        du_sv = tape.gradient(du_s, V)
        del tape
        du_ss = tf.zeros_like(du_s)
        
        res = self.pde(du_t, du_ss, du_vv, du_sv, du_s, du_v, u, S, V)
        return res
    
    def losses(self, model, training):
        int_loss = self.int_loss(model, training)
        ic_loss = self.ic_loss(model, training)
        b_losses = self.boundary_losses(model, training)
        losses_list = [
            int_loss, ic_loss, b_losses[0], b_losses[1], b_losses[2], b_losses[3]
        ]
        return losses_list


class RiskyHestonData(object):

    def __init__(
        self, 
        flag: str, 
        strike: float, 
        process: HestonProcess,
        xva_data: xVAData, 
        geom: Domain3D, 
        test: tuple
    ) -> None:
        self.payoff = VanillaPayoff(flag, strike)
        self.process = process
        self.xva_data = xva_data
        self.geom = geom
        self.test = test

        r = tf.constant(self.process.r, dtype="float64")
        q = tf.constant(self.process.q, dtype="float64")
        rR = tf.constant(self.process.rR, dtype="float64")
        kappa = tf.constant(self.process.kappa, dtype="float64")
        eta = tf.constant(self.process.eta, dtype="float64")
        sigma = tf.constant(self.process.sigma, dtype="float64")
        rho = tf.constant(self.process.rho, dtype="float64")
        un_medio = tf.constant(0.5, dtype="float64")

        def pde(u_t, u_ss, u_vv, u_sv, u_s, u_v, u, s, v):
            term = u_t - un_medio * tf.pow(s, 2) * v * u_ss \
                - rho * sigma * s * v * u_sv - un_medio * tf.pow(sigma, 2) * v * u_vv \
                - rR * s * u_s - kappa * (eta - v) * u_v + r * u + xva_data.source_function(u)
            return term
        
        def feller_pde(u_t, u_s, u_v, u, s, ):
            term = u_t - rR * s * u_s - kappa * eta * u_v + r * u + xva_data.source_function(u)
            return term
        
        # Allocate functions
        self.pde = pde
        self.feller_pde = feller_pde

        # Loss
        if self.geom.label == "LHS":

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
        
        elif self.geom.label == "uniform":

            def int_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.int, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_int)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_int, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_int, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_int, axis=0)
                return loss
            
            def ic_loss(model, training):
                loss = tf.pow(self.ic_residual(self.geom.ic, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_ic)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_ic, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_ic, axis=0)
                loss = tf.squeeze(loss)
                return loss
            
            def s_bot_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.s_bot, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_s_bot)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_sBot, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_sBot, axis=0)
                return loss
            
            def v_bot_loss(model, training):
                loss = tf.pow(self.feller_residual(self.geom.v_bot, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_v_bot)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_vBot, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_vBot, axis=0)
                return loss
            
            def s_top_loss(model, training):
                loss = tf.pow(self.neumann_s_residual(self.geom.s_top, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_s_top)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_sTop, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_sTop, axis=0)
                return loss
            
            def v_top_loss(model, training):
                loss = tf.pow(self.neumann_v_residual(self.geom.v_top, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_v_top)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_vTop, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_vTop, axis=0)
                return loss
            
        def boundary_losses(model, training):
            s_bot = s_bot_loss(model, training)
            v_bot = v_bot_loss(model, training)
            s_top = s_top_loss(model, training)
            v_top = v_top_loss(model, training)
            return [s_bot, v_bot, s_top, v_top]
        
        # Allocate loss function
        self.int_loss = int_loss
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
    
    def feller_residual(self, inputs, model, training):
        S, V, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S), tape.watch(V), tape.watch(T)
            inputs = tf.stack((S[:, 0], V[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s = tape.gradient(u, S)
            du_v = tape.gradient(u, V)
        du_t = tape.gradient(u, T)
        del tape

        res = self.feller_pde(du_t, du_s, du_v, u, S)
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
        res = self.pde(du_t, du_ss, du_vv, du_sv, du_s, tf.zeros_like(du_s), u, S, V)
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
        du_vv = tape.gradient(du_v, V)
        du_sv = tape.gradient(du_s, V)
        del tape
        
        du_ss = tf.zeros_like(du_s)

        res = self.pde(du_t, du_ss, du_vv, du_sv, du_s, du_v, u, S, V)
        return res
    
    def losses(self, model, training):
        int_loss = self.int_loss(model, training)
        ic_loss = self.ic_loss(model, training)
        b_losses = self.boundary_losses(model, training)
        losses_list = [
            int_loss, ic_loss, b_losses[0], b_losses[1], b_losses[2], b_losses[3]
        ]
        return losses_list