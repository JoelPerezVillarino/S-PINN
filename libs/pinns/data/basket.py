import tensorflow as tf
import tensorflow_probability as tfp

from libs.stochastic_processes.processes2D import LognormalProcess2D
from ..utils.payoffs import ArithmeticBasketPayoff
from ..geometry.domains_3d import Domain3D
from ..utils.xva import xVAData


class BasketData(object):

    def __init__(
        self, 
        flag: str, 
        strike: float, 
        process: LognormalProcess2D, 
        geom: Domain3D,
        test: tuple
    ) -> None:
        self.payoff = ArithmeticBasketPayoff(flag, strike)
        self.process = process
        self.geom = geom
        self.test = test

        r = tf.constant(self.process.r, dtype="float64")
        rR_1 = tf.constant(self.process.rR[0], dtype="float64")
        rR_2 = tf.constant(self.process.rR[1], dtype="float64")
        sigma_1 = tf.constant(self.process.sigma[0], dtype="float64")
        sigma_2 = tf.constant(self.process.sigma[1], dtype="float64")
        rho = tf.constant(self.process.rho, dtype="float64")
        un_medio = tf.constant(0.5, dtype="float64")

        def pde(u_t, u_xx, u_yy, u_xy, u_x, u_y, u, x, y):
            res = u_t - un_medio * tf.pow(sigma_1, 2) * tf.pow(x, 2) * u_xx \
                - rho * sigma_1 * sigma_2 * x * y * u_xy \
                - un_medio * tf.pow(sigma_2, 2) * tf.pow(y, 2) * u_yy - rR_1 * x * u_x \
                - rR_2 * y * u_y + r * u
            return res
        
        self.pde = pde

        # Loss
        if self.geom.label == "LHS":

            def loss_fn(inputs):
                return tf.reduce_mean(tf.square(inputs))
            
            def int_loss(model, training):
                return loss_fn(self.pde_residual(self.geom.int, model, training))
            
            def ic_loss(model, training):
                return loss_fn(self.ic_residual(self.geom.ic, model, training))
            
            def s1_bot_loss(model, training):
                return loss_fn(self.pde_residual(self.geom.s_bot, model, training))
            
            def s2_bot_loss(model, training):
                return loss_fn(self.pde_residual(self.geom.v_bot, model, training))
            
            def s1_top_loss(model, training):
                return loss_fn(self.neumann_s1_residual(self.geom.s_top, model, training))
            
            def s2_top_loss(model, training):
                return loss_fn(self.neumann_s2_residual(self.geom.v_top, model, training))
        
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
            
            def s1_bot_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.s_bot, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_s_bot)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_sBot, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_sBot, axis=0)
                return loss
            
            def s2_bot_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.v_bot, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_v_bot)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_vBot, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_vBot, axis=0)
                return loss
            
            def s1_top_loss(model, training):
                loss = tf.pow(self.neumann_s1_residual(self.geom.s_top, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_s_top)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_sTop, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_sTop, axis=0)
                return loss
            
            def s2_top_loss(model, training):
                loss = tf.pow(self.neumann_s2_residual(self.geom.v_top, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_v_top)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_vTop, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_vTop, axis=0)
                return loss
        
        def boundary_losses(model, training):
            s1_bot = s1_bot_loss(model, training)
            s2_bot = s2_bot_loss(model, training)
            s1_top = s1_top_loss(model, training)
            s2_top = s2_top_loss(model, training)
            return [s1_bot, s2_bot, s1_top, s2_top]

        # Allocate loss function
        self.int_loss = int_loss
        self.ic_loss = ic_loss
        self.boundary_losses = boundary_losses
    
    def pde_residual(self, inputs, model, training):
        S1, S2, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S1), tape.watch(S2), tape.watch(T)
            inputs = tf.stack((S1[:, 0], S2[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s1 = tape.gradient(u, S1)
            du_s2 = tape.gradient(u, S2)
        du_t = tape.gradient(u, T)
        du_s1s1 = tape.gradient(du_s1, S1)
        du_s1s2 = tape.gradient(du_s1, S2)
        du_s2s2 = tape.gradient(du_s2, S2)
        del tape

        residual = self.pde(du_t, du_s1s1, du_s2s2, du_s1s2, du_s1, du_s2, u, S1, S2)
        return residual

    def ic_residual(self, inputs, model, training):
        S1, S2 = inputs[:, 0:1], inputs[:, 1:2]
        targets = self.payoff(S1, S2)
        residual = model.net(inputs, training=training) - targets
        return residual
    
    def neumann_s1_residual(self, inputs, model, training):
        S1, S2, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S1), tape.watch(S2), tape.watch(T)
            inputs = tf.stack((S1[:, 0], S2[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s1 = tape.gradient(u, S1)
            du_s2 = tape.gradient(u, S2)
        du_t = tape.gradient(u, T)
        du_s1s2 = tape.gradient(du_s1, S2)
        du_s2s2 = tape.gradient(du_s2, S2)
        del tape
        # Set du_s1s1 = 0
        du_s1s1 = tf.zeros_like(du_s1)
        residual = self.pde(du_t, du_s1s1, du_s2s2, du_s1s2, du_s1, du_s2, u, S1, S2)
        return residual
    
    def neumann_s2_residual(self, inputs, model, training):
        S1, S2, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S1), tape.watch(S2), tape.watch(T)
            inputs = tf.stack((S1[:, 0], S2[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s1 = tape.gradient(u, S1)
        du_t = tape.gradient(u, T)
        du_s2 = tape.gradient(u, S2)
        du_s1s1 = tape.gradient(du_s1, S1)
        du_s1s2 = tape.gradient(du_s1, S2)
        del tape
        # Set du_s2s2=0
        du_s2s2 = tf.zeros_like(du_s2)
        residual = self.pde(du_t, du_s1s1, du_s2s2, du_s1s2, du_s1, du_s2, u, S1, S2)
        return residual
    
    def losses(self, model, training):
        int_loss = self.int_loss(model, training)
        ic_loss = self.ic_loss(model, training)
        b_losses = self.boundary_losses(model, training)
        losses_list = [
            int_loss, ic_loss, b_losses[0], b_losses[1], b_losses[2], b_losses[3]
        ]
        return losses_list


class RiskyBasketData(object):

    def __init__(
        self, 
        flag: str, 
        strike: float, 
        process: LognormalProcess2D,
        xva_data: xVAData,
        geom: Domain3D,
        test: tuple
    ) -> None:
        self.payoff = ArithmeticBasketPayoff(flag, strike)
        self.process = process
        self.xva_data = xva_data
        self.geom = geom
        self.test = test

        r = tf.constant(self.process.r, dtype="float64")
        rR_1 = tf.constant(self.process.rR[0], dtype="float64")
        rR_2 = tf.constant(self.process.rR[1], dtype="float64")
        sigma_1 = tf.constant(self.process.sigma[0], dtype="float64")
        sigma_2 = tf.constant(self.process.sigma[1], dtype="float64")
        rho = tf.constant(self.process.rho, dtype="float64")
        un_medio = tf.constant(0.5, dtype="float64")

        def pde(u_t, u_xx, u_yy, u_xy, u_x, u_y, u, x, y):
            res = u_t - un_medio * tf.pow(sigma_1, 2) * tf.pow(x, 2) * u_xx \
                - rho * sigma_1 * sigma_2 * x * y * u_xy \
                - un_medio * tf.pow(sigma_2, 2) * tf.pow(y, 2) * u_yy - rR_1 * x * u_x \
                - rR_2 * y * u_y + r * u + self.xva_data.source_function(u)
            return res
        
        self.pde = pde

        # Loss
        if self.geom.label == "LHS":

            def loss_fn(inputs):
                return tf.reduce_mean(tf.square(inputs))
            
            def int_loss(model, training):
                return loss_fn(self.pde_residual(self.geom.int, model, training))
            
            def ic_loss(model, training):
                return loss_fn(self.ic_residual(self.geom.ic, model, training))
            
            def s1_bot_loss(model, training):
                return loss_fn(self.pde_residual(self.geom.s_bot, model, training))
            
            def s2_bot_loss(model, training):
                return loss_fn(self.pde_residual(self.geom.v_bot, model, training))
            
            def s1_top_loss(model, training):
                return loss_fn(self.neumann_s1_residual(self.geom.s_top, model, training))
            
            def s2_top_loss(model, training):
                return loss_fn(self.neumann_s2_residual(self.geom.v_top, model, training))
        
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
            
            def s1_bot_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.s_bot, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_s_bot)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_sBot, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_sBot, axis=0)
                return loss
            
            def s2_bot_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.v_bot, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_v_bot)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_vBot, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_vBot, axis=0)
                return loss
            
            def s1_top_loss(model, training):
                loss = tf.pow(self.neumann_s1_residual(self.geom.s_top, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_s_top)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dv_sTop, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_sTop, axis=0)
                return loss
            
            def s2_top_loss(model, training):
                loss = tf.pow(self.neumann_s2_residual(self.geom.v_top, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_v_top)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.ds_vTop, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dtau_vTop, axis=0)
                return loss
        
        def boundary_losses(model, training):
            s1_bot = s1_bot_loss(model, training)
            s2_bot = s2_bot_loss(model, training)
            s1_top = s1_top_loss(model, training)
            s2_top = s2_top_loss(model, training)
            return [s1_bot, s2_bot, s1_top, s2_top]
        
        # Allocate loss function
        self.int_loss = int_loss
        self.ic_loss = ic_loss
        self.boundary_losses = boundary_losses
    
    def pde_residual(self, inputs, model, training):
        S1, S2, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S1), tape.watch(S2), tape.watch(T)
            inputs = tf.stack((S1[:, 0], S2[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s1 = tape.gradient(u, S1)
            du_s2 = tape.gradient(u, S2)
        du_t = tape.gradient(u, T)
        du_s1s1 = tape.gradient(du_s1, S1)
        du_s1s2 = tape.gradient(du_s1, S2)
        du_s2s2 = tape.gradient(du_s2, S2)
        del tape

        residual = self.pde(du_t, du_s1s1, du_s2s2, du_s1s2, du_s1, du_s2, u, S1, S2)
        return residual

    def ic_residual(self, inputs, model, training):
        S1, S2 = inputs[:, 0:1], inputs[:, 1:2]
        targets = self.payoff(S1, S2)
        residual = model.net(inputs, training=training) - targets
        return residual
    
    def neumann_s1_residual(self, inputs, model, training):
        S1, S2, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S1), tape.watch(S2), tape.watch(T)
            inputs = tf.stack((S1[:, 0], S2[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s1 = tape.gradient(u, S1)
            du_s2 = tape.gradient(u, S2)
        du_t = tape.gradient(u, T)
        du_s1s2 = tape.gradient(du_s1, S2)
        du_s2s2 = tape.gradient(du_s2, S2)
        del tape
        # Set du_s1s1 = 0
        du_s1s1 = tf.zeros_like(du_s1)
        residual = self.pde(du_t, du_s1s1, du_s2s2, du_s1s2, du_s1, du_s2, u, S1, S2)
        return residual
    
    def neumann_s2_residual(self, inputs, model, training):
        S1, S2, T = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S1), tape.watch(S2), tape.watch(T)
            inputs = tf.stack((S1[:, 0], S2[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s1 = tape.gradient(u, S1)
        du_t = tape.gradient(u, T)
        du_s2 = tape.gradient(u, S2)
        du_s1s1 = tape.gradient(du_s1, S1)
        du_s1s2 = tape.gradient(du_s1, S2)
        del tape
        # Set du_s2s2=0
        du_s2s2 = tf.zeros_like(du_s2)
        residual = self.pde(du_t, du_s1s1, du_s2s2, du_s1s2, du_s1, du_s2, u, S1, S2)
        return residual
    
    def losses(self, model, training):
        int_loss = self.int_loss(model, training)
        ic_loss = self.ic_loss(model, training)
        b_losses = self.boundary_losses(model, training)
        losses_list = [
            int_loss, ic_loss, b_losses[0], b_losses[1], b_losses[2], b_losses[3]
        ]
        return losses_list

