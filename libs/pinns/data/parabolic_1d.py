import tensorflow as tf
import tensorflow_probability as tfp

from libs.stochastic_processes.processes1D import LognormalProcess1D
from  ..utils.payoffs import VanillaPayoff
from ..geometry.domains_2d import Domain2D
from ..utils.xva import xVAData


class BlackScholes1D(object):

    def __init__(
        self,
        flag: str,
        strike: float,
        process: LognormalProcess1D,
        geom: Domain2D, 
        test: tuple
    ):
        if not isinstance(geom, Domain2D):
            raise ValueError("geom must be a instance of Domain2D!.")
        self.payoff = VanillaPayoff(flag, strike)
        self.process = process
        self.geom = geom
        self.test = test

        # Loading pde
        r = tf.constant(self.process.r, dtype="float64")
        rR = tf.constant(self.process.rR, dtype="float64")
        sigma = tf.constant(self.process.sigma, dtype="float64")
        un_medio = tf.constant(0.5, dtype="float64")

        def pde(u_t, u_xx, u_x, u, x):
            res = u_t - un_medio * tf.pow(sigma, 2) * tf.pow(x, 2) * u_xx - rR * x * u_x + r * u
            return res
        
        self.pde = pde


        if self.geom.label == "LHS":

            def loss_fn(inputs):
                return tf.reduce_mean(tf.square(inputs))
            
            def int_loss(model, training):
                return loss_fn(self.pde_residual(self.geom.int, model, training))
            
            def ic_loss(model, training):
                return loss_fn(self.ic_residual(self.geom.ic, model, training))
            
            def bot_loss(model, training):
                return loss_fn(self.pde_residual(self.geom.bot, model, training))
            
            def top_loss(model, training):
                return loss_fn(self.neumann_residual(self.geom.top, model, training))

        elif self.geom.label == "uniform":

            def int_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.int, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_int)
                loss = tfp.math.trapz(loss, dx=self.geom.dx, axis=0)
                loss = tfp.math.trapz(loss, dx=self.geom.dt, axis=0)
                return loss
            
            def ic_loss(model, training):
                loss = tf.pow(self.ic_residual(self.geom.ic, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_ic)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dx, axis=0)
                return loss

            def bot_loss(model, training):
                loss = tf.pow(self.pde_residual(self.geom.bot, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_bot)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dt, axis=0)
                return loss
            
            def top_loss(model, training):
                loss = tf.pow(self.neumann_residual(self.geom.top, model, training), 2)
                loss = tf.reshape(tf.transpose(loss), self.geom.shape_top)
                loss = tf.squeeze(loss)
                loss = tfp.math.trapz(loss, dx=self.geom.dt, axis=0)
                return loss

        def boundary_losses(model, training):
            bot = bot_loss(model, training)
            top = top_loss(model, training)
            return [bot, top]
        
        # Allocate loss function
        self.int_loss = int_loss
        self.ic_loss = ic_loss
        self.boundary_losses = boundary_losses

    def pde_residual(self, inputs, model, training):
        S, T = inputs[:, 0:1], inputs[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S), tape.watch(T)
            inputs = tf.stack((S[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
            du_s = tape.gradient(u, S)

        du_t = tape.gradient(u, T)
        du_ss = tape.gradient(du_s, S)
        del tape

        residual = self.pde(du_t, du_ss, du_s, u, S, )
        return residual

    def ic_residual(self, inputs, model, training):
        targets = self.ic(inputs)
        residual = model.net(inputs, training=training) - targets
        return residual

    def neumann_residual(self, inputs, model, training):
        S, T = inputs[:, 0:1], inputs[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S), tape.watch(T)
            inputs = tf.stack((S[:, 0], T[:, 0]), axis=1)
            u = model.net(inputs, training=training)
        du_s = tape.gradient(u, S)
        du_t = tape.gradient(u, T)
        del tape
        du_ss = tf.zeros_like(du_s)

        residual = self.pde(du_t, du_ss, du_s, u, S, )
        return residual
    
    def losses(self, model, training):
        int_loss = self.int_loss(model, training)
        ic_loss = self.ic_loss(model, training)
        b_losses = self.boundary_losses(model, training)
        losses_list = [int_loss, ic_loss, b_losses[0], b_losses[1]]
        return losses_list


class RiskyBlackScholes1D(BlackScholes1D):

    def __init__(
        self, 
        flag: str, 
        strike: float, 
        process: LognormalProcess1D,
        xva_data: xVAData,
        geom: Domain2D, 
        test: tuple
    ):
        super().__init__(flag, strike, process, geom, test)
        self.xva_data = xva_data
        # Loading pde
        r = tf.constant(self.process.r, dtype="float64")
        rR = tf.constant(self.process.rR, dtype="float64")
        sigma = tf.constant(self.process.sigma, dtype="float64")
        un_medio = tf.constant(0.5, dtype="float64")

        def pde(u_t, u_xx, u_x, u, x):
            res = u_t - un_medio * tf.pow(sigma, 2) * tf.pow(x, 2) * u_xx - rR * x * u_x + r * u
            return res + self.xva_data.source_function(u)
        
        self.pde = pde