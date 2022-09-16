import tensorflow as tf

class xVAData(object):

    def __init__(self, lamB: float, lamC: float, rB: float, rC: float) -> None:
        self.lamB = tf.constant(lamB, dtype="float64")
        self.lamC = tf.constant(lamC, dtype="float64")
        self.rB = tf.constant(rB, dtype="float64")
        self.rC = tf.constant(rC, dtype="float64")

    def source_function(self, u: tf.constant) -> tf.constant:
        zeros = tf.zeros_like(u)
        val = (1. - self.rB) * self.lamB * tf.minimum(u, zeros) \
            + (1. - self.rB) * self.lamB * tf.maximum(u, zeros) \
            + (1. - self.rC) * self.lamC * tf.maximum(u, zeros)
        return val
        
    def adjustment_function(self, tau: tf.constant) -> tf.constant:
        cte = self.lamB * (1. - self.rB) + self.lamC * (1. - self.rC)
        val = tf.math.exp(- cte * tau)
        return val