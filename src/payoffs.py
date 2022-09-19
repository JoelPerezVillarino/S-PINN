import tensorflow as tf


class Payoff(object):
    
    def __init__(self, CP: str, strike: float) -> None:
        if CP == "c":
            self.CP = CP
        elif CP == "p":
            self.CP = CP
        else:
            raise ValueError("CP must be c for Call, p for Put!")

        self.strike = tf.constant(strike, dtype="float64")
    
    def __call__(self, *args, **kwds):
        pass


class VanillaPayoff(Payoff):

    def __init__(self, CP: str, strike: float) -> None:
        super().__init__(CP, strike)
    
    def __call__(self, x: tf.constant) -> tf.constant:
        if self.CP == "c":
            return tf.maximum(x - self.strike, tf.zeros_like(x))
        return tf.maximum(self.strike - x, tf.zeros_like(x))


class ArithmeticBasketPayoff(Payoff):

    def __init__(self, CP: str, strike: float) -> None:
        super().__init__(CP, strike)
        # weights
        self._w1 = tf.constant(0.5, dtype="float64")
        self._w2 = tf.constant(0.5, dtype="float64")
    
    def __call__(self, x1: float, x2: float) -> tf.constant:
        if self.CP == "c":
            return tf.maximum(self._w1 * x1 + self.w2 * x2 - self.strike, tf.zeros_like(x1))
        return tf.maximum(self.strike - self._w1 * x1 - self._w2 * x2, tf.zeros_like(x1))


class WorstOfPayoff(Payoff):

    def __init__(self, CP: str, strike: float) -> None:
        super().__init__(CP, strike)
    
    def __call__(self, x1: float, x2: float) -> tf.constant:
        if self.CP == "c":
            return tf.maximum(tf.minimum(x1, x2) - self.strike, tf.zeros_like(x1))
        return tf.maximum(self.strike - tf.minimum(x1, x2), tf.zeros_like(x1))