from numpy import dtype
import tensorflow as tf


class ScaledOutputLayer(tf.keras.layers.Layer):
    
    def __init__(self, units, scaling_factor=1., name=None, **kwargs) -> None:
        super(ScaledOutputLayer, self).__init__(name=name)
        self.units = units
        self.scaling_factor = tf.Variable(
            initial_value=scaling_factor, dtype=tf.float64,
            name="scaling_factor"
            )
        super(ScaledOutputLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(ScaledOutputLayer).get_config()
        config.update(
            {
                "scaling_factor": self.scaling_factor,
                "units": self.units
            }
        )
        return config

    def build(self, input_shape):
        tf.print("Calling build method...")
        self.w = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            dtype=tf.float64
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,), initializer="zeros", trainable=True,
            dtype=tf.float64
        )
        
    
    def call(self, inputs, training=None, mask=None):
        return tf.matmul(inputs, self.scaling_factor * self.w) + self.b
