import tensorflow as tf

class FNN(tf.keras.Model):
    """ Fully conected neural network """

    def __init__(
            self,
            layer_sizes,
            activation,
            kernel_init,
            regularization=None,
            dropout_rate=0,
            name=None
    ):
        super(FNN, self).__init__(name=name)
        self.dropout_rate = dropout_rate

        self._input_transform = None
        self._output_transform = None
        self.n_inputs = layer_sizes[0]

        # Hidden layers
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape = (self.n_inputs,)))
        for units in layer_sizes[1:-1]:
            self.model.add(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer=kernel_init,
                )
            )


        # Output layer
        self.model.add(
            tf.keras.layers.Dense(
                layer_sizes[-1],
                activation="linear",
                kernel_initializer=kernel_init,
            )
        )

    def call(self, inputs, training=None, mask=None):
        y = inputs
        # Input normalization
        if self._input_transform is not None:
            y = self._input_transform(y)
        
        # Model call
        y = self.model(y)
        
        # Output normalization
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

    def apply_feature_transform(self, transform):
        """Compute the features by applying a transform to the network inputs, i.e.
        features = transform(inputs). Then, outputs = network(features)"""
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(input, outputs)"""
        self._output_transform = transform
