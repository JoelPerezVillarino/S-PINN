import tensorflow as tf

from ..neural_network import regularizers
from .custom_layers import ScaledOutputLayer


class ScaledOutputFNN(tf.keras.Model):
    """ Fully conected neural network """

    def __init__(
            self,
            layer_sizes,
            activation,
            kernel_init,
            scaling_factor=1.,
            regularization=None,
            dropout_rate=0,
            name=None
    ):
        super(ScaledOutputFNN, self).__init__(name=name)
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate

        self._input_transform = None
        self._output_transform = None

        # Hidden layers
        self.denses = []
        for units in layer_sizes[1:-1]:
            self.denses.append(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer=kernel_init,
                    kernel_regularizer=self.regularizer,
                )
            )

            if dropout_rate > 0:
                self.denses.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        # Output layer
        self.denses.append(
            ScaledOutputLayer(layer_sizes[-1], scaling_factor)
        )

    def call(self, inputs, training=None, mask=None):
        y = inputs
        if self._input_transform is not None:
            y = self._input_transform(y)
        for f in self.denses[:-1]:
            y = f(y, training=training)
        y = self.denses[-1](y, training=training)
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
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate

        self._input_transform = None
        self._output_transform = None

        # Hidden layers
        self.denses = []
        for units in layer_sizes[1:-1]:
            self.denses.append(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer=kernel_init,
                    kernel_regularizer=self.regularizer,
                )
            )

            if dropout_rate > 0:
                self.denses.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        # Output layer
        self.denses.append(
            tf.keras.layers.Dense(
                layer_sizes[-1],
                activation="linear",
                kernel_initializer=kernel_init,
                kernel_regularizer=self.regularizer
            )
        )

    def call(self, inputs, training=None, mask=None):
        y = inputs
        if self._input_transform is not None:
            y = self._input_transform(y)
        for f in self.denses[:-1]:
            y = f(y, training=training)
        y = self.denses[-1](y, training=training)
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