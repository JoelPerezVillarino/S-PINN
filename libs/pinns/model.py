import numpy as np
import tensorflow as tf

from .optimizers import optimizers
from .optimizers.config import LBFGS_options
from . import metrics as metrics_module
from . import display


class Model(object):

    def __init__(self, data, net, ):
        self.data = data
        self.net = net

        self.opt_name = None
        self.opt = None
        self.metrics = None

        # methods initialized in compile
        self._outputs = None
        self.output_losses = None
        self.train_step = None

        self.loss_history = LossHistory()
        self.train_state = TrainState()

    def compile(self, optimizer, lr=None, decay=None, loss_weights=None, metrics=None):
        print("Compiling model...")
        self.opt_name = optimizer
        opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)

        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

        # Loading methods
        @tf.function
        def outputs(training, inputs):
            return self.net(inputs, training=training)

        def output_losses(training):
            losses = self.data.losses(self, training)
            if not isinstance(losses, list):
                losses = [losses]
            if self.net.regularizer is not None:
                losses += [tf.math.reduce_sum(self.net.losses)]
            losses = tf.convert_to_tensor(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= loss_weights
                self.loss_history.set_loss_weights(loss_weights)
            return losses
        
        def output_int_losses():
            losses = self.data.int_loss(self, False)
            if not isinstance(losses, list):
                losses = [losses]
            losses = tf.convert_to_tensor(losses)
            if loss_weights is not None:
                losses *= loss_weights[0]
            return losses
        
        def output_ic_losses():
            losses = self.data.ic_loss(self, False)
            if not isinstance(losses, list):
                losses = [losses]
            losses = tf.convert_to_tensor(losses)
            if loss_weights is not None:
                losses *= loss_weights[1]
            return losses
        
        def output_boundary_losses():
            losses = self.data.boundary_losses(self, False)
            if not isinstance(losses, list):
                losses = [losses]
            losses = tf.convert_to_tensor(losses)
            if loss_weights is not None:
                losses *= loss_weights[1:]
            return losses

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                losses = output_losses(True)
                total_loss = tf.math.reduce_sum(losses)
            grads = tape.gradient(total_loss, self.net.trainable_variables)
            opt.apply_gradients(zip(grads, self.net.trainable_variables))

        def train_step_tfp(previous_optimizer_result=None):
            def build_losses():
                losses = output_losses(True)
                return tf.math.reduce_sum(losses)
            return opt(self.net.trainable_variables, build_losses, previous_optimizer_result)

        # Saving methods
        self._outputs = outputs
        self.output_losses = output_losses
        self.train_step = (
            train_step
            if not optimizers.is_external_optimizer(self.opt_name)
            else train_step_tfp
        )
        self.output_int_losses = output_int_losses
        self.output_ic_losses = output_ic_losses
        self.output_boundary_losses = output_boundary_losses

    def train(self, epochs=None, display_every=100):
        print("Training model...\n")
        self.train_state.x_test = self.data.test[0]
        self.train_state.y_test = self.data.test[1]
        self._test()

        if optimizers.is_external_optimizer(self.opt_name):
            self._train_tensorflow_tfp()
        else:
            if epochs is None:
                raise ValueError(f"No epochs for {self.opt_name}.")
            self._train_sgd(epochs, display_every)

        print("-" * 82)
        display.training_display.summary(self.train_state)
        return self.loss_history, self.train_state

    def _train_sgd(self, epochs, display_every):
        for i in range(epochs):
            self.train_step()
            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i+1 == epochs:
                self._test()

    def _train_tensorflow_tfp(self):
        n_iter = 0
        while n_iter < LBFGS_options["maxiter"]:
            results = self.train_step()
            n_iter += results.num_iterations.numpy()
            self.train_state.epoch += results.num_iterations.numpy()
            self.train_state.step += results.num_iterations.numpy()
            self._test()

            if results.converged or results.failed:
                break

    def _test(self):
        self.train_state.loss_train = self.output_losses(False)
        self.train_state.y_pred = self.net(self.train_state.x_test)

        self.train_state.metrics_test = [
            m(self.train_state.y_test, self.train_state.y_pred)
            for m in self.metrics
        ]
        self.train_state.update_best()
        self.loss_history.append(
            self.train_state.step,
            self.train_state.loss_train,
            tf.math.reduce_sum(self.train_state.loss_train),
            self.train_state.metrics_test
        )

        display.training_display(self.train_state)

    def predict(self, x, operator=None):
        if isinstance(x, tuple):
            x = tuple(np.array(xi, dtype="float64") for xi in x)
        else:
            x = np.array(x, dtype="float64")

        if operator is None:
            y = self._outputs(False, x)
            return y

        @tf.function
        def op(inputs):
            y = self.net(inputs,)
            return operator(inputs, y)

        y = op(x)
        return y.numpy()
    
    def take_jacobian(self, key):
        if key == "int":
            func = self.output_int_losses
        elif key == "ic":
            func = self.output_ic_losses
        elif key == "bound":
            func = self.output_boundary_losses
        elif key == "all":
            func = self.output_losses
        else:
            raise ValueError("Wrong key!")
        with tf.GradientTape() as tape:
            l = func()
        dl = tape.jacobian(l, self.net.trainable_variables)
        del tape
        return dl
    
    def take_hessian(self, key):
        if key == "int":
            func = self.output_int_losses
        elif key == "ic":
            func = self.output_ic_losses
        elif key == "bound":
            func = self.output_boundary_losses
        elif key == "all":
            func = self.output_losses
        else:
            raise ValueError("Wrong key!")
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                l = func()
            dl = tape2.jacobian(l, self.net.trainable_variables)
        d2l = tape.jacobian(dl, self.net.trainable_variables)
        del tape2
        del tape
        return d2l


class TrainState(object):

    def __init__(self):
        self.epoch = 0
        self.step = 0

        self.x_test = None
        self.y_test = None
        self.y_pred = None

        self.loss_train = None
        self.metrics_test = None

        self.best_step = None
        self.best_loss_train = np.inf
        self.best_y = None
        self.best_metrics = None

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_y = self.y_pred
            self.best_metrics = self.metrics_test

    def _packed_data(self):
        def merge_values(values):
            if values is None:
                return None
            return np.hstack(values) if isinstance(values, (tuple, list)) else values

        x_test = merge_values(self.x_test)
        y_test = merge_values(self.y_test)
        y_pred = merge_values(self.y_pred)
        best_y_pred = merge_values(self.best_y)
        return x_test, y_test, y_pred, best_y_pred

    def save_best_state(self, fname):
        print("Saving training data to {} ...".format(fname))
        x_test, y_test, y_pred, best_y_pred = self._packed_data()
        test = np.hstack((x_test, y_test, y_pred, best_y_pred))
        np.savetxt(fname, test, header="x, y_true, y_pred, best_y_pred")


class LossHistory(object):

    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.total_loss = []
        self.metrics_test = []
        self.loss_weights = 1

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, total_loss, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        self.total_loss.append(total_loss)
        self.metrics_test.append(metrics_test)

    def save(self, fname):
        print("Saving loss history to {} ...".format(fname))
        loss = np.hstack(
            (
                np.array(self.steps)[:, None],
                np.array(self.loss_train),
                np.array(self.total_loss)[:, None],
                np.array(self.metrics_test),
            )
        )
        np.savetxt(fname, loss, header="step, loss_train, total_loss, metrics_test")
