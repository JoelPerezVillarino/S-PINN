import tensorflow as tf

from ..optimizers.tfp_optimizer import lbfgs_minimize


__all__ = ["get", "is_external_optimizer"]


def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(optimizer, learning_rate=None, decay=None):
    """Retrieve a Keras Optimizer instance."""
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
        return optimizer

    if is_external_optimizer(optimizer):
        if learning_rate is not None or decay is not None:
            print(f"Warning: learning rate is ignored for {optimizer}.\n")
        return lbfgs_minimize

    if learning_rate is None:
        raise ValueError(f"No learning rate for {optimizer}")

    lr_schedule = _get_learning_rate(learning_rate, decay)
    if optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if optimizer == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=lr_schedule)
    if optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    raise NotImplementedError(f"{optimizer} to be implemented.\n")


def _get_learning_rate(lr, decay):
    if decay is None:
        return lr

    if decay[0] == "inverse time":
        return tf.keras.optimizers.schedules.InverseTimeDecay(lr, decay[1], decay[2])
    if decay[0] == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(lr, decay[1], alpha=decay[2])

    raise NotImplementedError(
        f"{decay[0]} learning rate decay to be implemented.\n"
    )
