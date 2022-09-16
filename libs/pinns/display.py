import sys

import numpy as np


def list_to_str(nums, precision=2):
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))


class TrainingDisplay:
    """Display training progress."""
    def __init__(self):
        self.len_train = None
        self.len_metric = None
        self.is_header_print = False

    def print_one(self, s1, s2, s3):
        print("{:{l1}s}{:{l2}s}{:{l3}s}".format(
                s1,
                s2,
                s3,
                l1=10,
                l2=self.len_train,
                l3=self.len_metric,
            )
        )
        sys.stdout.flush()

    def header(self):
        self.print_one("Step", "Train loss", "Test metric")
        self.is_header_print = True

    def __call__(self, train_state):
        if not self.is_header_print:
            self.len_train = len(train_state.loss_train) * 10 + 4
            self.len_metric = len(train_state.metrics_test) * 10 + 4
            self.header()
        self.print_one(
            str(train_state.step),
            list_to_str(train_state.loss_train.numpy()),
            list_to_str(train_state.metrics_test),
        )

    def summary(self, train_state):
        print("Best model at step {:d}:".format(train_state.best_step))
        print("  train loss: {:.2e}".format(train_state.best_loss_train))
        print("  test metric: {:s}".format(list_to_str(train_state.best_metrics)))
        print("")
        self.is_header_print = False


training_display = TrainingDisplay()
