
import numpy as np
import tensorflow as tf

import keras
from keras import backend
from keras.utils import io_utils

class CustomLearningRateScheduler(keras.callbacks.Callback):
    """ Custom Learning rate scheduler.
    At the beginning of every epoch, this callback gets the updated learning
    rate value from `schedule` function provided at `__init__`, with the current
    epoch and current learning rate, and applies the updated learning rate on
    the optimizer.
    Args:
      schedule: a function that takes an epoch index (integer, indexed from 0)
          and current learning rate (float) as inputs and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages. """

    def __init__(self, lr_schedule, w_schedule, lam_schedule, verbose=0):
        super().__init__()
        self.lr_schedule = lr_schedule
        self.w_schedule = w_schedule
        self.lam_schedule = lam_schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(backend.get_value(self.model.optimizer.lr))
            w = float(self.model.w)
            lam = float(self.model.lam)
            lr = self.lr_schedule(epoch, lr)
            w = self.w_schedule(epoch, w)
            lam = self.lam_schedule(epoch, lam)
        except TypeError:  # Support for old API for backward compatibility
            lr, w = self.schedule(epoch)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
        self.model.w = w
        # self.model.w_tf = tf.constant(w, dtype=tf.float64)
        self.model.lam = lam
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: CustomLearningRateScheduler setting"
                f"learning rate to {lr}, w to {w} and lambda to {lam}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = backend.get_value(self.model.optimizer.lr)
        logs["w"] = self.model.w
        logs["lam"] = self.model.lam