import os
import sys

import numpy as np

import keras
import tensorflow as tf


class SVDMainMAEAEModel(keras.Model):

    def __init__(self,*args,**kwargs):
        super(SVDMainMAEAEModel,self).__init__(*args,**kwargs)
        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.kratos_simulation = None
        self.w=0
        self.lam=0

        self.run_eagerly=False

    def set_config_values(self, ae_config, data_normalizer, kratos_simulation, residual_scale_factor):
        self.data_normalizer=data_normalizer
        self.kratos_simulation=kratos_simulation

    def set_config_values_eval(self, data_normalizer):
        self.data_normalizer=data_normalizer

    # Mean absolute error of the data
    def diff_norm_loss(self, y_true, y_pred):
        return tf.math.reduce_sum(np.abs(y_true - y_pred), axis=1)

    def train_step(self,data):
        x_true_batch, (x_orig_batch,r_orig_batch) = data
        trainable_vars = self.trainable_variables

        batch_len=x_true_batch.shape[0]

        total_gradients = []
        total_loss_x = 0

        for sample_id in range(batch_len):

            x_true=tf.expand_dims(x_true_batch[sample_id],axis=0)
            # x_orig=tf.expand_dims(x_orig_batch[sample_id],axis=0)

            with tf.GradientTape(persistent=False) as tape_d:
                tape_d.watch(trainable_vars)
                x_pred = self(x_true, training=True)
                # x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)
                loss_x = self.diff_norm_loss(x_true, x_pred)

            grad_loss = tape_d.gradient(loss_x, trainable_vars)

            total_loss_x+=loss_x/batch_len
        
            for i in range(len(grad_loss)):
                if sample_id == 0:
                    total_gradients.append(grad_loss[i]/batch_len)
                else:
                    total_gradients[i]+=grad_loss[i]/batch_len
        


        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        return {"loss_x": self.loss_x_tracker.result()}

    def test_step(self, data):
        x_true_batch, (x_orig_batch,r_orig_batch) = data

        batch_len=x_true_batch.shape[0]

        total_loss_x = 0

        for sample_id in range(batch_len):
            x_true=tf.expand_dims(x_true_batch[sample_id],axis=0)
            # x_orig=tf.expand_dims(x_orig_batch[sample_id],axis=0)

            x_pred = self(x_true, training=False)
            # x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)
            loss_x = self.diff_norm_loss(x_true, x_pred)
            total_loss_x+=loss_x/batch_len

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        return {"loss_x": self.loss_x_tracker.result()}
    
    # def predict_snapshot(self,snapshot):
    #     norm_snapshot=self.data_normalizer.normalize_data(snapshot)
    #     norm_2d_snapshot=self.data_normalizer.reorganize_into_channels(norm_snapshot)
    #     norm_2d_pred=self.predict(norm_2d_snapshot)
    #     norm_pred=self.data_normalizer.reorganize_into_original(norm_2d_pred)
    #     pred=self.data_normalizer.normalize_data(norm_pred)
    #     return pred

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_x_tracker]
