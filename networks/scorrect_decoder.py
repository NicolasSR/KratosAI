import os
import sys

import numpy as np

import keras
import tensorflow as tf


class SnapshotCorrectDecoderModel(keras.Model):

    def __init__(self,*args,**kwargs):
        super(SnapshotCorrectDecoderModel,self).__init__(*args,**kwargs)
        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.loss_rel_x_tracker = keras.metrics.Mean(name="loss_rel_x")
        self.kratos_simulation = None
        self.w=0
        self.lam=0

        self.run_eagerly=False

    def set_config_values(self, ae_config, data_normalizer, kratos_simulation, residual_scale_factor):
        self.data_normalizer=data_normalizer
        self.kratos_simulation=kratos_simulation

    def set_config_values_eval(self, data_normalizer):
        self.data_normalizer=data_normalizer

    # Mean square error of the data
    def diff_norm_loss(self, y_true, y_pred):
        return tf.math.reduce_sum((y_true - y_pred) ** 2, axis=1)

    def train_step(self,data):
        q_true_batch, (s_norm_true_batch) = data
        trainable_vars = self.trainable_variables

        batch_len=q_true_batch.shape[0]

        total_gradients = []
        total_loss_x = 0
        total_loss_rel_x = 0

        for sample_id in range(batch_len):

            q_true=tf.expand_dims(q_true_batch[sample_id],axis=0)
            s_norm_true=tf.expand_dims(s_norm_true_batch[sample_id],axis=0)

            with tf.GradientTape(persistent=False) as tape_d:
                tape_d.watch(trainable_vars)
                s_norm_pred = self(q_true, training=True)
                loss_x = self.diff_norm_loss(s_norm_true, s_norm_pred)

            grad_loss = tape_d.gradient(loss_x, trainable_vars)

            total_loss_x+=loss_x/batch_len

            loss_rel_x = tf.norm(s_norm_true-s_norm_pred)/tf.norm(s_norm_true)
            total_loss_rel_x+=loss_rel_x/batch_len
        
            for i in range(len(grad_loss)):
                if sample_id == 0:
                    total_gradients.append(grad_loss[i]/batch_len)
                else:
                    total_gradients[i]+=grad_loss[i]/batch_len

        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_rel_x_tracker.update_state(total_loss_rel_x)
        return {"loss_x": self.loss_x_tracker.result(), "loss_rel_x": self.loss_rel_x_tracker.result()}

    def test_step(self, data):
        q_true_batch, (s_norm_true_batch) = data

        batch_len=q_true_batch.shape[0]

        total_loss_x = 0
        total_loss_rel_x = 0

        for sample_id in range(batch_len):
            q_true=tf.expand_dims(q_true_batch[sample_id],axis=0)
            s_norm_true=tf.expand_dims(s_norm_true_batch[sample_id],axis=0)

            s_norm_pred = self(q_true, training=False)
            loss_x = self.diff_norm_loss(s_norm_true, s_norm_pred)
            total_loss_x+=loss_x/batch_len

            loss_rel_x = tf.norm(s_norm_true-s_norm_pred)/tf.norm(s_norm_true)
            total_loss_rel_x+=loss_rel_x/batch_len

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_rel_x_tracker.update_state(total_loss_rel_x)
        return {"loss_x": self.loss_x_tracker.result(), "loss_rel_x": self.loss_rel_x_tracker.result()}
    
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
        return [self.loss_x_tracker,self.loss_rel_x_tracker]
