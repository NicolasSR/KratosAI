import os
import sys

import numpy as np

import keras
import tensorflow as tf


class Conv2DSnaphotOnlyAEModel(keras.Model):

    def __init__(self,*args,**kwargs):
        super(Conv2DSnaphotOnlyAEModel,self).__init__(*args,**kwargs)
        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.kratos_simulation = None

    def set_config_values(self, ae_config, data_normalizer, kratos_simulation):
        self.data_normalizer=data_normalizer
        self.kratos_simulation=kratos_simulation

    def set_config_values_eval(self, data_normalizer):
        self.data_normalizer=data_normalizer

    # Mean square error of the data
    def diff_norm_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def train_step(self,data):
        x_true_batch, (x_orig_batch,r_orig_batch,f_true_batch) = data
        trainable_vars = self.trainable_variables

        batch_len=x_true_batch.shape[0]

        total_gradients = []
        total_loss_x = 0

        for sample_id in range(batch_len):
            x_true=np.expand_dims(x_true_batch[sample_id], axis=0)

            with tf.GradientTape(persistent=True) as tape_d:
                tape_d.watch(trainable_vars)
                x_pred = self(x_true, training=True)
                loss_x = self.diff_norm_loss(x_true, x_pred)

            total_loss_x+=loss_x

            # Compute gradients
            gradients_loss_x = tape_d.gradient(loss_x, trainable_vars)
        
            for i in range(len(gradients_loss_x)):
                if sample_id == 0:
                    total_gradients.append(gradients_loss_x[i])
                else:
                    total_gradients[i]+=gradients_loss_x[i]
        
        for i in range(len(total_gradients)):
            total_gradients[i]=total_gradients[i]/batch_len

        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        total_loss_x = total_loss_x/batch_len

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        return {"loss_x": self.loss_x_tracker.result()}

    def test_step(self, data):
        x_true_batch, (r_orig_batch,f_true_batch) = data

        batch_len=x_true_batch.shape[0]

        total_loss_x = 0

        for sample_id in range(batch_len):
            x_true=np.expand_dims(x_true_batch[sample_id], axis=0)

            x_pred = self(x_true, training=True)
            loss_x = self.diff_norm_loss(x_true, x_pred)
            total_loss_x+=loss_x
                
        total_loss_x = total_loss_x/batch_len

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        return {"loss_x": self.loss_x_tracker.result()}
    
    def predict_snapshot(self,snapshot):
        norm_snapshot=self.data_normalizer.normalize_data(snapshot)
        norm_2d_snapshot=self.data_normalizer.reorganize_into_channels(norm_snapshot)
        norm_2d_pred=self.predict(norm_2d_snapshot)
        norm_pred=self.data_normalizer.reorganize_into_original(norm_2d_pred)
        pred=self.data_normalizer.normalize_data(norm_pred)
        return pred

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        if self.w==0.0:
            return [self.loss_x_tracker]
        else:
            return [self.loss_x_tracker, self.loss_r_tracker, self.loss_orth_tracker]
