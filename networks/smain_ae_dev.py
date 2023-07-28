import os
import sys

import numpy as np

import keras
import tensorflow as tf

import time

class SnaphotMainAEModel(keras.Model):

    def __init__(self,*args,**kwargs):
        super(SnaphotMainAEModel,self).__init__(*args,**kwargs)
        self.w=0
        self.w_tf=tf.constant(0, dtype=tf.float64)
        self.lam=0

        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.loss_r_tracker = keras.metrics.Mean(name="loss_r")

        self.kratos_simulation = None

        self.run_eagerly = False

        self.residual_scale_factor = None

        self.sample_gradient_sum_functions_list=None
        self.generate_gradient_sum_functions()

        self.total_gradients=[]
        for i in range(len(self.trainable_variables)):
            self.total_gradients.append(tf.Variable(tf.zeros_like(self.trainable_variables[i])))

    def set_config_values(self, ae_config, data_normalizer, kratos_simulation, residual_scale_factor):
        self.data_normalizer=data_normalizer
        self.kratos_simulation=kratos_simulation
        self.residual_scale_factor=tf.constant(residual_scale_factor)

    def set_config_values_eval(self, data_normalizer):
        self.data_normalizer=data_normalizer
    
    @tf.function
    def get_v_loss_x(self, x_true, x_orig):
        x_pred = self(x_true, training=True)
        x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)
        v_loss_x = x_orig - x_pred_denorm  # We get the loss on the error for the denormalised snapshot

        return v_loss_x, x_pred_denorm
    
    @tf.function
    def get_gradients(self, trainable_vars, x_true, v_loss_x, v_loss_r, w):

        v_loss = -2*(v_loss_x+w*v_loss_r)

        with tf.GradientTape(persistent=False) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)
            v_u_dotprod = tf.linalg.matmul(v_loss, x_pred_denorm, transpose_b=True)
        grad_loss=tape_d.gradient(v_u_dotprod, trainable_vars)

        return grad_loss

    def generate_gradient_sum_functions(self):
        @tf.function
        def gradient_sum_sample(total_gradients, gradients, batch_len):
            total_gradients.assign(total_gradients+gradients/batch_len)
        
        self.sample_gradient_sum_functions_list=[]
        for i in range(len(self.trainable_variables)):
            self.sample_gradient_sum_functions_list.append(gradient_sum_sample)

    def train_step(self,data):
        x_true_batch, (x_orig_batch,r_orig_batch) = data
        trainable_vars = self.trainable_variables

        batch_len=x_true_batch.shape[0]

        # tf.print(trainable_vars)

        for i in range(len(self.total_gradients)):
            self.total_gradients[i].assign(tf.zeros_like(self.total_gradients[i]))
        
        total_loss_x = 0
        total_loss_r = 0

        for sample_id in range(batch_len):
            
            x_true=tf.expand_dims(x_true_batch[sample_id],axis=0)
            x_orig=tf.expand_dims(x_orig_batch[sample_id],axis=0)
            b_true=tf.expand_dims(r_orig_batch[sample_id]/self.residual_scale_factor,axis=0)

            v_loss_x, x_pred_denorm = self.get_v_loss_x(x_true, x_orig)

            err_r, v_loss_r = self.kratos_simulation.get_v_loss_r(x_pred_denorm,b_true)
            
            loss_x = tf.linalg.matmul(v_loss_x,v_loss_x,transpose_b=True)
            loss_r = tf.linalg.matmul(err_r,err_r,transpose_b=True)

            total_loss_x+=loss_x/batch_len
            total_loss_r+=loss_r/batch_len

            grad_loss = self.get_gradients(trainable_vars, x_true, v_loss_x, v_loss_r, self.w_tf)

            for i in range(len(self.total_gradients)):
                self.sample_gradient_sum_functions_list[i](self.total_gradients[i], grad_loss[i], batch_len)

        self.optimizer.apply_gradients(zip(self.total_gradients, trainable_vars))

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_r_tracker.update_state(total_loss_r)

        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result()}

    def test_step(self, data):
        x_true_batch, (x_orig_batch, r_orig_batch) = data

        batch_len=x_true_batch.shape[0]

        total_loss_x = 0
        total_loss_r = 0

        for sample_id in range(batch_len):

            x_true=tf.expand_dims(x_true_batch[sample_id],axis=0)
            x_orig=tf.expand_dims(x_orig_batch[sample_id],axis=0)
            b_true=tf.expand_dims(r_orig_batch[sample_id]/self.residual_scale_factor,axis=0)

            x_pred = self(x_true, training=False)
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)

            b_pred = self.kratos_simulation.get_r(x_pred_denorm)

            err_x = x_orig - x_pred_denorm
            err_r = b_true - b_pred
            loss_x = tf.linalg.matmul(err_x,err_x,transpose_b=True)
            loss_r = tf.linalg.matmul(err_r,err_r,transpose_b=True)

            total_loss_x+=loss_x/batch_len
            total_loss_r+=loss_r/batch_len

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_r_tracker.update_state(total_loss_r)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result()}
    
    # def predict_snapshot(self,snapshot):
    #     norm_2d_snapshot=self.data_normalizer.process_raw_to_input_format(snapshot)
    #     norm_2d_pred=self.predict(norm_2d_snapshot)
    #     pred=self.data_normalizer.process_input_to_raw_format(norm_2d_pred)
    #     return pred

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        
        return [self.loss_x_tracker, self.loss_r_tracker]
