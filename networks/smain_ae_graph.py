import os
import sys

import numpy as np

import keras
import tensorflow as tf


class SnaphotMainAEModel(keras.Model):

    def __init__(self,*args,**kwargs):
        super(SnaphotMainAEModel,self).__init__(*args,**kwargs)
        self.w=0
        self.lam=0

        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.loss_r_tracker = keras.metrics.Mean(name="loss_r")
        self.loss_orth_tracker = keras.metrics.Mean(name="loss_orth")

        self.kratos_simulation = None

        self.run_eagerly = True
        self.residual_scale_factor = None

        self.gradient_calc_functions_list=None
        self.generate_gradient_calc_functions()

    def set_config_values(self, ae_config, data_normalizer, kratos_simulation, residual_scale_factor):
        self.data_normalizer=data_normalizer
        self.kratos_simulation=kratos_simulation
        self.residual_scale_factor=residual_scale_factor

    def set_config_values_eval(self, data_normalizer):
        self.data_normalizer=data_normalizer

    # Mean square error of the data
    def diff_norm_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2
    
    def norm_loss(self, y_pred):
        return (y_pred) ** 2

    @tf.function
    def get_jacobians(self, trainable_vars, x_true, b_true):
        with tf.GradientTape(persistent=True) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            loss_x = self.diff_norm_loss(x_true, x_pred)
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)

        grad_loss_x = tape_d.gradient(loss_x,trainable_vars)
        jac_u = tape_d.jacobian(x_pred_denorm, trainable_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)

        return grad_loss_x, jac_u, loss_x, x_pred_denorm
    
    @tf.function
    def get_jacobians_lam(self, trainable_vars, x_true, b_true):
        with tf.GradientTape(persistent=True) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            loss_x = self.diff_norm_loss(x_true, x_pred)
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)
            print(b_true.shape)
            print(trainable_vars[0].shape)
            loss_orth=self.norm_loss(tf.linalg.matmul(b_true,trainable_vars[0]))

        grad_loss_x = tape_d.gradient(loss_x,trainable_vars)
        grad_loss_orth = tape_d.gradient(loss_orth,trainable_vars)
        jac_u = tape_d.jacobian(x_pred_denorm, trainable_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)
        return grad_loss_x, jac_u, loss_x, x_pred_denorm, loss_orth, grad_loss_orth
    
    def generate_gradient_calc_functions(self):
        @tf.function
        def gradient_calc(jac_layer, A_pred, err_r, total_gradients_layer):
            jac_layer=tf.reshape(jac_layer,(jac_layer.shape[0], jac_layer.shape[1], tf.math.reduce_prod(total_gradients_layer.shape)))
            # jac_layer=tf.reshape(jac_layer,(jac_layer.shape[0], jac_layer.shape[1], total_gradients_layer.shape[0]*total_gradients_layer.shape[1]))
            pre_grad=tf.linalg.matmul(A_pred,tf.squeeze(jac_layer,axis=0),a_is_sparse=True)
            grad_loss_r=tf.linalg.matmul(err_r,pre_grad)*(-2)
            grad_loss_r=tf.reshape(grad_loss_r, total_gradients_layer.shape)
            total_gradients_layer+=self.w*grad_loss_r
        
        self.gradient_calc_functions_list=[]
        for i in range(len(self.trainable_variables)):
            self.gradient_calc_functions_list.append(gradient_calc)

    def train_step(self,data):
        x_true_batch, (x_orig_batch,r_orig_batch,f_true_batch) = data
        trainable_vars = self.trainable_variables

        batch_len=x_true_batch.shape[0]

        total_gradients = []
        for i in range(len(trainable_vars)):
            total_gradients.append(tf.zeros_like(trainable_vars[i]))

        total_loss_x = 0
        total_loss_r = 0
        total_loss_orth = 0

        for sample_id in range(batch_len):
            x_true=tf.constant(np.expand_dims(x_true_batch[sample_id], axis=0))
            r_orig=tf.constant(np.expand_dims(r_orig_batch[sample_id], axis=0))
            f_true=tf.constant(np.expand_dims(f_true_batch[sample_id], axis=0))
            
            b_true=r_orig/self.residual_scale_factor
            
            if self.lam == 0.0:
                grad_loss_x, jac_u, loss_x, x_pred_denorm = self.get_jacobians(trainable_vars, x_true, b_true)
                loss_orth=0.0
                grad_loss_orth=grad_loss_x  # This is very ugly. The idea is that we just need this variable to have the shape of grad_loss_x.
                                            # The values will be multiplied by zero later
            else:
                grad_loss_x, jac_u, loss_x, x_pred_denorm, loss_orth, grad_loss_orth = self.get_jacobians(trainable_vars, x_true, b_true)
            
            total_loss_x+=loss_x
            total_loss_orth+=loss_orth

            A_pred, b_pred = self.kratos_simulation.get_r(x_pred_denorm,f_true)
            A_pred  = tf.constant(A_pred)
            b_pred  = tf.constant(b_pred)

            err_r = b_true-b_pred
            err_r = tf.expand_dims(tf.constant(err_r),axis=0)
            loss_r = self.diff_norm_loss(b_true, b_pred)
            total_loss_r+=loss_r

            for i in range(len(jac_u)):
                self.gradient_calc_functions_list[i](jac_u[i], A_pred, err_r, total_gradients[i])
                total_gradients[i]+=grad_loss_x[i]+self.lam*grad_loss_orth[i]


        for i in range(len(total_gradients)):
            total_gradients[i]=total_gradients[i]/batch_len

        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        total_loss_x = total_loss_x/batch_len
        total_loss_orth = total_loss_orth/batch_len
        total_loss_r = total_loss_r/batch_len
        
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_orth_tracker.update_state(total_loss_orth)
        self.loss_r_tracker.update_state(total_loss_r)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result(), "loss_orth":self.loss_orth_tracker.result()}


    def test_step(self, data):
        x_true_batch, (r_orig_batch,f_true_batch) = data

        batch_len=x_true_batch.shape[0]

        total_loss_x = 0
        total_loss_r = 0
        total_loss_orth = 0

        for sample_id in range(batch_len):

            x_true=np.expand_dims(x_true_batch[sample_id], axis=0)
            r_orig=np.expand_dims(r_orig_batch[sample_id], axis=0)
            f_true=np.expand_dims(f_true_batch[sample_id], axis=0)

            b_true=r_orig/self.residual_scale_factor

            x_pred = self(x_true, training=False)
            loss_x = self.diff_norm_loss(x_true, x_pred)
            total_loss_x+=loss_x
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)

            _, b_pred = self.kratos_simulation.get_r(x_pred_denorm,f_true)
        
            loss_r = self.diff_norm_loss(b_true, b_pred)
            total_loss_r+=loss_r

            if self.lam == 0.0:
                loss_orth=0
            else:
                loss_orth=self.norm_loss(tf.linalg.matmul(b_true,self.trainable_variables[0]))
            total_loss_orth+=loss_orth
                
        total_loss_x = total_loss_x/batch_len
        total_loss_r = total_loss_r/batch_len
        total_loss_orth = total_loss_orth/batch_len

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_r_tracker.update_state(total_loss_r)
        self.loss_orth_tracker.update_state(total_loss_orth)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result(), "loss_orth":self.loss_orth_tracker.result()}
    
    def predict_snapshot(self,snapshot):
        norm_2d_snapshot=self.data_normalizer.process_raw_to_input_format(snapshot)
        norm_2d_pred=self.predict(norm_2d_snapshot)
        pred=self.data_normalizer.process_input_to_raw_format(norm_2d_pred)
        return pred

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_x_tracker, self.loss_r_tracker, self.loss_orth_tracker]
