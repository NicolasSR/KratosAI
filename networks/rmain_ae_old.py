import os
import sys

import numpy as np

import keras
import tensorflow as tf

import time

class ResidualMainAEModel(keras.Model):

    def __init__(self,*args,**kwargs):
        super(ResidualMainAEModel,self).__init__(*args,**kwargs)
        self.w=0
        self.lam=0

        self.loss_x_tracker = keras.metrics.Mean(name="loss_x")
        self.loss_r_tracker = keras.metrics.Mean(name="loss_r")
        self.loss_orth_tracker = keras.metrics.Mean(name="loss_orth")

        self.kratos_simulation = None

        self.run_eagerly = True

        self.residual_scale_factor = None

        self.sample_gradient_sum_functions_list=None
        self.generate_gradient_sum_functions()

    def set_config_values(self, ae_config, data_normalizer, kratos_simulation, residual_scale_factor):
        self.data_normalizer=data_normalizer
        self.kratos_simulation=kratos_simulation
        self.residual_scale_factor=tf.constant(residual_scale_factor)

    def set_config_values_eval(self, data_normalizer):
        self.data_normalizer=data_normalizer

    # Mean square error of the data
    def diff_norm_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2
    
    def norm_loss(self, y_pred):
        return (y_pred) ** 2

    @tf.function
    def get_gradients_x(self, trainable_vars, x_true):

        with tf.GradientTape(persistent=False) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            loss_x=tf.math.pow(x_true-x_pred,2)     
        grad_loss_x = tape_d.gradient(loss_x,trainable_vars)
        x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)

        return grad_loss_x, loss_x, x_pred_denorm
    
    @tf.function
    def get_gradients_x_orth(self, trainable_vars, x_true, b_true):

        b_true_norm = self.data_normalizer.process_raw_to_input_format_tf(b_true)
        
        with tf.GradientTape(persistent=True) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            loss_x=tf.math.pow(x_true-x_pred,2)
            loss_orth=self.norm_loss(tf.linalg.matmul(b_true_norm,trainable_vars[0]))
        grad_loss_x = tape_d.gradient(loss_x,trainable_vars)
        grad_loss_orth = tape_d.gradient(loss_orth,trainable_vars, unconnected_gradients='zero')
        print(grad_loss_orth)
        x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)

        return grad_loss_x, loss_x, x_pred_denorm, loss_orth, grad_loss_orth

    @tf.function
    def get_gradients_r(self, trainable_vars, x_true, err_r, A_pred):

        err_r=tf.squeeze(err_r,axis=0)
        # v_vec_T=-1.0*tf.matmul(A_pred, err_r, adjoint_a=True, adjoint_b=True)
        v_vec_T=-1.0*tf.sparse.sparse_dense_matmul(A_pred, err_r, adjoint_a=True, adjoint_b=True)
        v_vec=tf.transpose(v_vec_T)

        with tf.GradientTape(persistent=False) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)
            v_u_dotprod = tf.linalg.matmul(v_vec, x_pred_denorm, transpose_b=True)
        grad_loss_r=tape_d.gradient(v_u_dotprod, trainable_vars)

        return grad_loss_r

    def generate_gradient_sum_functions(self):
        @tf.function
        def gradient_sum_sample(previous_gradients, x_gradients, r_gradients, orth_gradients, w, lam, batch_len):
            updated_gradients=previous_gradients+(r_gradients+w*x_gradients+lam*orth_gradients)/batch_len
            return updated_gradients
        
        self.sample_gradient_sum_functions_list=[]
        for i in range(len(self.trainable_variables)):
            self.sample_gradient_sum_functions_list.append(gradient_sum_sample)

    def convert_sparse_matrix_to_sparse_tensor(self, A_scipy):
        coo = A_scipy.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def train_step(self,data):
        x_true_batch, (x_orig_batch,r_orig_batch,f_true_batch) = data
        trainable_vars = self.trainable_variables

        batch_len=x_true_batch.shape[0]

        total_gradients=[]
        for i in range(len(trainable_vars)):
            total_gradients.append(tf.zeros_like(trainable_vars[i]))
        
        total_loss_x = 0
        total_loss_r = 0
        total_loss_orth = 0

        for sample_id in range(batch_len):
            
            x_true=tf.expand_dims(x_true_batch[sample_id], axis=0)
            r_orig=tf.expand_dims(r_orig_batch[sample_id], axis=0)
            f_true=tf.expand_dims(f_true_batch[sample_id], axis=0)

            b_true=r_orig/self.residual_scale_factor

            if self.lam == 0.0:
                grad_loss_x, loss_x, x_pred_denorm = self.get_gradients_x(trainable_vars, x_true)
                loss_orth=0.0
                grad_loss_orth=grad_loss_x  # This is very ugly. The idea is that we just need this variable to have the shape of grad_loss_x.
                                            # The values will be multiplied by zero later
            else:
                grad_loss_x, loss_x, x_pred_denorm, loss_orth, grad_loss_orth = self.get_gradients_x_orth(trainable_vars, x_true, b_true)
            
            total_loss_x+=loss_x/batch_len
            total_loss_orth+=loss_orth/batch_len

            A_pred_scipy, b_pred = self.kratos_simulation.get_r(x_pred_denorm,f_true)
            A_pred_tf = self.convert_sparse_matrix_to_sparse_tensor(A_pred_scipy)
            b_pred  = tf.constant(b_pred)
            
            err_r = b_true-b_pred
            err_r = tf.expand_dims(err_r,axis=0)
            loss_r = self.diff_norm_loss(b_true, b_pred)
            total_loss_r+=loss_r/batch_len

            # grad_loss_r = self.get_gradients_r(trainable_vars, x_true, err_r, A_pred_scipy)
            grad_loss_r = self.get_gradients_r(trainable_vars, x_true, err_r, A_pred_tf)

            for i in range(len(total_gradients)):
                total_gradients[i]=self.sample_gradient_sum_functions_list[i](total_gradients[i],grad_loss_x[i],grad_loss_r[i],grad_loss_orth[i],self.w,self.lam, batch_len)

        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        # Compute our own metrics
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
