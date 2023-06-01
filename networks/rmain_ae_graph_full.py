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

        self.gradient_calc_functions_list=None
        self.generate_gradient_calc_functions()

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
    def get_gradient(self, trainable_vars, x_true):
        with tf.GradientTape(persistent=False) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            loss_x = self.diff_norm_loss(x_true, x_pred)
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)

        grad_loss_x = tape_d.gradient(loss_x,trainable_vars)
        return grad_loss_x, loss_x, x_pred_denorm

    """ @tf.function
    def get_jacobians(self, trainable_vars, x_true, A_pred, err_r):

        with tf.GradientTape(persistent=False) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            # loss_x = self.diff_norm_loss(x_true, x_pred)
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)      

        jac_u=tape_d.jacobian(x_pred_denorm, trainable_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)

        jac_layer_0=tf.reshape(jac_u[0],(jac_u[0].shape[0], jac_u[0].shape[1], tf.math.reduce_prod(jac_u[0].shape[2:])))
        pre_grad_0=tf.linalg.matmul(A_pred,tf.squeeze(jac_layer_0,axis=0),a_is_sparse=True)
        grad_loss_r_0=tf.linalg.matmul(err_r,pre_grad_0)*(-2)
        grad_loss_r_0=tf.reshape(grad_loss_r_0, jac_u[0].shape[2:])
        total_gradients_contribution_0=grad_loss_r_0

        jac_layer_1=tf.reshape(jac_u[1],(jac_u[1].shape[0], jac_u[1].shape[1], tf.math.reduce_prod(jac_u[1].shape[2:])))
        pre_grad_1=tf.linalg.matmul(A_pred,tf.squeeze(jac_layer_1,axis=0),a_is_sparse=True)
        grad_loss_r_1=tf.linalg.matmul(err_r,pre_grad_1)*(-2)
        grad_loss_r_1=tf.reshape(grad_loss_r_1, jac_u[1].shape[2:])
        total_gradients_contribution_1=grad_loss_r_1

        jac_layer_2=tf.reshape(jac_u[2],(jac_u[2].shape[0], jac_u[2].shape[1], tf.math.reduce_prod(jac_u[2].shape[2:])))
        pre_grad_2=tf.linalg.matmul(A_pred,tf.squeeze(jac_layer_2,axis=0),a_is_sparse=True)
        grad_loss_r_2=tf.linalg.matmul(err_r,pre_grad_2)*(-2)
        grad_loss_r_2=tf.reshape(grad_loss_r_2, jac_u[2].shape[2:])
        total_gradients_contribution_2=grad_loss_r_2

        jac_layer_3=tf.reshape(jac_u[3],(jac_u[3].shape[0], jac_u[3].shape[1], tf.math.reduce_prod(jac_u[3].shape[2:])))
        pre_grad_3=tf.linalg.matmul(A_pred,tf.squeeze(jac_layer_3,axis=0),a_is_sparse=True)
        grad_loss_r_3=tf.linalg.matmul(err_r,pre_grad_3)*(-2)
        grad_loss_r_3=tf.reshape(grad_loss_r_3, jac_u[3].shape[2:])
        total_gradients_contribution_3=grad_loss_r_3

        jac_layer_4=tf.reshape(jac_u[4],(jac_u[4].shape[0], jac_u[4].shape[1], tf.math.reduce_prod(jac_u[4].shape[2:])))
        pre_grad_4=tf.linalg.matmul(A_pred,tf.squeeze(jac_layer_4,axis=0),a_is_sparse=True)
        grad_loss_r_4=tf.linalg.matmul(err_r,pre_grad_4)*(-2)
        grad_loss_r_4=tf.reshape(grad_loss_r_4, jac_u[4].shape[2:])
        total_gradients_contribution_4=grad_loss_r_4

        jac_layer_5=tf.reshape(jac_u[5],(jac_u[5].shape[0], jac_u[5].shape[1], tf.math.reduce_prod(jac_u[5].shape[2:])))
        pre_grad_5=tf.linalg.matmul(A_pred,tf.squeeze(jac_layer_5,axis=0),a_is_sparse=True)
        grad_loss_r_5=tf.linalg.matmul(err_r,pre_grad_5)*(-2)
        grad_loss_r_5=tf.reshape(grad_loss_r_5, jac_u[5].shape[2:])
        total_gradients_contribution_5=grad_loss_r_5

        return total_gradients_contribution_0,total_gradients_contribution_1,total_gradients_contribution_2,total_gradients_contribution_3,total_gradients_contribution_4,total_gradients_contribution_5
        # return grad_loss_x, loss_x, x_pred_denorm """

    @tf.function
    def get_jacobians(self, trainable_vars, x_true, weighted_A_pred):

        with tf.GradientTape(persistent=True) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            # loss_x = self.diff_norm_loss(x_true, x_pred)
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)      

        jac_u=tape_d.jacobian(x_pred_denorm, trainable_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)

        # print(jac_u_0.shape)
        jac_layer_0=tf.reshape(jac_u[0],(4804, 5360))
        grad_loss_r_0=tf.linalg.matmul(weighted_A_pred,jac_layer_0)
        grad_resh_0=tf.reshape(grad_loss_r_0, (67, 80))

        jac_layer_1=tf.reshape(jac_u[1],(4804, 6400))
        grad_loss_r_1=tf.linalg.matmul(weighted_A_pred,jac_layer_1)
        grad_resh_1=tf.reshape(grad_loss_r_1, (80, 80))

        jac_layer_2=tf.reshape(jac_u[2],(4804, 160))
        grad_loss_r_2=tf.linalg.matmul(weighted_A_pred,jac_layer_2)
        grad_resh_2=tf.reshape(grad_loss_r_2, (80, 2))

        jac_layer_3=tf.reshape(jac_u[3],(4804,160))
        grad_loss_r_3=tf.linalg.matmul(weighted_A_pred,jac_layer_3)
        grad_resh_3=tf.reshape(grad_loss_r_3, (2, 80))

        jac_layer_4=tf.reshape(jac_u[4],(4804, 6400))
        grad_loss_r_4=tf.linalg.matmul(weighted_A_pred,jac_layer_4)
        grad_resh_4=tf.reshape(grad_loss_r_4, (80, 80))

        jac_layer_5=tf.reshape(jac_u[5],(4804, 5360))
        grad_loss_r_5=tf.linalg.matmul(weighted_A_pred,jac_layer_5)
        grad_resh_5=tf.reshape(grad_loss_r_5, (80, 67))

        return grad_resh_0, grad_resh_1, grad_resh_2, grad_resh_3, grad_resh_4, grad_resh_5
        # return grad_loss_x, loss_x, x_pred_denorm
    
    @tf.function
    def get_jacobians_lam(self, trainable_vars, x_true, b_true):
        with tf.GradientTape(persistent=True) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            loss_x = self.diff_norm_loss(x_true, x_pred)
            x_pred_denorm = self.data_normalizer.process_input_to_raw_format_tf(x_pred)
            print(x_pred_denorm)
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
            total_gradients_layer+=grad_loss_r
        
        self.gradient_calc_functions_list=[]
        for i in range(len(self.trainable_variables)):
            self.gradient_calc_functions_list.append(gradient_calc)


    def train_step(self,data):
        print('')
        # time_start=time.time()
        x_true_batch, (x_orig_batch,r_orig_batch,f_true_batch) = data
        trainable_vars = self.trainable_variables        

        batch_len=x_true_batch.shape[0]

        total_gradients=[]
        total_gradient_shapes=[]
        for i in range(len(trainable_vars)):
            total_gradients.append(tf.zeros_like(trainable_vars[i]))
            total_gradient_shapes.append(trainable_vars[i].shape)
               
        total_loss_x = 0
        total_loss_r = 0
        total_loss_orth = 0

        for sample_id in range(batch_len):
            # print('First step')

            x_true=tf.constant(np.expand_dims(x_true_batch[sample_id], axis=0))
            r_orig=tf.constant(np.expand_dims(r_orig_batch[sample_id], axis=0))
            f_true=tf.constant(np.expand_dims(f_true_batch[sample_id], axis=0))

            # x_orig=np.expand_dims(x_orig_batch[sample_id], axis=0)
            # print(x_orig)
            # x_pred_norm = self.data_normalizer.process_raw_to_input_format(x_orig+1)
            # x_pred_norm_tf = self.data_normalizer.process_raw_to_input_format_tf(x_orig+1)
            # print(x_pred_norm)
            # print(x_pred_norm_tf)
            # x_pred_denorm = self.data_normalizer.process_input_to_raw_format(x_pred_norm)
            # x_pred_denorm_tf = self.data_normalizer.process_input_to_raw_format_tf(x_pred_norm)
            # print(x_pred_denorm)
            # print(x_pred_denorm_tf)
            # exit()

            b_true=r_orig/self.residual_scale_factor

            # _, b_orig = self.kratos_simulation.get_r(x_orig,f_true)

            # print(b_orig)
            # print(b_true)
            # print('')

            time_start_jacobs=time.time()

            # print('Second step')
            if self.lam == 0.0:
                time_tape_outside=time.time()
                # grad_loss_x, jac_u, loss_x, x_pred_denorm = self.get_jacobians(trainable_vars, x_true)
                grad_loss_x, loss_x, x_pred_denorm = self.get_gradient(trainable_vars, x_true)
                print(time.time()-time_tape_outside)
                loss_orth=0.0
                grad_loss_orth=grad_loss_x  # This is very ugly. The idea is that we just need this variable to have the shape of grad_loss_x.
                                            # The values will be multiplied by zero later
            else:
                grad_loss_x, jac_u, loss_x, x_pred_denorm, loss_orth, grad_loss_orth = self.get_jacobians_lam(trainable_vars, x_true, b_true)

            print('Duration GradientTape: ', time.time()-time_start_jacobs)

            # print('Third step')
            total_loss_x+=loss_x
            total_loss_orth+=loss_orth

            # print('Fourth step')
            time_start_getr=time.time()
            A_pred, b_pred = self.kratos_simulation.get_r(x_pred_denorm,f_true)
            print('Duration GetR: ', time.time()-time_start_getr)
            # print('Duration untilNow: ', time.time()-time_start)
            # time_start_makeCt=time.time()
            A_pred  = tf.constant(A_pred)
            b_pred  = tf.constant(b_pred)
            # print('Duration MakeCt: ', time.time()-time_start_makeCt)
            # print('Duration untilNow: ', time.time()-time_start)

            # print('Fifth step')
            # time_start_lossr=time.time()
            err_r = b_true-b_pred
            # err_r = tf.expand_dims(err_r,axis=0)
            loss_r = self.diff_norm_loss(b_true, b_pred)
            total_loss_r+=loss_r
            # print('Duration LossR: ', time.time()-time_start_lossr)
            # print('Duration untilNow: ', time.time()-time_start)
            
            weighted_A_pred=tf.linalg.matmul(err_r,A_pred,b_is_sparse=True)*(-2)
            print(weighted_A_pred.shape)

            print(trainable_vars[0].shape)
            print(trainable_vars[1].shape)
            print(trainable_vars[2].shape)
            print(trainable_vars[3].shape)
            print(trainable_vars[4].shape)
            print(trainable_vars[5].shape)

            time_start_getgrads=time.time()
            grad_resh_0, grad_resh_1, grad_resh_2, grad_resh_3, grad_resh_4, grad_resh_5=self.get_jacobians(trainable_vars, x_true, weighted_A_pred)
            print('Duration GetGrads: ', time.time()-time_start_getgrads)

            time_start_sumgrads=time.time()
            total_gradients[0]+=grad_resh_0
            total_gradients[1]+=grad_resh_1
            total_gradients[2]+=grad_resh_2
            total_gradients[3]+=grad_resh_3
            total_gradients[4]+=grad_resh_4
            total_gradients[5]+=grad_resh_5

            
            for i in range(len(total_gradients)):
                # self.gradient_calc_functions_list[i](jac_u[i], A_pred, err_r, total_gradients[i])
                total_gradients[i]+=+self.w*grad_loss_x[i]+self.lam*grad_loss_orth[i]
            print('Duration Sum: ', time.time()-time_start_sumgrads)
            # print('Duration untilNow: ', time.time()-time_start)

            # self.gradient_calc_0(jac_u[0], A_pred, err_r, total_gradients[0])
            # total_gradients[0]+=self.w*grad_loss_x[0]+self.lam*grad_loss_orth[0]
            # self.gradient_calc_1(jac_u[1], A_pred, err_r, total_gradients[1])
            # total_gradients[1]+=self.w*grad_loss_x[1]+self.lam*grad_loss_orth[1]
            # self.gradient_calc_2(jac_u[2], A_pred, err_r, total_gradients[2])
            # total_gradients[2]+=self.w*grad_loss_x[2]+self.lam*grad_loss_orth[2]
            # self.gradient_calc_3(jac_u[3], A_pred, err_r, total_gradients[3])
            # total_gradients[3]+=self.w*grad_loss_x[3]+self.lam*grad_loss_orth[3]
            # self.gradient_calc_4(jac_u[4], A_pred, err_r, total_gradients[4])
            # total_gradients[4]+=self.w*grad_loss_x[4]+self.lam*grad_loss_orth[4]
            # self.gradient_calc_5(jac_u[5], A_pred, err_r, total_gradients[5])
            # total_gradients[5]+=self.w*grad_loss_x[5]+self.lam*grad_loss_orth[5]


            # self.gradient_calc_loop(jac_u, A_pred, sample_id, err_r, grad_loss_x, grad_loss_orth, total_gradients)

            
        # print('Sixth step')
        # print('Getting gradients2')

        for i in range(len(total_gradients)):
            total_gradients[i]=total_gradients[i]/batch_len

        # print('Seventh step')
        # time_start_applygrads=time.time()
        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))
        # print('Duration ApplyGrads: ', time.time()-time_start_applygrads)
        # print('Duration untilNow: ', time.time()-time_start)

        # print('Eigth step')
        total_loss_x = total_loss_x/batch_len
        total_loss_orth = total_loss_orth/batch_len
        total_loss_r = total_loss_r/batch_len

        # Compute our own metrics
        self.loss_x_tracker.update_state(total_loss_x)
        self.loss_orth_tracker.update_state(total_loss_orth)
        self.loss_r_tracker.update_state(total_loss_r)
        # print('Duration Total: ', time.time()-time_start)
        return {"loss_x": self.loss_x_tracker.result(), "loss_r": self.loss_r_tracker.result(), "loss_orth":self.loss_orth_tracker.result()}

    def test_step(self, data):
        x_true_batch, (r_orig_batch,f_true_batch) = data
        # x_true, (r_orig,f_true) = data

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
