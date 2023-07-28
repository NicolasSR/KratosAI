import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from utils.kratos_simulation import KratosSimulator

from networks.conv2d_ae_factory import  Conv2D_AE_Factory
from networks.dense_ae_factory import Dense_AE_Factory
from utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error

tf.keras.backend.set_floatx('float64')

class POD_Evaluator():
        
    def get_r_matrices(self, S_pred_test, S_pred_train, R_true_test, R_true_train):
        print('Scale factor:', self.residual_scale_factor)
        R_true_scaled_test=R_true_test/self.residual_scale_factor
        R_true_scaled_train=R_true_train/self.residual_scale_factor

        R_pred_scaled_test=self.kratos_simulation.get_r_array(S_pred_test)
        R_pred_scaled_train=self.kratos_simulation.get_r_array(S_pred_train)

        print(R_true_scaled_test.shape)
        print(R_true_scaled_train.shape)
        print(R_pred_scaled_test.shape)
        print(R_pred_scaled_train.shape)

        return R_true_scaled_test, R_true_scaled_train, R_pred_scaled_test, R_pred_scaled_train
        
    def calculate_R_norm_error(self, R_true, R_pred):
        l2_error=mean_relative_l2_error(R_true,R_pred)
        forb_error=relative_forbenius_error(R_true,R_pred)
        print('Residual. Mean rel L2 error l:', l2_error)
        print('Residual. Rel Forb. error l:', forb_error)
        return [l2_error,forb_error]

    def calculate_X_norm_error(self, S_true, S_pred):
        l2_error=mean_relative_l2_error(S_true,S_pred)
        forb_error=relative_forbenius_error(S_true,S_pred)
        return [l2_error, forb_error]

    def prepare_input_finetune(self):
        S_flat_orig=np.load(self.dataset_path+'FOM.npy')

        S_flat_orig_train=np.load(self.dataset_path+'S_finetune_train.npy')
        S_flat_orig_test=np.load(self.dataset_path+'S_finetune_test.npy')

        R_train=np.load(self.dataset_path+'R_finetune_noF_train.npy')
        R_test=np.load(self.dataset_path+'R_finetune_noF_test.npy')

        return S_flat_orig, S_flat_orig_train, S_flat_orig_test, R_train, R_test

    
    def execute_evaluation(self):


        self.dataset_path = 'datasets_two_forces_dense_extended/'
        
        
        # Create a fake Analysis stage to calculate the predicted residuals
        ae_config={
            "dataset_path": self.dataset_path,
            "project_parameters_file": 'ProjectParameters_fom_workflow.json'
        }
        working_path=''
        self.residual_scale_factor=np.load(self.dataset_path+'residual_scale_factor.npy')
        self.kratos_simulation = KratosSimulator(working_path, ae_config, self.residual_scale_factor)

        S_FOM, S_FOM_train, S_FOM_test, R_FOM_train, R_FOM_test = self.prepare_input_finetune()
        print('Shape S_FOM: ', S_FOM.shape)

        try:
            self.phi=np.load(self.dataset_path+'svd_raw_phi.npy')
        except IOError:
            self.phi, _, _=np.linalg.svd(S_FOM.T)
            np.save(self.dataset_path+'svd_raw_phi.npy', self.phi)

        self.phi = self.phi[:,:2]

        S_reconstr_train = np.matmul(self.phi,np.matmul(self.phi.T,S_FOM_train.T)).T
        S_reconstr_test = np.matmul(self.phi,np.matmul(self.phi.T,S_FOM_test.T)).T

        x_norm_err_matrix=[]
        # Test errors
        x_norm_err_matrix.append(self.calculate_X_norm_error(S_FOM_test, S_reconstr_test))
        # Train errors
        x_norm_err_matrix.append(self.calculate_X_norm_error(S_FOM_train, S_reconstr_train))
        x_norm_err_matrix=np.array(x_norm_err_matrix)

        print(x_norm_err_matrix)

        R_true_scaled_test, R_true_scaled_train, R_pred_scaled_test, R_pred_scaled_train = self.get_r_matrices(S_FOM_test, S_FOM_train, S_reconstr_test, S_reconstr_train)

        r_norm_err_matrix=[]
        # Test errors
        r_norm_err_matrix.append(self.calculate_R_norm_error(R_true_scaled_test,R_pred_scaled_test))
        # Train errors
        r_norm_err_matrix.append(self.calculate_R_norm_error(R_true_scaled_train,R_pred_scaled_train))
        r_norm_err_matrix=np.array(r_norm_err_matrix)

        print(r_norm_err_matrix)

        
if __name__ == "__main__":

    nn_evaluator = POD_Evaluator()

    nn_evaluator.execute_evaluation()