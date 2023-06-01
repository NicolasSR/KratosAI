import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from utils.kratos_simulation import KratosSimulator

from utils.normalizers import AE_Normalizer_SVD, AE_Normalizer_ChannelScale

import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

class ResidualConvergenceTest():
    def __init__(self,ae_config):
        self.ae_config=ae_config

    def prepare_input_finetune(self, dataset_path, use_force, cropped_ids):
        S_flat_orig=np.load(dataset_path+'FOM.npy')
        S_flat_orig_train=np.load(dataset_path+'S_finetune_train.npy')
        S_flat_orig_test=np.load(dataset_path+'S_finetune_test.npy')

        if use_force is None or use_force == True:
            R_train=np.load(dataset_path+'R_finetune_train.npy')
            R_test=np.load(dataset_path+'R_finetune_test.npy')
        else:
            print('Not using Force')
            R_train=np.load(dataset_path+'R_finetune_noF_train.npy')
            R_test=np.load(dataset_path+'R_finetune_noF_test.npy')
        F_train=np.load(dataset_path+'F_finetune_train.npy')[:,0,:]
        F_test=np.load(dataset_path+'F_finetune_test.npy')[:,0,:]

        S_flat_orig=np.delete(S_flat_orig, cropped_ids, axis=1)
        S_flat_orig_train=np.delete(S_flat_orig_train, cropped_ids, axis=1)
        S_flat_orig_test=np.delete(S_flat_orig_test, cropped_ids, axis=1)

        return S_flat_orig, S_flat_orig_train, S_flat_orig_test, R_train, R_test, F_train, F_test
    
    def execute_test(self):

        # Select the type of preprocessimg (normalisation). This also decides if cropping the snapshot is needed
        # data_normalizer=AE_Normalizer_SVD(self.ae_config["dataset_path"])
        data_normalizer=AE_Normalizer_ChannelScale()


        # Create a fake Analysis stage to calculate the predicted residuals
        residual_scale_factor=np.load(self.ae_config['dataset_path']+'residual_scale_factor.npy')
        kratos_simulation = KratosSimulator(self.ae_config, data_normalizer.needs_cropping, residual_scale_factor)

        
        S_flat_orig, S_flat_orig_train, S_flat_orig_test, R_train, R_test, F_train, F_test = self.prepare_input_finetune(self.ae_config['dataset_path'], self.ae_config["use_force"], kratos_simulation.get_cropped_dof_ids())
        print('Shape S_flat_orig: ', S_flat_orig.shape)
        print('Shape S_flat_orig_train:', S_flat_orig_train.shape)
        print('Shape S_flat_orig_test:', S_flat_orig_test.shape)
        print('Shape R_train: ', R_train.shape)
        print('Shape R_est: ', R_test.shape)
        print('Shape F_train: ', F_train.shape)
        print('Shape F_test: ', F_test.shape)

        data_normalizer.configure_normalization_data(S_flat_orig)

        S = data_normalizer.process_raw_to_input_format(S_flat_orig)
        S_train = data_normalizer.process_raw_to_input_format(S_flat_orig_train)
        S_test = data_normalizer.process_raw_to_input_format(S_flat_orig_test)
        print('Shape S: ', S.shape)
        print('Shape S_train: ', S_train.shape)
        print('Shape S_test: ', S_test.shape)

        sample_orig=np.expand_dims(S_flat_orig[495], axis=0)
        sample_norm=np.expand_dims(S[495], axis=0)
        
        A_true, b_true = kratos_simulation.get_r(sample_orig, None)

        eps_vec = np.logspace(1, 12, 1000)/1e13

        print('Shape sample: ', sample_norm.shape)

        v=np.random.rand(1,sample_orig.shape[1])
        v=v/np.linalg.norm(v)
        print(v)
        print(np.linalg.norm(v))

        err_vec=[]
        noise_norm_vec=[]
        for eps in eps_vec:

            ev=v*eps
            sample_norm_noise=sample_norm+ev

            sample_noise = data_normalizer.process_input_to_raw_format(sample_norm_noise)
            _, b_app = kratos_simulation.get_r(sample_noise, None)

            noise=sample_noise-sample_orig
            noise_norm_vec.append(np.linalg.norm(noise))

            first_order=A_true@noise.T
            L=b_app-b_true-first_order.T
            err_vec.append(np.linalg.norm(L))

        square=np.power(eps_vec,2)

        plt.plot(noise_norm_vec, square, "--", label="square")
        plt.plot(noise_norm_vec, eps_vec, "--", label="linear")
        plt.plot(noise_norm_vec, err_vec, label="error")
        # plt.plot(eps_vec, err_h, label="error_h")
        # plt.plot(eps_vec, err_l, label="error_l")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(loc="upper left")
        plt.show()