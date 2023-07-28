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

class Q_Distances_test():

    def __init__(self, working_path, model_path, sim_path, best, test_large=False):
        self.working_path=working_path
        self.model_path=working_path+model_path
        self.sim_path=working_path+sim_path
        if best=='x':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_x_')
            self.best_name_part='_bestx_'
        elif best=='r':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_r_')
            self.best_name_part='_bestr_'
        elif best is None:
            self.model_weights_path=self.model_path
            self.model_weights_filename='model_weights.h5'
            self.best_name_part=''
        else:
            print('Value for --best argument is not recognized. Terminating')
            exit()

        with open(self.model_path+"ae_config.npy", "rb") as ae_config_file:
            self.ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()
        print(self.ae_config)
        self.dataset_path=working_path+self.ae_config['dataset_path']

        self.test_large=test_large
        if self.test_large:
            self.name_complement='_test_large_'
        else:
            self.name_complement = ''

    def get_last_best_filename(self, model_weights_path, prefix):
        matching_files = [file for file in os.listdir(model_weights_path) if file.startswith(prefix)]
        highest_filename = sorted(matching_files, key=lambda x: int(x[len(prefix):][:-len('.h5')]))[-1]
        return highest_filename
    
    def network_factory_selector(self, nn_type):
        if 'conv2d' in nn_type:
            return Conv2D_AE_Factory()
        elif 'dense' in nn_type:
            return Dense_AE_Factory()
        else:
            print('No valid network type was selected')
            return None
    
    def execute_evaluation(self):

        # Select the network to use
        network_factory = self.network_factory_selector(self.ae_config["nn_type"])

        # Select the type of preprocessimg (normalisation). This also decides if cropping the snapshot is needed
        self.data_normalizer=network_factory.normalizer_selector(self.working_path, self.ae_config)

        # Create a fake Analysis stage to calculate the predicted residuals
        self.residual_scale_factor=np.load(self.working_path+self.ae_config['dataset_path']+'residual_scale_factor.npy')
        self.kratos_simulation = KratosSimulator(self.working_path, self.ae_config, self.residual_scale_factor)
        crop_mat_tf, crop_mat_scp = self.kratos_simulation.get_crop_matrix()

        sim_snapshots=np.load(self.sim_path)

        S_flat_orig_FOM=np.load(self.dataset_path+'FOM.npy')
        print('Shape S_flat_orig_FOM: ', S_flat_orig_FOM.shape)

        self.data_normalizer.configure_normalization_data(S_flat_orig_FOM, crop_mat_tf, crop_mat_scp)

        S = self.data_normalizer.process_raw_to_input_format(S_flat_orig_FOM)

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        self.autoencoder, encoder, decoder = network_factory.define_network(S, self.ae_config)
        self.autoencoder.load_weights(self.model_weights_path+self.model_weights_filename)
        self.autoencoder.set_config_values_eval(self.data_normalizer)


        sim_snapshots_norm = self.data_normalizer.process_raw_to_input_format(sim_snapshots)
        q_snapshot = encoder(sim_snapshots_norm, training=False)
        norm_dq = np.linalg.norm(q_snapshot[1:]-q_snapshot[:-1], axis=1)
        print(norm_dq.shape)

        plt.plot(norm_dq)
        plt.semilogy()
        plt.xlabel('step')
        plt.ylabel('||dq||')
        plt.show()

        
if __name__ == "__main__":

    working_path = ''
    model_path = 'Cont_Dense_extended_SOnly_w0_lay40_LRe5_svd_white_nostand_2000ep'

    training_routine=Q_Distances_test(working_path, 'saved_models_newexample/'+model_path+'/', 'Kratos_results/LSPG_LR5_1000ep_fine/EqualForces/FOM.npy', 'x', test_large=False)
    training_routine.execute_evaluation()