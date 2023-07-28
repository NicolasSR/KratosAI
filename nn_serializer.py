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
from networks.linear_ae_factory import Linear_AE_Factory

tf.keras.backend.set_floatx('float64')

class NN_Serializer():

    def __init__(self, working_path, model_path, best):
        self.working_path=working_path
        self.model_path=working_path+model_path
        if best=='x':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_x_')
            self.best_name_part='_bestx'
        elif best=='r':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_r_')
            self.best_name_part='_bestr'
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

    def get_last_best_filename(self, model_weights_path, prefix):
        matching_files = [file for file in os.listdir(model_weights_path) if file.startswith(prefix)]
        highest_filename = sorted(matching_files, key=lambda x: int(x[len(prefix):][:-len('.h5')]))[-1]
        return highest_filename

    def prepare_input_finetune(self):
        S_flat_orig=np.load(self.dataset_path+'FOM.npy')
        return S_flat_orig
    
    def network_factory_selector(self, nn_type):
        if 'conv2d' in nn_type:
            return Conv2D_AE_Factory()
        elif 'dense' in nn_type:
            return Dense_AE_Factory()
        elif 'linear' in nn_type:
            return Linear_AE_Factory()
        else:
            print('No valid network type was selected')
            return None
    
    def execute_serialization(self):

        # Select the network to use
        network_factory = self.network_factory_selector(self.ae_config["nn_type"])

        # Select the type of preprocessimg (normalisation). This also decides if cropping the snapshot is needed
        self.data_normalizer=network_factory.normalizer_selector(self.working_path, self.ae_config)

        # Create a fake Analysis stage to calculate the predicted residuals
        self.residual_scale_factor=np.load(self.working_path+self.ae_config['dataset_path']+'residual_scale_factor.npy')
        self.kratos_simulation = KratosSimulator(self.working_path, self.ae_config, self.residual_scale_factor)
        crop_mat_tf, crop_mat_scp = self.kratos_simulation.get_crop_matrix()

        S_flat_orig = self.prepare_input_finetune()
        print('Shape S_flat_orig: ', S_flat_orig.shape)

        self.data_normalizer.configure_normalization_data(S_flat_orig, crop_mat_tf, crop_mat_scp)

        S = self.data_normalizer.process_raw_to_input_format(S_flat_orig)
        print('Shape S: ', S.shape)

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        autoencoder, encoder, decoder = network_factory.define_network(S, self.ae_config, keras_default=True)
        autoencoder.load_weights(self.model_weights_path+self.model_weights_filename)

        encoder.save(self.model_path+'encoder_model'+self.best_name_part)
        decoder.save(self.model_path+'decoder_model'+self.best_name_part)