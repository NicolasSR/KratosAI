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

from networks.conv2d_ae_factory import  Conv2D_AE_Factory
from networks.dense_ae_factory import Dense_AE_Factory

tf.keras.backend.set_floatx('float64')

class NN_Trainer():
    def __init__(self,ae_config):
        self.ae_config=ae_config

    def setup_output_directory(self):
        self.case_path=self.ae_config["models_path"]+self.ae_config["name"]+"/"
        os.makedirs(os.path.dirname(self.case_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.case_path+"best/"), exist_ok=True)
        os.makedirs(os.path.dirname(self.case_path+"last/"), exist_ok=True)
    
    def prepare_input_augmented(self, dataset_path, use_force, cropped_ids):
        S_flat_orig=np.load(dataset_path+'FOM.npy')
        S_flat_orig_train=np.load(dataset_path+'S_augm_train.npy')
        S_flat_orig_test=np.load(dataset_path+'S_finetune_test.npy')
        if use_force is None or use_force == True:
            R_train=np.load(dataset_path+'R_augm_train.npy')
            R_test=np.load(dataset_path+'R_finetune_test.npy')
        else:
            print('Not using Force')
            R_train=np.load(dataset_path+'R_augm_noF_train.npy')
            R_test=np.load(dataset_path+'R_finetune_noF_test.npy')
        F_train=np.load(dataset_path+'F_augm_train.npy')[:,0,:]
        F_test=np.load(dataset_path+'F_finetune_test.npy')[:,0,:]

        S_flat_orig=np.delete(S_flat_orig, cropped_ids, axis=1)
        S_flat_orig_train=np.delete(S_flat_orig_train, cropped_ids, axis=1)
        S_flat_orig_test=np.delete(S_flat_orig_test, cropped_ids, axis=1)

        return S_flat_orig, S_flat_orig_train, S_flat_orig_test, R_train, R_test, F_train, F_test

    def prepare_input_finetune(self, dataset_path, use_force, cropped_ids):
        S_flat_orig=np.load(dataset_path+'FOM.npy')[:,4:]
        S_flat_orig_train=np.load(dataset_path+'S_finetune_noF_train.npy')[:,4:]
        S_flat_orig_test=np.load(dataset_path+'S_finetune_test.npy')[:,4:]

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
    
    def network_factory_selector(self, nn_type):
        if 'conv2d' in nn_type:
            return Conv2D_AE_Factory()
        elif 'dense' in nn_type:
            return Dense_AE_Factory()
        else:
            print('No valid network type was selected')
            return None
    
    def execute_training(self):

        self.setup_output_directory()

        # Select the network to use
        network_factory = self.network_factory_selector(self.ae_config["nn_type"])

        # Select the type of preprocessimg (normalisation). This also decides if cropping the snapshot is needed
        data_normalizer=network_factory.normalizer_selector(self.ae_config["normalization_strategy"])

        # Create a fake Analysis stage to calculate the predicted residuals
        kratos_simulation = KratosSimulator(self.ae_config, data_normalizer.needs_truncation)

        # Get input data
        if self.ae_config["augmented"]:
            S_flat_orig, S_flat_orig_train, S_flat_orig_test, R_train, R_test, F_train, F_test = self.prepare_input_augmented(self.ae_config['dataset_path'], self.ae_config["use_force"], kratos_simulation.get_cropped_dof_ids())
        else:
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

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        autoencoder, encoder, decoder = network_factory.define_network(S, self.ae_config)

        if not self.ae_config["finetune_from"] is None:
            print('======= Loading saved weights =======')
            autoencoder.load_weights(self.ae_config["finetune_from"]+'model_weights.h5')

        autoencoder.set_config_values(self.ae_config, data_normalizer, kratos_simulation)

        print('======= Saving AE Config =======')
        with open(self.case_path+"ae_config.npy", "wb") as ae_config_file:
            np.save(ae_config_file, self.ae_config)
        with open(self.case_path+"ae_config.json", "w") as ae_config_json_file:
            json.dump(self.ae_config, ae_config_json_file)

        print(self.ae_config)

        print('=========== Starting training routine ============')
        history = network_factory.train_network(autoencoder, S_train, (S_flat_orig_train,R_train,F_train), S_test, (R_test,F_test), self.ae_config)

        print('=========== Saving weights and history ============')
        autoencoder.save_weights(self.case_path+"model_weights.h5")
        with open(self.case_path+"history.json", "w") as history_file:
            json.dump(history.history, history_file)

        print(self.ae_config)

        # Dettach the fake sim (To prevent problems saving the model)
        autoencoder.kratos_simulation = None
        
    