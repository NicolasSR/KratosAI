import os
import sys
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import math

import contextlib

import h5py
import numpy as np
import scipy

import keras
import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from keras import layers
from itertools import repeat

import matplotlib.pyplot as plt

import KratosMultiphysics as KMP

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from fom_analysis import CreateRomAnalysisInstance

from sklearn.model_selection import train_test_split

import pandas as pd

from utils.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
from utils.normalizers import Conv2D_AE_Normalizer_ChannelRange, Conv2D_AE_Normalizer_FeatureStand
from utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error

import kratos_io
from networks.conv2d_residual_ae_experimental import  Conv2D_Residual_AE

def print_gpu_info():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

tf.keras.backend.set_floatx('float64')

def prepare_input(dataset_path):

    S_flat_orig=np.load(dataset_path+'FOM.npy')[:,4:]
    S_flat_orig_train=np.load(dataset_path+'S_train.npy')[:,4:]
    S_flat_orig_test=np.load(dataset_path+'S_test.npy')[:,4:]
    R_train=np.load(dataset_path+'R_train.npy')
    R_test=np.load(dataset_path+'R_test.npy')
    F_train=np.load(dataset_path+'F_train.npy')[:,0,:]
    F_test=np.load(dataset_path+'F_test.npy')[:,0,:]

    return S_flat_orig, S_flat_orig_train, S_flat_orig_test, R_train, R_test, F_train, F_test

def InitializeKratosAnalysis():
    with open("ProjectParameters_fom.json", 'r') as parameter_file:
        parameters = KMP.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KMP.Model()
    fake_simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    fake_simulation.Initialize()
    fake_simulation.InitializeSolutionStep()

    return fake_simulation


def normalize_snapshots_data(S, normalization_strategy):
    if normalization_strategy == 'per_feature':
        print('Normalizing each feature in S')
        S_flat=S.reshape((S.shape[0],48))
        print(S[0])
        print(S_flat[0])
        print(S.reshape((S.shape[0],2,2,12))[0])
        feat_means = []
        feat_stds = []
        S_df = pd.DataFrame(S_flat)
        for col in range(len(S_df.columns)):
            feat_means.append(S_df[col].mean())
            feat_stds.append(S_df[col].std())
        feat_means=np.reshape(feat_means,(2,2,12))
        feat_stds=np.reshape(feat_stds,(2,2,12))
        autoencoder.set_normalization_data(normalization_strategy, (feat_means, feat_stds))
    elif normalization_strategy == 'global':
        print('Global normalisation not implemented')
        exit()
        print('Applying global min-max normalization on S')
        data_min = np.min(S)
        data_max = np.max(S)
        autoencoder.set_normalization_data(normalization_strategy, (data_min, data_max))
    else:
        print('No normalization')
    SNorm = autoencoder.normalize_data(S)
    return SNorm

def normalizer_selector(normalization_strategy):
    if normalization_strategy == 'channel_range':
        return Conv2D_AE_Normalizer_ChannelRange()
    elif normalization_strategy == 'feature_stand':
        return Conv2D_AE_Normalizer_FeatureStand()
    else:
        print('Normalization strategy is not valid')
        return None


if __name__ == "__main__":

    # Defining variable values:
    custom_loss = tf.keras.losses.MeanSquaredError()

    # Some configuration
    config = {
        "test_model":       True,
        "save_model":       True
    }
    
    ae_config = {
        "name": 'r_as_regularizer_wbinary_experimental',
        "encoding_size": 1,
        "hidden_layers": ((16,(2,5),(1,2)),
                          (32,(2,5),(1,2))
                          ),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'channel_range',  # ['feature_stand','channel_range']
        "residual_loss_ratio": ('binary', 0.0, 0.1, 4), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "residual_norm_factor": ('const',1.0e7),
        # "activation_functtion": tf.keras.activations.linear, ['elu', ]
        "dataset_path": 'datasets_low/',
        "models_path": 'saved_models_conv2d_experimental/',
        "finetune_from": None,
        "residual_grad_normalisation": None # For now it is fixed to the identity
     }
    
    print(ae_config)

    case_path=ae_config["models_path"]+ae_config["name"]+"/"
    os.makedirs(os.path.dirname(case_path), exist_ok=True)
    os.makedirs(os.path.dirname(case_path+"best/"), exist_ok=True)
    os.makedirs(os.path.dirname(case_path+"last/"), exist_ok=True)

    # Create a fake Analysis stage to calculate the predicted residuals
    fake_simulation = InitializeKratosAnalysis()

    # Select the network to use
    kratos_network = Conv2D_Residual_AE()

    # Get input data
    S_flat_orig, S_flat_orig_train, S_flat_orig_test, R_train, R_test, F_train, F_test = prepare_input(ae_config['dataset_path'])
    print('Shape S_flat_orig: ', S_flat_orig.shape)
    print('Shape S_flat_orig_train:', S_flat_orig_train.shape)
    print('Shape S_flat_orig_test:', S_flat_orig_test.shape)
    print('Shape R_train: ', R_train.shape)
    print('Shape R_est: ', R_test.shape)
    print('Shape F_train: ', F_train.shape)
    print('Shape F_test: ', F_test.shape)

    data_normalizer=normalizer_selector(ae_config["normalization_strategy"])
    data_normalizer.configure_normalization_data(S_flat_orig)

    S_flat = data_normalizer.normalize_data(S_flat_orig)
    S_flat_train = data_normalizer.normalize_data(S_flat_orig_train)
    S_flat_test = data_normalizer.normalize_data(S_flat_orig_test)
    print('Shape S_flat: ', S_flat.shape)
    print('Shape S_flat_train:', S_flat_train.shape)
    print('Shape S_flat_test:', S_flat_test.shape)

    S = data_normalizer.reorganize_into_channels(S_flat)
    S_train = data_normalizer.reorganize_into_channels(S_flat_train)
    S_test = data_normalizer.reorganize_into_channels(S_flat_test)
    print('Shape S: ', S.shape)
    print('Shape S_train: ', S_train.shape)
    print('Shape S_test: ', S_test.shape)

    # Load the autoencoder model
    print('======= Instantiating new autoencoder =======')
    autoencoder, encoder, decoder = kratos_network.define_network(S, ae_config)
    autoencoder.fake_simulation = fake_simulation # Attach the fake sim

    print(autoencoder.trainable_variables[0])

    if not ae_config["finetune_from"] is None:
        print('======= Loading saved weights =======')
        autoencoder.load_weights(ae_config["finetune_from"]+'model_weights.h5')

    print(autoencoder.trainable_variables[0])

    autoencoder.set_config_values(ae_config, np.concatenate((R_train, R_test), axis=0), data_normalizer)

    # def calculate_X_norm_error(S_true, S_input):
    #     S_pred_norm=autoencoder(S_input).numpy()
    #     S_pred_flat_norm = data_normalizer.reorganize_into_original(S_pred_norm)
    #     S_pred_flat = data_normalizer.denormalize_data(S_pred_flat_norm)
    #     l2_error=mean_relative_l2_error(S_true,S_pred_flat)
    #     forb_error=relative_forbenius_error(S_true,S_pred_flat)

    #     print('X. Mean rel L2 error:', l2_error)
    #     print('X. Rel Forb. error:', forb_error)

    # print('Test errors')
    # calculate_X_norm_error(S_flat_orig_test, S_test)
    # print('Train errors')
    # calculate_X_norm_error(S_flat_orig_train, S_train)

    # exit()

    if config["save_model"]:
        with open(case_path+"ae_config.npy", "wb") as ae_config_file:
            np.save(ae_config_file, ae_config)
        with open(case_path+"ae_config.json", "w") as ae_config_json_file:
            json.dump(ae_config, ae_config_json_file)
        print("Model config saved")

    print('')
    print('=========== Starting training routine ============')
    history = kratos_network.train_network(autoencoder, S_train, (S_flat_orig_train,R_train,F_train), S_test, (R_test,F_test), ae_config)
    print("Model trained")

    if config["save_model"]:
        autoencoder.save_weights(case_path+"model_weights.h5")
        encoder.save_weights(case_path+"encoder_model_weights.h5")
        decoder.save_weights(case_path+"decoder_model_weights.h5")
        with open(case_path+"history.json", "w") as history_file:
            json.dump(history.history, history_file)
        print("Model saved")

    # Dettach the fake sim (To prevent problems saving the model)
    autoencoder.fake_simulation = None

    print('============= Outputting results ================')

    S_pred_test = autoencoder(S_test).numpy()
    S_pred_flat_test  = data_normalizer.reorganize_into_original(S_pred_test)
    S_pred_denorm_test  = data_normalizer.denormalize_data(S_pred_flat_test)


    S_pred_train = autoencoder(S_train).numpy()
    S_pred_flat_train  = data_normalizer.reorganize_into_original(S_pred_train)
    S_pred_denorm_train = data_normalizer.denormalize_data(S_pred_flat_train)

    print("S_pred_denorm_test  Shape:", S_pred_denorm_test.shape)
    print("S_pred_denorm_train Shape:", S_pred_denorm_train.shape)

    print(f"Mean relative l2 error, test data : {mean_relative_l2_error(S_flat_orig_test,S_pred_denorm_test)}")
    print(f"Mean relative l2 error, train data: {mean_relative_l2_error(S_flat_orig_train,S_pred_denorm_train)}")

    print(f"Relative Forbenius error, test data : {relative_forbenius_error(S_flat_orig_test,S_pred_denorm_test)}")
    print(f"Relative Forbenius error, train data: {relative_forbenius_error(S_flat_orig_train,S_pred_denorm_train)}")

    print(ae_config)

    """ # With Kratos enabled this prints the predicted results in mdpa format for GiD
    if config["print_results"]:
        current_model = KMP.Model()
        model_part = current_model.CreateModelPart("main_model_part")

        kratos_io.create_out_mdpa(model_part, "beam_nonlinear_cantileaver_fom_coarse")
        kratos_io.print_results_to_gid(model_part, S.T, SP1.T) """
