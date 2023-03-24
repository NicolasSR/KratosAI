import os
import sys
import logging
from collections import Counter

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import math

import contextlib

import h5py
import numpy as np
import scipy

import keras
import keras.backend as K
import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from keras import layers
from itertools import repeat

import matplotlib.pyplot as plt

import pandas as pd

import KratosMultiphysics as KMP

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from fom_analysis import CreateRomAnalysisInstance

from utils.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
from utils.normalizers import Conv2D_AE_Normalizer_ChannelRange
from utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error

import kratos_io
from networks.conv2d_residual_ae_proof import Conv2D_Residual_AE
from utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error, mean_l2_error, forbenius_error

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

def normalizer_selector(normalization_strategy):
    if normalization_strategy == 'channel_range':
        return Conv2D_AE_Normalizer_ChannelRange()
    else:
        print('Normalization strategy is not valid')
        return None

def InitializeKratosAnalysis():
    with open("ProjectParameters_fom.json", 'r') as parameter_file:
        parameters = KMP.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KMP.Model()
    fake_simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    fake_simulation.Initialize()
    fake_simulation.InitializeSolutionStep()

    return fake_simulation

if __name__ == "__main__":

    data_path='saved_models_conv2d_proof/'

    print('======= Loading saved ae config =======')
    with open(data_path+"ae_config.npy", "rb") as ae_config_file:
        ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()

    print(ae_config)

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

    F=np.concatenate((F_train, F_test), axis=0)

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

    autoencoder.gen_random_matrices(S_flat_orig_train.shape[1])

    autoencoder.set_config_values(ae_config, np.concatenate((R_train, R_test), axis=0), data_normalizer)

    print('======= Loading saved weights =======')
    autoencoder.load_weights(data_path+'model_weights.h5')
    encoder.load_weights(data_path+'encoder_model_weights.h5')

    R=autoencoder.get_r_array(S,F)
    R_train=autoencoder.get_r_array(S_train,F_train)
    R_test=autoencoder.get_r_array(S_test,F_test)
    print(R.shape)
    print(R_train.shape)
    print(R_test.shape)

    def calculate_R_norm_error(S_input, R_true, F_true):
        S_pred_norm=autoencoder(S_input).numpy()
        R_pred=autoencoder.get_r_array(S_pred_norm, F_true)
        l2_error=mean_relative_l2_error(R_true,R_pred)
        forb_error=relative_forbenius_error(R_true,R_pred)
        print('Residual. Mean rel L2 error l:', l2_error)
        print('Residual. Rel Forb. error l:', forb_error)

    def calculate_X_norm_error(S_input):
        S_pred_norm=autoencoder(S_input).numpy()
        S_pred_flat_norm = data_normalizer.reorganize_into_original(S_pred_norm)
        S_pred_flat = data_normalizer.denormalize_data(S_pred_flat_norm)
        l2_error=mean_relative_l2_error(S_input,S_pred_flat)
        forb_error=relative_forbenius_error(S_input,S_pred_flat)

        print('X. Mean rel L2 error:', l2_error)
        print('X. Rel Forb. error:', forb_error)

    # print('Test errors')
    # calculate_X_norm_error(S_test)
    # print('Train errors')
    # calculate_X_norm_error(S_train)

    def plot_embeddings():
        embeddings=tf.transpose(encoder(S))

        for i in range(embeddings.shape[0]):
            plt.scatter(F[:,1],embeddings[i])
        plt.show()

    plot_embeddings()

    def print_x_and_r_vecs(sample_id):
        x_orig=np.expand_dims(S_flat_orig[sample_id], axis=0)
        x_norm=np.expand_dims(S_flat[sample_id], axis=0)
        x_pred_norm=autoencoder(np.expand_dims(S[sample_id], axis=0)).numpy()
        x_pred_flat_norm=data_normalizer.reorganize_into_original(x_pred_norm)
        x_pred_flat=data_normalizer.denormalize_data(x_pred_flat_norm)
        r_orig=np.expand_dims(R[sample_id], axis=0)
        r_pred=autoencoder.get_r_array(x_pred_norm,np.expand_dims(F[sample_id], axis=0))
        print('x_orig:', x_orig)
        print('x_pred:', x_pred_flat)
        print('x_norm:', x_norm)
        print('x_pred_norm:', x_pred_norm)
        print('r_orig:', r_orig)
        print('r_pred:', r_pred)

        plt.plot(x_orig[0], '.')
        plt.plot(x_pred_flat[0], '.')
        plt.show()

        plt.plot(x_norm[0], '.')
        plt.plot(x_pred_flat_norm[0], '.')
        plt.show()

        plt.plot(r_orig[0])
        plt.plot(r_pred[0])
        plt.show()

    sample_id = 0
    print(sample_id)
    print(F[sample_id])
    print_x_and_r_vecs(sample_id)

    sample_id = S.shape[0]//3
    print(sample_id)
    print(F[sample_id])
    print_x_and_r_vecs(sample_id)

    sample_id = S.shape[0]//2
    print(sample_id)
    print(F[sample_id])
    print_x_and_r_vecs(sample_id)

    sample_id = S.shape[0]*2//3
    print(sample_id)
    print(F[sample_id])
    print_x_and_r_vecs(sample_id)

    sample_id = S_train.shape[0]-1
    print(sample_id)
    print(F[sample_id])
    print_x_and_r_vecs(sample_id)

    sample_id = S_train.shape[0]+100
    print(sample_id)
    print(F[sample_id])
    print_x_and_r_vecs(sample_id)

    def draw_x_error_image():
        S_pred = autoencoder(S).numpy()
        S_pred_flat=data_normalizer.reorganize_into_original(S_pred)
        err_df = pd.DataFrame(S_pred_flat-S_flat)
        err_df['force']=np.abs(F[:,1])
        err_df=err_df.sort_values('force')
        print(err_df)
        print(err_df.describe())
        fig, (ax1) = plt.subplots(ncols=1)
        image=err_df.iloc[:,:-1].to_numpy()
        im1 = ax1.imshow(image, extent=[1,S_flat.shape[1],np.max(np.abs(F[:,1])),np.min(np.abs(F[:,1]))], interpolation='none')
        ax1.set_aspect(1/2e5)
        cbar1 = plt.colorbar(im1)
        plt.show()

    draw_x_error_image()

    print('Test errors')
    calculate_R_norm_error(S_test,R_test,F_test)
    print('Train errors')
    calculate_R_norm_error(S_train,R_train,F_train)

    def output_to_GID():
        # With Kratos enabled this prints the predicted results in mdpa format for GiD
        current_model = KMP.Model()
        model_part = current_model.CreateModelPart("main_model_part")

        kratos_io.create_out_mdpa(model_part, "beam_nonlinear_cantileaver_fom_coarse_result")
        S_pred_norm = autoencoder(SNorm.T)
        S_pred=autoencoder.denormalize_data(S_pred_norm)
        S_pred_complete=np.concatenate([np.zeros((S_pred.shape[0],4),np.float64),S_pred], axis=1)
        S_complete=np.concatenate([np.zeros((S.shape[1],4),np.float64),S.T], axis=1)
        kratos_io.print_results_to_gid(model_part, S_complete, S_pred_complete)

    def plot_node_momentum():
        displ=S.T[1:,-2:]-S.T[:-1,-2:]
        displ=np.append(displ, np.expand_dims(displ[-1,:], axis=0), axis=0)
        displ_norm=np.linalg.norm(displ, axis=1)
        displ_norm=(displ_norm-np.min(displ_norm))/(np.max(displ_norm)-np.min(displ_norm))
        plt.plot(displ_norm)
        displ_norm=np.power([np.e for i in range(len(displ_norm))], displ_norm)
        # plt.plot(displ_norm)
        plt.show()
        weights=scipy.special.softmax(displ_norm)
        print(np.sum(weights))
        plt.plot(weights)
        plt.show()

        rng = np.random.default_rng(2023)
        kept_ids=rng.choice(np.arange(S.shape[1]), size=int(S.shape[1]*0.5), replace=False, p=weights)

        plt.hist(kept_ids)
        plt.show()

    # plot_node_momentum()