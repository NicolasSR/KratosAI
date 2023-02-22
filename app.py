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

import keras
import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from keras import layers
from itertools import repeat

import kratos_io
import clustering
import networks.gradient_shallow_nico as gradient_shallow_ae
import utils.check_gradient as check_gradients
import matplotlib.pyplot as plt

import KratosMultiphysics as KMP

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from fom_analysis import CreateRomAnalysisInstance
from utils.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

from sklearn.model_selection import train_test_split

import pandas as pd

def print_gpu_info():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

tf.keras.backend.set_floatx('float64')

def prepare_input(data_inputs_files, residuals_files, pointloads_files):

    variables_list=['DISPLACEMENT'] # List of variables to run be used (VELOCITY, PRESSURE, etc...)
    S = kratos_io.build_snapshot_grid(data_inputs_files,variables_list) # Both ST and S are the same total S matrix

    R = None
    for r in residuals_files:
        a = np.load(r) / 3e9 # We scale the resiuals because they are too close to 0 originally?
        if R is None:
            R = a
        else:
            R = np.concatenate((R, a), axis=0)

    F = None
    for f in pointloads_files:
        a = np.load(f)
        if F is None:
            F = a
        else:
            F = np.concatenate((F, a), axis=0)

    return S, R, F

def InitializeKratosAnalysis():
    with open("ProjectParameters_fom.json", 'r') as parameter_file:
        parameters = KMP.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KMP.Model()
    fake_simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    fake_simulation.Initialize()
    fake_simulation.InitializeSolutionStep()

    return fake_simulation


def normalize_snapshots_data(SReduced, normalization_strategy):
    if normalization_strategy == 'per_feature':
        feat_means = []
        feat_stds = []
        print('Normalizing each feature in SReduced')
        S_df = pd.DataFrame(SReduced.T)
        for i in range(len(S_df.columns)):
            feat_means.append(S_df[i].mean())
            feat_stds.append(S_df[i].std())
            # S_df[i] = (S_df[i]-S_df[i].mean()) / (S_df[i].std()+0.00000001)
        autoencoder.set_normalization_data(normalization_strategy, (feat_means, feat_stds))
        # SNorm=S_df.to_numpy().T
    elif normalization_strategy == 'global':
        print('Applying global min-max normalization on S')
        data_min = np.min(SReduced)
        data_max = np.max(SReduced)
        autoencoder.set_normalization_data(normalization_strategy, (data_min, data_max))
    else:
        print('No normalization')
    SNorm = autoencoder.normalize_data(SReduced.T)
    return SNorm.T

def analyze_residuals(R):
    print("====== Residuals analysis phase ======")

    R_df = pd.DataFrame(R)
    print(R_df.head)
    print(R_df.describe())

    residuals_threshold = np.mean(np.abs(R))
    residuals_mask = np.mean(np.where(np.abs(R)>residuals_threshold, 1, 0), axis=0)

    trim_matrix_high = np.identity(len(R[0]))
    trim_matrix_low = np.identity(len(R[0]))
    for i in range(len(R[0])):
        trim_matrix_high[i,i] = residuals_mask[i]
        trim_matrix_low[i,i] = 1-residuals_mask[i]

    R_high = R@trim_matrix_high
    R_low = R@trim_matrix_low

    R_high_max = np.max(R_high)
    R_high_min = np.min(R_high)
    R_low_max = np.max(R_low)
    R_low_min = np.min(R_low)

    print(residuals_mask)
    print(R_high_max)
    print(R_high_min)
    print(R_low_max)
    print(R_low_min)

    return residuals_mask, R_high_max, R_high_min, R_low_max, R_low_min

def snapshot_reduction(S):

    print("=== Calculating Randomized Singular Value Decomposition ===")

    with contextlib.redirect_stdout(None):
        # Note: we redirect the output here because there is no option to reduce the log level of this method.
        #U,_,_,error = RandomizedSingularValueDecomposition().Calculate(S,1e-16)
        U,_,_ = np.linalg.svd(S, full_matrices=True, compute_uv=True, hermitian=False)
        K = U[:,0:2]
        U = U[:,0:svd_reduction]
        SPri = K @ K.T @ S # Why do we project S into the first two modes uf U?

    print("U shape: ", U.shape)

    # Select the reduced snapshot or the full input
    if config["use_reduced"]:                   # Project the original S matrix into the basis of modes in truncated U
        print('Using projection of S matrix into modes of U')
        SReduced = U.T @ ST
    elif config["use_2d_layered"]:              # Separate S in two layers (one per var)
        print("Separating variables into separate layers")
        SReduced = np.zeros((2,26,ST.shape[1]))
        for j in range(ST.shape[1]):
            for i in range(26):
                SReduced[0,i,j] = ST[i*2+0,j]
                SReduced[1,i,j] = ST[i*2+1,j]
    else:
        print('Keeping original S matrix')
        SReduced = S

    return SReduced, U


if __name__ == "__main__":

    # Defining variable values:
    svd_reduction = 20
    custom_loss = tf.keras.losses.MeanSquaredError()
    test_ratio = 0.1

    # Some configuration
    config = {
        "train_model":      True,
        "finetune":         False,
        "test_model":       True,
        "save_model":       True,
        "print_results":    True,
        "use_reduced":      False,
        "use_2d_layered":   False,
    }

    if not config["train_model"] or config["finetune"]:
        print('======= Loading saved ae config =======')
        with open("saved_models/ae_config.npy", "rb") as ae_config_file:
            ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()
    else: 
        ae_config = {
            "encoding_factor": 1/4,
            "normalization_strategy": 'none',   # OPTIONS: 'none', 'per_feature', 'global'
            "residual_loss_ratio": 1,           # 1 being only loss on x, 0 only residual loss
            "hidden_layers": (2,2),             # Number of hidden layers in encoder and decoder
            "use_batch_normalisation": True,    # Whether to add batch normalisation layers
            "dropout_rate": 0.0,                # If set to 0, the dropout layers won't be added
            "learning_rate": 0.001
        }

    # List of files to read from
    data_inputs_files = [
        "training/bases/result_80000.h5",
        "training/bases/result_90000.h5",
        # "training/bases/result_100000.h5",
        "training/bases/result_110000.h5",
        "training/bases/result_120000.h5",
    ]

    residuals_files = [
        "training/residuals/result_80000.npy",
        "training/residuals/result_90000.npy",
        # "training/residuals/result_100000.npy",
        "training/residuals/result_110000.npy",
        "training/residuals/result_120000.npy",
    ]

    pointloads_files = [
        "training/pointloads/result_80000.npy",
        "training/pointloads/result_90000.npy",
        # "training/pointloads/result_100000.npy",
        "training/pointloads/result_110000.npy",
        "training/pointloads/result_120000.npy",
    ]

    # Create a fake Analysis stage to calculate the predicted residuals
    fake_simulation = InitializeKratosAnalysis()

    # Select the network to use
    kratos_network = gradient_shallow_ae.GradientShallow()

    # Get input data
    S, R, F = prepare_input(data_inputs_files, residuals_files, pointloads_files)
    print('Shape S: ', S.shape)
    print('Shape R: ', R.shape)
    print('Shape F: ', F.shape)

    # Perfroming SVD on shapshot matrix
    SReduced, U = snapshot_reduction(S)

    # Load the autoencoder model
    print('======= Instantiating new autoencoder =======')
    autoencoder = kratos_network.define_network(SReduced, custom_loss, ae_config)
    autoencoder.fake_simulation = fake_simulation # Attach the fake sim


    # Normalize the snapshots according to the desired normalization mode
    SNorm = normalize_snapshots_data(SReduced, ae_config["normalization_strategy"])

    # Analyze the residuals and generate trim matrices and such
    residuals_mask, R_high_max, R_high_min, R_low_max, R_low_min = analyze_residuals(R)

    # Divide snapshots into train and test sets
    S_train_T, S_test_T, R_train, R_test, F_train, F_test = train_test_split(SNorm.T, R, F, test_size=test_ratio, shuffle=True)
    print('Shape S_train_T: ', S_train_T.shape)
    print('Shape R_train: ', R_train.shape)
    print('Shape F_train: ', F_train.shape)
    print('Shape S_test_T: ', S_test_T.shape)
    print('Shape R_test: ', R_test.shape)
    print('Shape F_test: ', F_test.shape)
    S_train = S_train_T.T
    S_test = S_test_T.T
    print('Shape S_train: ', S_train.shape)
    print('Shape S_test: ', S_test.shape)

    
    autoencoder.U = U # Attach the svd for the projection
    autoencoder.w = ae_config["residual_loss_ratio"]   # Weight for the custom loss defined inside GradModel2
    autoencoder.residuals_mask = residuals_mask
    autoencoder.R_high_max = R_high_max
    autoencoder.R_high_min = R_high_min
    autoencoder.R_low_max = R_low_max
    autoencoder.R_low_min = R_low_min
    if not config["train_model"] or config["finetune"]:
        print('======= Loading saved weights =======')
        autoencoder.load_weights('saved_models/model_weights.h5')

    if config["train_model"]:
        print('')
        print('=========== Starting training routine ============')
        history = kratos_network.train_network(autoencoder, S_train, R_train, ae_config["learning_rate"], 100)
        print("Model trained")

    # Dettach the fake as (To prevent problems saving the model)
    autoencoder.fake_simulation = None

    if config["save_model"] and config["train_model"]:
        json_model = autoencoder.to_json()
        with open("saved_models/model.json", "w") as model_file:
            model_file.write(json_model)
        autoencoder.save_weights('saved_models/model_weights.h5')
        with open("saved_models/history.json", "w") as history_file:
            json.dump(str(history.history), history_file)
        with open("saved_models/ae_config.npy", "wb") as ae_config_file:
            np.save(ae_config_file, ae_config)
        print("Model saved")

    # Dettach the fake (As to prevent problems saving the model)
    autoencoder.fake_simulation = fake_simulation
    
    if config["test_model"]:
        print('=========== Starting test routine ============')
        result = kratos_network.test_network(autoencoder, S_train, R_train)
        result = kratos_network.test_network(autoencoder, S_test, R_test)
        print('Model tested')

    test_sample=(S_test.T)[20]
    test_sample=np.array([test_sample.T])
    predicted_x = kratos_network.predict_snapshot(autoencoder, test_sample)
    print(test_sample)
    print(predicted_x)
    print(test_sample-predicted_x)

    NSPredict1 = kratos_network.predict_snapshot(autoencoder, S_test.T)   # This is u', or f(q), or f(g(u))
    SP1  = autoencoder.denormalize_data(NSPredict1)
    SP1 = SP1.T
    ST = autoencoder.denormalize_data(S_test.T)
    ST = ST.T

    print("ST  Shape:",   ST.shape)
    print("SP Shape:", SP1.shape)

    print(f"NNM norm error : {np.linalg.norm(SP1.T-ST.T)/np.linalg.norm(ST.T)}")
    print(f"NNM norm error*: {np.linalg.norm(SP1.T[5]-ST.T[5])/np.linalg.norm(ST.T[5])}")

    exit()

    # With Kratos enabled this prints the predicted results in mdpa format for GiD
    if config["print_results"]:
        current_model = KMP.Model()
        model_part = current_model.CreateModelPart("main_model_part")

        kratos_io.create_out_mdpa(model_part, "beam_nonlinear_cantileaver_fom_coarse")
        kratos_io.print_results_to_gid(model_part, S.T, SP1.T)
