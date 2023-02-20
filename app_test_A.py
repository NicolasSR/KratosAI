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
import networks.gradient_shallow_test_A as gradient_shallow_ae
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


if __name__ == "__main__":

    # Defining variable values:
    encoding_factor=1/4
    svd_reduction = 20
    custom_loss = tf.keras.losses.MeanSquaredError()
    normalize_input=True
    test_ratio = 0.2

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

    # Some configuration
    config = {
        "train_model":      True,
        "test_model":       True,
        "save_model":       True,
        "print_results":    True,
        "use_reduced":      False,
        "use_2d_layered":   False,
    }

    ### Next section is commented because the HDF5 App for Kratos crashes the script.
    #   Because of this, we cannot get the predictions for the residuals at each timestep

    # Create a fake Analysis stage to calculate the predicted residuals
    with open("ProjectParameters_fom.json", 'r') as parameter_file:
        parameters = KMP.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KMP.Model()
    fake_simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    fake_simulation.Initialize()
    fake_simulation.InitializeSolutionStep()

    ###

    # Select the network to use
    kratos_network = gradient_shallow_ae.GradientShallow()
    
    # Enable this line if you want to print the snapshot in npy format.
    # kratos_io.print_npy_snapshot(S, True)

    S, R, F = prepare_input(data_inputs_files, residuals_files, pointloads_files)
    print('Shape S: ', S.shape)
    print('Shape R: ', R.shape)
    print('Shape F: ', F.shape)


    print("====== Residuals analysis phase ======")
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
        SReduced = S

    kratos_network.calculate_data_limits(SReduced)

    S2 = SReduced.copy()[:,1:]
    SReduced=SReduced[:,:-1]
    R=R[:-1,:]
    F=F[:-1,:]

    S_train_T, S_test_T, S2_train, S2_test, R_train, R_test, F_train, F_test = train_test_split(SReduced.T, S2.T, R, F, test_size=test_ratio, shuffle=True)

    S_train = S_train_T.T
    S_test = S_test_T.T

    S_train_orig=S_train.copy()
    S_test_orig=S_test.copy()

    num_decoded_var=SReduced.shape[0]
    num_encoding_var=int(num_decoded_var*encoding_factor)

    # Load the model or train a new one. TODO: Fix custom_loss not being saved correctly
    
    print('======= Instantiating new autoencoder =======')
    autoencoder = kratos_network.define_network(S_train, custom_loss, num_encoding_var)
    autoencoder.fake_simulation = fake_simulation # Attach the fake as  # Uncomment when able to use HDF5 App in Kratos
    autoencoder.U = U # Attach the svd for the projection
    autoencoder.w = 0.5   # Weight for the custom loss defined inside GradModel2
    autoencoder.residuals_mask = residuals_mask
    autoencoder.R_high_max = R_high_max
    autoencoder.R_high_min = R_high_min
    autoencoder.R_low_max = R_low_max
    autoencoder.R_low_min = R_low_min
    if not config["train_model"]:
        print('======= Loading saved weights =======')
        # with open("saved_models/model.json", "r") as model_file:
        #     json_model = model_file.read()
        # autoencoder = keras.models.model_from_json(json_model)
        autoencoder.load_weights('saved_models/model_weights.h5')

    # Prepare the data
    if normalize_input==True:
        SRedNorm_train = kratos_network.normalize_data(S_train)
        SRedNorm_test = kratos_network.normalize_data(S_test)
    else:
        SRedNorm_train = S_train
        SRedNorm_test = S_test

    autoencoder.data_min=kratos_network.data_min
    autoencoder.data_max=kratos_network.data_max



    if config["train_model"]:
        print('')
        print('=========== Starting training routine ============')
        history = kratos_network.train_network(autoencoder, SRedNorm_train, (R_train, S2_train), 100)
        with open("saved_models/history.json", "w") as history_file:
            json.dump(str(history.history), history_file)
        print("Model trained")

    # Dettach the fake as (To prevent problems saving the model)
    autoencoder.fake_simulation = None

    if config["save_model"] and config["train_model"]:
        json_model = autoencoder.to_json()
        with open("saved_models/model.json", "w") as model_file:
            model_file.write(json_model)
        autoencoder.save_weights('saved_models/model_weights.h5')
        print("Model saved")

    # Dettach the fake as (To prevent problems saving the model)
    autoencoder.fake_simulation = fake_simulation
    
    if config["test_model"]:
        print('=========== Starting test routine ============')
        result = kratos_network.test_network(autoencoder, SRedNorm_train, R_train)
        result = kratos_network.test_network(autoencoder, SRedNorm_test, R_test)
        print('Model tested')

    test_sample=(SRedNorm_test.T)[20]
    test_sample=np.array([test_sample.T])
    predicted_x = kratos_network.predict_snapshot(autoencoder, test_sample)
    print(test_sample)
    print(predicted_x)
    print(test_sample-predicted_x)

    exit()








    NSPredict1 = kratos_network.predict_snapshot(autoencoder, SRedNorm_test)   # This is u', or f(q), or f(g(u))
    if normalize_input==True:
        SPredict1  = kratos_network.denormalize_data(NSPredict1.T)
    else:
        SPredict1 = NSPredict1

    print(f"{kratos_network.data_min=}")
    print(f"{kratos_network.data_max=}")

    print("U shape:", U.shape)

    if config["use_reduced"]:
        SP1 = U@(SPredict1)
    elif config["use_2d_layered"]:
        SP1 = np.zeros((svd_reduction,ST.shape[1]))
        for j in range(ST.shape[1]):
            for i in range(26):
                SP1[i*2+0,j] = SPredict1[0,i,j]
                SP1[i*2+1,j] = SPredict1[1,i,j]
    else:
        SP1 = SPredict1

    print("S  Shape:",   S.shape)
    print("SP Shape:", SP1.shape)

    print(f"NNM norm error : {np.linalg.norm(SP1-ST)/np.linalg.norm(ST)}")
    print(f"NNM norm error*: {np.linalg.norm(SP1[5]-ST[5])/np.linalg.norm(ST[5])}")

    print(f"SVD norm error : {np.linalg.norm(SPri-S)/np.linalg.norm(S)}")
    print(f"SVD norm error*: {np.linalg.norm(SPri[5]-S[5])/np.linalg.norm(S[5])}")

    print("Trying results calling model directly:")
    print("nsc shape:", SReduced.shape)

    fh = 0
    TI = SReduced[:,fh:fh+1]

    print("TI shape:", TI.shape)

    TP = kratos_network.predict_snapshot(autoencoder, TI)
    TP = TP[0].T

    print("TI norm error", np.linalg.norm((TP+1)-(TI+1))/np.linalg.norm(TI+1))

    # print(SP1.T[1])

    # With Kratos enabled this prints the predicted results in mdpa format for GiD
    if config["print_results"]:
        current_model = KMP.Model()
        model_part = current_model.CreateModelPart("main_model_part")

        kratos_io.create_out_mdpa(model_part, "beam_nonlinear_cantileaver_fom_coarse")
        kratos_io.print_results_to_gid(model_part, S.T, SP1.T)
