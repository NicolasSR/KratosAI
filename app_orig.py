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

import logging
tf.get_logger().setLevel(logging.ERROR)

from keras import layers
from itertools import repeat

import kratos_io
import clustering
import networks.gradient_shallow as gradient_shallow_ae
import utils.check_gradient as check_gradients
import matplotlib.pyplot as plt

import KratosMultiphysics as KMP

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from fom_analysis import CreateRomAnalysisInstance
from utils.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

def print_gpu_info():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

tf.keras.backend.set_floatx('float64')

if __name__ == "__main__":
    
    # List of files to read from
    data_inputs = [
        "training/bases/result_80000.h5",
        "training/bases/result_90000.h5",
        # "training/bases/result_100000.h5",
        "training/bases/result_110000.h5",
        "training/bases/result_120000.h5",
    ]

    residuals = [
        "training/residuals/result_80000.npy",
        "training/residuals/result_90000.npy",
        # "training/residuals/result_100000.npy",
        "training/residuals/result_110000.npy",
        "training/residuals/result_120000.npy",
    ]

    pointloads = [
        "training/pointloads/result_80000.npy",
        "training/pointloads/result_90000.npy",
        # "training/pointloads/result_100000.npy",
        "training/pointloads/result_110000.npy",
        "training/pointloads/result_120000.npy",
    ]

    # List of variables to run be used (VELOCITY, PRESSURE, etc...)
    ST, S = kratos_io.build_snapshot_grid(
        data_inputs, 
        [
            "DISPLACEMENT",
        ]
    )

    R = None
    for r in residuals:
        a = np.load(r) / 3e9
        if R is None:
            print("None")
            R = a
        else:
            R = np.concatenate((R, a), axis=0)

    F = None
    for f in pointloads:
        a = np.load(f)
        if F is None:
            print("None")
            F = a
        else:
            F = np.concatenate((F, a), axis=0)

    # Some configuration
    config = {
        "load_model":       False,
        "train_model":      True,
        "save_model":       True,
        "print_results":    True,
        "use_reduced":      False,
        "use_2d_layered":   False,
    }

    # Create a fake Analysis stage to calculate the predicted residuals
    with open("ProjectParameters_fom.json", 'r') as parameter_file:
        parameters = KMP.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KMP.Model()
    fake_simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    fake_simulation.Initialize()
    fake_simulation.InitializeSolutionStep()

    # Select the netwrok to use
    kratos_network = gradient_shallow_ae.GradientShallow()
    svd_reduction = 20
    
    # Enable this line if you want to print the snapshot in npy format.
    # kratos_io.print_npy_snapshot(S, True)

    print("S ", S.shape)
    print("ST", ST.shape)

    print("=== Calculating Randomized Singular Value Decomposition ===")
    with contextlib.redirect_stdout(None):
        # Note: we redirect the output here because there is no option to reduce the log level of this method.
        #U,_,_,error = RandomizedSingularValueDecomposition().Calculate(S,1e-16)
        U,_,_ = np.linalg.svd(S, full_matrices=True, compute_uv=True, hermitian=False)
        K = U[:,0:2]
        U = U[:,0:svd_reduction]

        SPri = K @ K.T @ S

    print("U", U.shape)

    # Select the reduced snapshot or the full input
    if config["use_reduced"]:
        SReduced = U.T @ ST
    elif config["use_2d_layered"]:
        SReduced = np.zeros((2,36,ST.shape[1]))
        for j in range(ST.shape[1]):
            for i in range(26):
                SReduced[0,i,j] = ST[i*2+0,j]
                SReduced[1,i,j] = ST[i*2+1,j]
    else:
        SReduced = ST

    print("SReduced", SReduced.shape)

    data_rows = SReduced.shape[0]
    data_cols = SReduced.shape[1]

    kratos_network.calculate_data_limits(SReduced)

    # Set the properties for the clusters
    num_clusters=1                      # Number of different bases chosen
    num_cluster_col=2                   # If I use num_cluster_col = num_variables result should be exact.
    num_encoding_var=num_cluster_col

    # Load the model or train a new one. TODO: Fix custom_loss not being saved correctly
    if config["load_model"]:
        autoencoder = tf.keras.models.load_model(kratos_network.model_name, custom_objects={
            "custom_loss":custom_loss,
            "set_m_grad":gradient_shallow_ae.GradModel2.set_m_grad
        })
        kratos_network.encoder_model = tf.keras.models.load_model(kratos_network.model_name+"_encoder", custom_objects={
            "custom_loss":custom_loss,
            "set_m_grad":gradient_shallow_ae.GradModel2.set_m_grad
        })
        kratos_network.decoder_model = tf.keras.models.load_model(kratos_network.model_name+"_decoder", custom_objects={
            "custom_loss":custom_loss,
            "set_m_grad":gradient_shallow_ae.GradModel2.set_m_grad
        })
    else:
        autoencoder, autoencoder_err = kratos_network.define_network(SReduced, None, num_encoding_var)
        autoencoder.fake_simulation = fake_simulation # Attach the fake as
        autoencoder.U = U # Attach the svd for the projection

    # Prepare the data
    SReducedNormalized = (SReduced - kratos_network.data_min) / (kratos_network.data_max - kratos_network.data_min)
    SReducedNormalized = SReduced

    print(np.min(SReducedNormalized))
    print(np.max(SReducedNormalized))

    # Obtain the reduced representations (that would be our G's)
    temp_size = num_cluster_col
    grad_size = num_cluster_col

    if config["train_model"]:
        autoencoder.w = 1
        autoencoder.save_weights('./checkpoints/autoencoder')
        kratos_network.train_network(autoencoder, SReduced, (R,F), len(data_inputs), 100, 5)
        # autoencoder.w = 0.5
        # kratos_network.train_network(autoencoder, SReduced, (R,F), len(data_inputs), 10, 100)
        print("Model Initialized")

    NSPredict1 = kratos_network.predict_snapshot(autoencoder, SReduced)        # This is u', or f(q), or f(g(u)) 
    SPredict1  = kratos_network.denormalize_data(NSPredict1.T)

    # Dettach the fake as (To prevent problems saving the model)
    autoencoder.fake_simulation = None

    if config["save_model"]:
        autoencoder.save(kratos_network.model_name)
    
    if config["save_model"]:
        kratos_network.encoder_model.save(kratos_network.model_name+"_encoder")
        kratos_network.decoder_model.save(kratos_network.model_name+"_decoder")

    print(f"{kratos_network.data_min=}")
    print(f"{kratos_network.data_max=}")

    print("U:", U.shape)

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
