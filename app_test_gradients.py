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

import networks.gradient_shallow_test_gradients as gradient_shallow_ae
import utils.check_gradient as check_gradients
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import pandas as pd

def print_gpu_info():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

tf.keras.backend.set_floatx('float64')

def prepare_input():

    variables_list=['DISPLACEMENT'] # List of variables to run be used (VELOCITY, PRESSURE, etc...)
    S = np.array([[1.0,2.0],[3.0,4.0]])

    A = np.array([[1.0, 1.0],[2.0, 2.0]])

    R = A @ S.T
    R = R.T

    print('S:', S)
    print('R:', R)
    print('A:', A)

    return S, R, A


if __name__ == "__main__":

    # Defining variable values:
    encoding_factor=1/4
    svd_reduction = 20
    custom_loss = tf.keras.losses.MeanSquaredError()
    normalize_input=False
    test_ratio = 0.2

    # Some configuration
    config = {
        "load_model":       False,
        "train_model":      True,
        "save_model":       True,
        "print_results":    True,
        "use_reduced":      False,
        "use_2d_layered":   False,
    }

    # Select the network to use
    kratos_network = gradient_shallow_ae.GradientShallow()

    S, R, A = prepare_input()
    print('Shape S: ', S.shape)
    print('Shape R: ', R.shape)
    print('Shape A: ', A.shape)
  
    autoencoder = kratos_network.define_network(S, custom_loss)

    if config["train_model"]:
        kratos_network.train_network(autoencoder, S, R, 1)
        print("Model Initialized")

    kratos_network.test_network(autoencoder, S_test, (R_test,F_test))
    
    NSPredict1 = kratos_network.predict_snapshot(autoencoder, SReducedNormalized)        # This is u', or f(q), or f(g(u))
    if normalize_input==True:
        SPredict1  = kratos_network.denormalize_data(NSPredict1.T)
    else:
        SPredict1 = NSPredict1
    

    # Dettach the fake as (To prevent problems saving the model)
    #autoencoder.fake_simulation = None

    if config["save_model"]:
        autoencoder.save(kratos_network.model_name)
        kratos_network.encoder_model.save(kratos_network.model_name+"_encoder")
        kratos_network.decoder_model.save(kratos_network.model_name+"_decoder")

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
