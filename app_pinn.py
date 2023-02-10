import os
import sys
import time
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

from itertools import repeat
from sklearn.preprocessing import normalize

import kratos_io
import networks.pinn as pinn_ae

import matplotlib.pyplot as plt

import KratosMultiphysics as KMP

from utils.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

def print_gpu_info():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

def custom_loss(y_true, y_pred):
    y_diff = abs(y_true-y_pred)
    # y_diff = y_diff ** 2

    return  y_diff

def custom_loss_pinn(y_true, y_pred, r_true, r_pred, w):
    y_diff = y_true-y_pred
    r_diff = r_true-r_pred
    
    return (w)*(y_diff ** 2) + (1-w)(r_diff ** 2)

if __name__ == "__main__":
    
    # List of files to read from
    data_inputs = [
        "hdf5_bases/result_80000.h5",
        "hdf5_bases/result_90000.h5",
        "hdf5_bases/result_100000.h5",
        "hdf5_bases/result_110000.h5",
        "hdf5_bases/result_120000.h5",
        # "hdf5_bases/result_105000.h5", # Not for training
        # "hdf5_bases/result_135000.h5", # Not for training
        # "hdf5_bases/result_150000.h5", # Not for training
    ]

    # List of variables to run be used (VELOCITY, PRESSURE, etc...)
    data, sres = kratos_io.build_snapshot_grid_pinn(
        data_inputs, 
        [
            "DISPLACEMENT",
        ]
    )

    # Data preprocess
    data_bounds = [
        [0.0, 100.0],
        [0.8, 1.2],
        [0.0, 2.0],
        [0.0, 0.1],
    ]

    normalize_using_bounds = True
 
    if normalize_using_bounds:
        for i in range(len(data)):
            if data_bounds[i][0] != data_bounds[i][1]:
                data[i] = (data[i] - data_bounds[i][0]) / (data_bounds[i][1] - data_bounds[i][0])
    else:
        for i in range(len(data)):
            if np.min(data[i]) != np.max(data[i]):
                data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))

    # Some configuration
    config = {
        "load_model":       True,
        "train_model":      False,
        "save_model":       False,
        "print_results":    True,
    }

    # Select the netwrok to use
    kratos_network = pinn_ae.Pinn()

    # Load the model or train a new one
    if config["load_model"]:
        autoencoder = tf.keras.models.load_model(kratos_network.model_name, custom_objects={
            "custom_loss":custom_loss,
            "set_m_grad":pinn_ae.PinnModel.set_m_grad
        })
        kratos_network.encoder_model = tf.keras.models.load_model(kratos_network.model_name+"_encoder", custom_objects={
            "custom_loss":custom_loss,
            "set_m_grad":pinn_ae.PinnModel.set_m_grad
        })
        kratos_network.decoder_model = tf.keras.models.load_model(kratos_network.model_name+"_decoder", custom_objects={
            "custom_loss":custom_loss,
            "set_m_grad":pinn_ae.PinnModel.set_m_grad
        })
    else:
        autoencoder, autoencoder_err = kratos_network.define_network(data, sres, custom_loss, 2)

    if config["train_model"]:
        kratos_network.train_network(autoencoder, data, sres, len(data_inputs), 2000)
        print("Model Initialized")

    # Predict the model
    beg_predict = time.perf_counter()
    pred = kratos_network.predict_snapshot(autoencoder, data)
    end_predict = time.perf_counter()

    print(f"Preditcting time: {end_predict-beg_predict}s for {100*5} time steps")
    print(f"-- Preditct time: {(end_predict-beg_predict)/(5)}s per simulation")

    if config["save_model"]:
        autoencoder.save(kratos_network.model_name)
        kratos_network.encoder_model.save(kratos_network.model_name+"_encoder")
        kratos_network.decoder_model.save(kratos_network.model_name+"_decoder")

    # With Kratos enabled this prints the predicted results in mdpa format for GiD
    if config["print_results"]:
        current_model = KMP.Model()
        model_part = current_model.CreateModelPart("main_model_part")

        kratos_io.create_out_mdpa(model_part, "beam_nonlinear_cantileaver_fom_coarse")
        kratos_io.print_results_to_gid_pinn(
            model_part, 
            sres, pred,
            num_batchs=1, 
            num_steps=101,
            print_batch=-1
        )
