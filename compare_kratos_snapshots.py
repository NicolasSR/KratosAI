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

import matplotlib.pyplot as plt

import pandas as pd
from utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error

from sklearn.model_selection import train_test_split


def prepare_input(dataset_path, rom_file):

    S_fom=np.load(dataset_path+'FOM.npy')[:,4:]
    S_rom=np.load(dataset_path+rom_file)[:,4:]

    return S_fom, S_rom

def calculate_X_norm_error(S_fom, S_rom):
    l2_error=mean_relative_l2_error(S_fom,S_rom)
    forb_error=relative_forbenius_error(S_fom,S_rom)
    print('X. Mean rel L2 error:', l2_error)
    print('X. Rel Forb. error:', forb_error)

def draw_x_error_abs_image(S_fom, S_rom):
        fig, (ax1) = plt.subplots(ncols=1)
        image=S_rom-S_fom
        # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', cmap='jet')
        im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=-0.015, vmax=0.03, cmap='jet')
        ax1.set_aspect(1/2e5)
        cbar1 = plt.colorbar(im1)
        plt.xlabel('index')
        plt.ylabel('force')
        plt.title('Displacement Abs Error')
        plt.show()

def draw_x_error_rel_image(S_fom, S_rom):
        fig, (ax1) = plt.subplots(ncols=1)
        image=(S_rom-S_fom)/S_fom
        # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', cmap='jet')
        im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=-0.1, vmax=0.1, cmap='jet')
        ax1.set_aspect(1/2e5)
        cbar1 = plt.colorbar(im1)
        plt.xlabel('index')
        plt.ylabel('force')
        plt.title('Displacement Rel Error')
        plt.show()



if __name__ == "__main__":

    data_path='Kratos_results/'

    rom_file='NN_ROM_finetuned1_noForce.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    rom_file='NN_ROM_finetuned2_noForce.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    rom_file='SVD1_ROM.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    rom_file='SVD2_ROM.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    rom_file='NN_ROM_reference.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    rom_file='NN_ROM_control.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    rom_file='NN_ROM_finetuned1.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    rom_file='NN_ROM_finetuned2.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    rom_file='NN_ROM_finetuned1_partial.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom[-S_rom.shape[0]:], S_rom)

    draw_x_error_abs_image(S_fom[-S_rom.shape[0]:], S_rom)
    draw_x_error_rel_image(S_fom[-S_rom.shape[0]:], S_rom)
