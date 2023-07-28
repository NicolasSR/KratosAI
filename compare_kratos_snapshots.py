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
from utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error, relative_l2_error_list

from sklearn.model_selection import train_test_split


def prepare_input(dataset_path, rom_file, fom_file):

    S_fom=np.load(dataset_path+fom_file)
    S_rom=np.load(dataset_path+rom_file)

    return S_fom, S_rom

def calculate_X_norm_error(S_fom, S_rom):
    l2_error=mean_relative_l2_error(S_fom,S_rom)
    forb_error=relative_forbenius_error(S_fom,S_rom)
    print('X. Mean rel L2 error:', l2_error)
    print('X. Rel Forb. error:', forb_error)

def plot_rel_l2_errors(S_fom,S_rom):
    rel_l2_errors_list=relative_l2_error_list(S_fom,S_rom)
    plt.plot(rel_l2_errors_list)
    # plt.semilogy()
    plt.xlabel('Step')
    plt.ylabel('Relative L2 error of displacements (nn vs fom)')
    plt.show()
    
def plot_residual_norms(R_fom, R_nn):
    crop_indices=np.array([0,1,5,10,16,24,34,46,61,77,94,113,133,155,182,209,238,267,298,329,362])
    crop_indices=np.concatenate([crop_indices*2,crop_indices*2+1])
    # R_nn_norms=np.linalg.norm(R_nn, axis=1)
    # plt.plot(R_nn_norms)
    # R_fom_norms=np.linalg.norm(R_fom, axis=1)
    # plt.plot(R_fom_norms)
    # lineal=np.linspace(200000,6e7,300)
    # plt.plot(np.sqrt(lineal**2+lineal**2))
    # plt.show()

    R_nn_sums_x=np.sum(R_nn[:,0::2], axis=1)
    R_nn_sums_y=np.sum(R_nn[:,1::2], axis=1)
    R_fom_sums_x=np.sum(R_nn[:,0::2], axis=1)
    R_fom_sums_y=np.sum(R_nn[:,1::2], axis=1)
    plt.plot(R_nn_sums_x, label='nn')
    plt.plot(R_fom_sums_x,'.', label='fom')
    plt.xlabel('Step')
    plt.ylabel('Sum of reactions_x')
    plt.legend()
    plt.show()
    plt.plot(R_nn_sums_y, label='nn')
    plt.plot(R_fom_sums_y,'.', label='fom')
    plt.xlabel('Step')
    plt.ylabel('Sum of reactions_y')
    plt.legend()
    plt.show()
    
    R_nn=np.delete(R_nn,crop_indices, axis=1)
    R_fom=np.delete(R_fom,crop_indices, axis=1)
    R_nn_norms=np.linalg.norm(R_nn, axis=1)
    plt.plot(R_nn_norms, label='nn')
    R_fom_norms=np.linalg.norm(R_fom, axis=1)
    plt.plot(R_fom_norms, label='fom')
    plt.xlabel('Step')
    plt.ylabel('Norm of residuals (excluding Dirichlet)')
    plt.legend()
    plt.show()

def draw_x_error_abs_image(S_fom, S_rom):
    fig, (ax1) = plt.subplots(ncols=1)
    image=np.abs(S_rom-S_fom)
    im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', cmap='jet')
    # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=0, vmax=0.004, cmap='jet')
    # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=-0.015, vmax=0.03, cmap='jet')
    ax1.set_aspect(1/2e3)
    cbar1 = plt.colorbar(im1)
    plt.xlabel('index')
    plt.ylabel('force')
    plt.title('Displacement Abs Error')
    plt.show()

def draw_x_error_rel_image(S_fom, S_rom):
    fig, (ax1) = plt.subplots(ncols=1)
    image=np.abs((S_rom-S_fom))/(np.abs(S_fom)+1e-14)
    im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', cmap='jet')
    # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=0, vmax=1, cmap='jet')
    # im1 = ax1.imshow(image, extent=[1,S_fom.shape[1],6.0e6,0], interpolation='none', vmin=-0.1, vmax=0.1, cmap='jet')
    ax1.set_aspect(1/2e3)
    cbar1 = plt.colorbar(im1)
    plt.xlabel('index')
    plt.ylabel('force')
    plt.title('Displacement Rel Error')
    plt.show()



if __name__ == "__main__":

    ###################
    # data_path='Kratos_results/LSPG_LR5_1000ep/EqualForces/'
    data_path=''

    # rom_file='NN_ROM_Reference.npy'
    rom_file='NN_snaps.npy'
    fom_file='FOM_snaps.npy'

    residuals_nn_matrix=np.load('NN_residuals.npy')
    residuals_fom_matrix=np.load('FOM_residuals.npy')
    plot_residual_norms(residuals_fom_matrix,residuals_nn_matrix)

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file, fom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    plot_rel_l2_errors(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    exit()
    #################

    ###################
    # data_path='Kratos_results/LSPG_LR5_1000ep/EqualForces/'
    """ data_path=''

    # rom_file='NN_ROM_Reference.npy'
    rom_file='Scipy_x_snapshots_absloss_diff1000.npy'
    fom_file='Scipy_FOM_x_snapshots.npy'

    # residuals_nn_matrix=np.load('NN_residuals.npy')
    # residuals_fom_matrix=np.load('FOM_residuals.npy')
    # plot_residual_norms(residuals_fom_matrix,residuals_nn_matrix)

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file, fom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    plot_rel_l2_errors(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    residual_norms=np.load('Scipy_residual_norms_absloss_diff1000.npy')
    plt.plot(residual_norms[:,0], label='staring point')
    plt.plot(residual_norms[:,1], label='converged')
    plt.xlabel('Step')
    plt.ylabel('Residual norm (excuding Dirichlet)')
    plt.legend()
    plt.show()

    exit() """
    #################


    ###################
    # data_path='Kratos_results/LSPG_LR5_1000ep/EqualForces/'
    """ data_path=''

    # rom_file='NN_ROM_Reference.npy'
    rom_file='Scipy_x_snapshots_rdiffloss_initq0.npy'
    fom_file='Scipy_FOM_x_snapshots.npy'

    # residuals_nn_matrix=np.load('NN_residuals.npy')
    # residuals_fom_matrix=np.load('FOM_residuals.npy')
    # plot_residual_norms(residuals_fom_matrix,residuals_nn_matrix)

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file, fom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    plot_rel_l2_errors(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    residual_norms=np.load('Scipy_residual_norms_rdiffloss_initq0.npy')
    plt.plot(residual_norms[:,0], label='staring point')
    plt.plot(residual_norms[:,1], label='converged')
    plt.xlabel('Step')
    plt.ylabel('Residual norm (excuding Dirichlet)')
    plt.legend()
    plt.show()

    exit() """
    #################


    ###################
    # data_path='Kratos_results/LSPG_LR5_1000ep/EqualForces/'
    """ data_path=''

    # rom_file='NN_ROM_Reference.npy'
    rom_file='Scipy_x_snapshots_rdiffloss_diff1000.npy'
    fom_file='Scipy_FOM_x_snapshots.npy'

    # residuals_nn_matrix=np.load('NN_residuals.npy')
    # residuals_fom_matrix=np.load('FOM_residuals.npy')
    # plot_residual_norms(residuals_fom_matrix,residuals_nn_matrix)

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file, fom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    plot_rel_l2_errors(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    residual_norms=np.load('Scipy_residual_norms_rdiffloss_diff1000.npy')
    plt.plot(residual_norms[:,0], label='staring point')
    plt.plot(residual_norms[:,1], label='converged')
    plt.xlabel('Step')
    plt.ylabel('Residual norm (excuding Dirichlet)')
    plt.legend()
    plt.show()

    exit() """
    #################

    ###################
    # data_path='Kratos_results/LSPG_LR5_1000ep/EqualForces/'
    """ data_path=''

    # rom_file='NN_ROM_Reference.npy'
    rom_file='Scipy_x_snapshots.npy'
    fom_file='Scipy_FOM_x_snapshots.npy'

    # residuals_nn_matrix=np.load('NN_residuals.npy')
    # residuals_fom_matrix=np.load('FOM_residuals.npy')
    # plot_residual_norms(residuals_fom_matrix,residuals_nn_matrix)

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file, fom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    plot_rel_l2_errors(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    residual_norms=np.load('Scipy_residual_norms.npy')
    plt.plot(residual_norms[:,0], label='staring point')
    plt.plot(residual_norms[:,1], label='converged')
    plt.xlabel('Step')
    plt.ylabel('Residual norm (excuding Dirichlet)')
    plt.legend()
    plt.show()

    exit() """
    #################


    """ rom_file='NN_ROM_Finetuned.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom)

    rom_file='POD2_ROM.npy'

    # Get snapshots
    S_fom, S_rom = prepare_input(data_path, rom_file)
    print('Shape S_fom:', S_fom.shape)
    print('Shape S_rom:', S_rom.shape)

    print('Error norms')
    calculate_X_norm_error(S_fom, S_rom)

    draw_x_error_abs_image(S_fom, S_rom)
    draw_x_error_rel_image(S_fom, S_rom) """