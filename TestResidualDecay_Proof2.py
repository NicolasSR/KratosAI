import os
import sys
import math

import numpy as np
np.random.seed(seed=245)
import scipy

import utils
import networks.network as network

import kratos_io
import matplotlib.pyplot as plt

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as ROM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from fom_analysis import CreateRomAnalysisInstance
from utils.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition


class DecayTester():

    def __init__(self, length):
        self.gen_random_matrices(length)

    def get_r(self, y_pred):

        id_mat=np.identity(y_pred.shape[1])
        ones_vec=np.ones((1,y_pred.shape[1]))
        
        b_1=ones_vec.T@y_pred
        b_2=b_1@y_pred.T
        b_3=y_pred.T@self.b_vec
        b_4=b_3@y_pred.T
        b_5=self.c_mat@y_pred.T
        b = self.a_mat@b_2+b_4+b_5
        b=b.T

        A_1=ones_vec.T@y_pred
        A_2=self.a_mat@A_1
        A_3=y_pred.T@self.b_vec
        A_4=y_pred@self.b_vec.T
        A_5=A_4*id_mat
        A = 2*A_2+A_3+A_5+self.c_mat#+self.noise_mat

        """ print('id_mat')
        print(id_mat)
        print(id_mat.shape)

        print('b_1')
        print(b_1)
        print(b_1.shape)
        print('b_2')
        print(b_2)
        print(b_2.shape)
        print('b_3')
        print(b_3)
        print(b_3.shape)
        print('b')
        print(b)
        print(b.shape)

        print('A_1')
        print(A_1)
        print(A_1.shape)
        print('A_2')
        print(A_2)
        print(A_2.shape)
        print('A_3')
        print(A_3)
        print(A_3.shape) """

        return A, b

    def gen_random_matrices(self, length):
        self.a_mat = np.random.rand(length,length)*40-20
        self.b_vec = np.random.rand(1,length)*30-15
        self.c_mat = np.random.rand(length,length)*26-13
        self.noise_mat = np.random.rand(length,length)*20-10

        print('A, b and C')
        print(self.a_mat)
        print(self.a_mat.shape)
        print(self.b_vec)
        print(self.b_vec.shape)
        print(self.c_mat)
        print(self.c_mat.shape)
    
    def test_error(self,x_true,r_true,eps_vec):
        
        x_true = np.expand_dims(x_true, axis=0)
        print('x_true')
        print(x_true)
        
        A_true, b_true = self.get_r(x_true)
        print(A_true)
        print(r_true)
        print(b_true)

        v=np.random.rand(1,52)
        v=v/np.linalg.norm(v)
        print(v)
        print(np.linalg.norm(v))

        err_vec=[]

        for eps in eps_vec:

            ev=v*eps
            x_app=x_true+ev
            _, b_app = self.get_r(x_app)

            first_order=A_true@ev.T
            L=b_app-b_true-first_order.T
            err_vec.append(np.linalg.norm(L))

        return err_vec


def prepare_input(data_inputs_files, residuals_files, pointloads_files):

    variables_list=['DISPLACEMENT'] # List of variables to run be used (VELOCITY, PRESSURE, etc...)
    S = kratos_io.build_snapshot_grid(data_inputs_files,variables_list) # Both ST and S are the same total S matrix

    R = None
    for r in residuals_files:
        # a = np.load(r) / 3e9 # We scale the resiuals because they are too close to 0 originally?
        a = np.load(r) # We scale the resiuals because they are too close to 0 originally?
        if R is None:
            R = a
        else:
            R = np.concatenate((R, a), axis=0)

    return S, R



if __name__ == "__main__":

    # Defining variable values:

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

    S, R = prepare_input(data_inputs_files, residuals_files, pointloads_files)
    print('Shape S: ', S.shape)

    resiudal_tester = DecayTester(S.shape[0])

    rand_id = np.random.choice(np.arange(S.shape[1]))
    snapshot = S.T[rand_id]
    residual = R[rand_id]

    eps_vec = np.logspace(1, 12, 1000)/1e13
    # eps_vec = np.linspace(0.001, 5.0, 1000)
    # eps_vec = np.linspace(0.000001, 0.001, 1000)
    square=np.power(eps_vec,2)

    errors = resiudal_tester.test_error(snapshot,residual,eps_vec)
    
    plt.plot(eps_vec, square, "--", label="square")
    plt.plot(eps_vec, eps_vec, "--", label="linear")
    plt.plot(eps_vec, errors, label="error")
    # plt.plot(eps_vec, err_h, label="error_h")
    # plt.plot(eps_vec, err_l, label="error_l")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left")
    plt.show()
