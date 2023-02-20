import os
import sys
import math

import numpy as np
np.random.seed(seed=74)
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

    def __init__(self, kratos_sim):
        self.fake_simulation=kratos_sim

    def project_prediction(self, y_pred, modelpart):
        values = y_pred[0]

        itr = 0

        for node in modelpart.Nodes:
            if not node.IsFixed(KMP.DISPLACEMENT_X):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_X, values[itr+0])
                node.X = node.X0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_X)

            if not node.IsFixed(KMP.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_Y, values[itr+1])
                node.Y = node.Y0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_Y)

            itr += 2

    def get_r(self, y_pred):
        space =     KMP.UblasSparseSpace()
        strategy  = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        buildsol  = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        scheme    = KMP.ResidualBasedIncrementalUpdateStaticScheme()
        modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()
        
        A = strategy.GetSystemMatrix()
        b = strategy.GetSystemVector()#KMP.Vector(52)

        space.SetToZeroMatrix(A)
        space.SetToZeroVector(b)

        self.project_prediction(y_pred, modelpart)

        buildsol.Build(scheme, modelpart, A, b)

        b=np.array(b,copy=False)# list(b.__iter__()))

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))

        raw_A = -Ascipy.todense()
        
        return raw_A/3e9, b/3e9

    
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
            # print(b_true)
            # print(b_app)
            # print('')

            first_order=A_true@ev.T
            L=b_app-b_true-first_order.T
            err_vec.append(np.linalg.norm(L))

        return err_vec

        #     L_high=L[:,:4]
        #     L_low=L[:,4:]
        #     err_vec_h.append(np.linalg.norm(L_high))
        #     err_vec_l.append(np.linalg.norm(L_low))

        # return err_vec, err_vec_h, err_vec_l


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

    # Create a fake Analysis stage to calculate the predicted residuals
    with open("ProjectParameters_fom.json", 'r') as parameter_file:
        parameters = KMP.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KMP.Model()
    fake_simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    fake_simulation.Initialize()
    fake_simulation.InitializeSolutionStep()    

    S, R = prepare_input(data_inputs_files, residuals_files, pointloads_files)
    print('Shape S: ', S.shape)
    print('Shape R: ', R.shape)

    resiudal_tester = DecayTester(fake_simulation)

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
