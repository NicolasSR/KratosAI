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

    def __init__(self, kratos_sim):
        self.fake_simulation=kratos_sim

    def project_prediction(self, y_pred, f_value, modelpart):
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

        for condition in modelpart.Conditions:
            condition.SetValue(SMA.POINT_LOAD, [0.0,0.0,0.0])

    def project_prediction_force(self, y_pred, f_value, modelpart):
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

        for condition in modelpart.Conditions:
            condition.SetValue(SMA.POINT_LOAD, f_value)

    def get_r(self, y_pred, f_value):
        space =     KMP.UblasSparseSpace()
        strategy  = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        buildsol  = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        scheme    = KMP.ResidualBasedIncrementalUpdateStaticScheme()
        modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()
        
        A = strategy.GetSystemMatrix()
        b = strategy.GetSystemVector()#KMP.Vector(52)

        space.SetToZeroMatrix(A)
        space.SetToZeroVector(b)

        self.project_prediction(y_pred, f_value, modelpart)

        buildsol.Build(scheme, modelpart, A, b)

        b=np.array(b)# list(b.__iter__()))

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))

        raw_A = -Ascipy.todense()
        
        return raw_A/1e9, b/1e9
        # return raw_A, b

    def get_r_force(self, y_pred, f_value):
        space =     KMP.UblasSparseSpace()
        strategy  = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        buildsol  = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        scheme    = KMP.ResidualBasedIncrementalUpdateStaticScheme()
        modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()
        
        A = strategy.GetSystemMatrix()
        b = strategy.GetSystemVector()#KMP.Vector(52)

        space.SetToZeroMatrix(A)
        space.SetToZeroVector(b)

        self.project_prediction_force(y_pred, f_value, modelpart)

        buildsol.Build(scheme, modelpart, A, b)

        b=np.array(b)# list(b.__iter__()))

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))

        raw_A = -Ascipy.todense()
        
        return raw_A/1e9, b/1e9
        # return raw_A, b

    
    def test_error(self,x_true,r_true, f_value, eps_vec):
        
        x_true = np.expand_dims(x_true, axis=0)
        print('x_true')
        print(x_true)
        
        A_true_1, b_true = self.get_r_force(x_true, f_value)
        A_true_2, _ = self.get_r(x_true, f_value)
        print(A_true_1)
        print(A_true_2)
        print(np.all(A_true_1-A_true_2==0))
        print(r_true)
        print(b_true)
        exit()

        # b_true=r_true/1e9

        v=np.random.rand(1,52)
        v[0,:4]=0
        v=v/np.linalg.norm(v)
        print(v)
        print(np.linalg.norm(v))

        err_vec=[]

        for eps in eps_vec:

            ev=v*eps
            x_app=x_true+ev
            _, b_app = self.get_r(x_app, f_value)

            first_order=A_true@ev.T
            L=b_app-b_true-first_order.T
            err_vec.append(np.linalg.norm(L))

        return err_vec


def prepare_input(dataset_path):

    S_orig=np.load(dataset_path+'FOM.npy')
    S_orig_train=np.load(dataset_path+'S_train.npy')
    S_orig_test=np.load(dataset_path+'S_test.npy')
    R_train=np.load(dataset_path+'R_train.npy')
    R_test=np.load(dataset_path+'R_test.npy')
    F_train=np.load(dataset_path+'F_train.npy')[:,0,:]
    F_test=np.load(dataset_path+'F_test.npy')[:,0,:]

    return S_orig, S_orig_train, S_orig_test, R_train, R_test, F_train, F_test



if __name__ == "__main__":

    # Defining variable values:

    S_orig, S_orig_train, S_orig_test, R_train, R_test, F_train, F_test = prepare_input('datasets_low/')
    print('Shape S_orig: ', S_orig.shape)
    print('Shape S_orig_train:', S_orig_train.shape)
    print('Shape S_orig_test:', S_orig_test.shape)
    print('Shape R_train: ', R_train.shape)
    print('Shape R_est: ', R_test.shape)
    print('Shape F_train: ', F_train.shape)
    print('Shape F_test: ', F_test.shape)

    # Create a fake Analysis stage to calculate the predicted residuals
    with open("ProjectParameters_fom.json", 'r') as parameter_file:
        parameters = KMP.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KMP.Model()
    fake_simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    fake_simulation.Initialize()
    fake_simulation.InitializeSolutionStep()

    resiudal_tester = DecayTester(fake_simulation)

    rand_id = np.random.choice(np.arange(S_orig_train.shape[0]))
    snapshot = S_orig_train[rand_id]
    residual = R_train[rand_id]
    f_value = F_train[rand_id]

    eps_vec = np.logspace(1, 12, 1000)/1e13
    # eps_vec = np.linspace(0.001, 5.0, 1000)
    # eps_vec = np.linspace(0.000001, 0.001, 1000)
    square=np.power(eps_vec,2)

    errors = resiudal_tester.test_error(snapshot,residual,f_value,eps_vec)
    
    plt.plot(eps_vec, square, "--", label="square")
    plt.plot(eps_vec, eps_vec, "--", label="linear")
    plt.plot(eps_vec, errors, label="error")
    # plt.plot(eps_vec, err_h, label="error_h")
    # plt.plot(eps_vec, err_l, label="error_l")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left")
    plt.show()
