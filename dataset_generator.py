import json

import h5py
import numpy as np
import scipy

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import pandas as pd

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as ROM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from fom_analysis import CreateRomAnalysisInstance


def generate_finetune_datasets(dataset_path):

    S=np.load(dataset_path+'FOM.npy')
    R=np.load(dataset_path+'FOM_RESIDUALS.npy')
    F=np.load(dataset_path+'FOM_POINLOADS.npy')

    train_size=5000/S.shape[0]
    test_size=10000/S.shape[0]

    S_train, S_test, R_train, R_test, F_train, F_test = train_test_split(S,R,F, test_size=test_size, train_size=train_size, random_state=250)

    print('Shape S_train: ', S_train.shape)
    print('Shape S_test:', S_test.shape)
    print('Shape R_train:', R_train.shape)
    print('Shape R_test: ', R_test.shape)
    print('Shape F_train: ', F_train.shape)
    print('Shape F_test: ', F_test.shape)

    with open(dataset_path+"S_finetune_train.npy", "wb") as f:
        np.save(f, S_train)
    with open(dataset_path+"S_finetune_test.npy", "wb") as f:
        np.save(f, S_test)
    with open(dataset_path+"R_finetune_train.npy", "wb") as f:
        np.save(f, R_train)
    with open(dataset_path+"R_finetune_test.npy", "wb") as f:
        np.save(f, R_test)
    with open(dataset_path+"F_finetune_train.npy", "wb") as f:
        np.save(f, F_train)
    with open(dataset_path+"F_finetune_test.npy", "wb") as f:
        np.save(f, F_test)

def generate_training_datasets(dataset_path):

    S=np.load(dataset_path+'FOM.npy')
    R=np.load(dataset_path+'FOM_RESIDUALS.npy')
    F=np.load(dataset_path+'FOM_POINLOADS.npy')

    test_size=1-20000/S.shape[0]

    S_train, S_test, R_train, R_test, F_train, F_test = train_test_split(S,R,F, test_size=test_size, random_state=274)

    print('Shape S_train: ', S_train.shape)
    print('Shape S_test:', S_test.shape)
    print('Shape R_train:', R_train.shape)
    print('Shape R_test: ', R_test.shape)
    print('Shape F_train: ', F_train.shape)
    print('Shape F_test: ', F_test.shape)

    with open(dataset_path+"S_train.npy", "wb") as f:
        np.save(f, S_train)
    with open(dataset_path+"S_test.npy", "wb") as f:
        np.save(f, S_test)
    with open(dataset_path+"R_train.npy", "wb") as f:
        np.save(f, R_train)
    with open(dataset_path+"R_test.npy", "wb") as f:
        np.save(f, R_test)
    with open(dataset_path+"F_train.npy", "wb") as f:
        np.save(f, F_train)
    with open(dataset_path+"F_test.npy", "wb") as f:
        np.save(f, F_test)

def InitializeKratosAnalysis():
    with open("ProjectParameters_fom.json", 'r') as parameter_file:
        parameters = KMP.Parameters(parameter_file.read())

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KMP.Model()
    fake_simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    fake_simulation.Initialize()
    fake_simulation.InitializeSolutionStep()

    return fake_simulation

def project_prediction(snapshot, f, modelpart):
        values = snapshot
        f_value=f[0]

        itr = 0
        for node in modelpart.Nodes:
            if not node.IsFixed(KMP.DISPLACEMENT_X):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_X, values[itr+0])
                node.X = node.X0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_X)

            if not node.IsFixed(KMP.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_Y, values[itr+1])
                node.Y = node.Y0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_Y)

            if not node.IsFixed(KMP.DISPLACEMENT_X):
                itr += 2
            
        for condition in modelpart.Conditions:
            condition.SetValue(SMA.POINT_LOAD, f_value)

def get_r(fake_simulation, snapshot, f):
        snapshot = snapshot[4:]
        space =     KMP.UblasSparseSpace()
        strategy  = fake_simulation._GetSolver()._GetSolutionStrategy()
        buildsol  = fake_simulation._GetSolver()._GetBuilderAndSolver()
        scheme    = KMP.ResidualBasedIncrementalUpdateStaticScheme()
        modelpart = fake_simulation._GetSolver().GetComputingModelPart()

        A = strategy.GetSystemMatrix()
        b = strategy.GetSystemVector()

        space.SetToZeroMatrix(A)
        space.SetToZeroVector(b)

        project_prediction(snapshot, f, modelpart)

        buildsol.Build(scheme, modelpart, A, b)

        # buildsol.ApplyDirichletConditions(scheme, modelpart, A, b, b)

        b=np.array(b)

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))

        raw_A = -Ascipy.todense()
        raw_A = raw_A[:,4:]
        
        return b

def apply_random_noise(x_true):
        v=np.random.rand(x_true[4:].shape[0])
        v=v/np.linalg.norm(v)
        eps=np.random.rand()*1e-4
        v=v*eps
        x_app=x_true
        x_app[4:]=x_app[4:]+v

        return x_app

def generate_augm_finetune_datasets(dataset_path, augm_order):
    
    # Create a fake Analysis stage to calculate the predicted residuals
    fake_simulation = InitializeKratosAnalysis()
    with open(dataset_path+"S_finetune_train.npy", "rb") as f:
        S_train=np.load(f)
    with open(dataset_path+"F_finetune_train.npy", "rb") as f:
        F_train=np.load(f)
    with open(dataset_path+"R_finetune_train.npy", "rb") as f:
        R_train=np.load(f)

    S_augm=[]
    F_augm=[]
    R_augm=[]

    for i in range(S_train.shape[0]):
        S_augm.append(S_train[i])
        F_augm.append(F_train[i])
        R_augm.append(R_train[i])
        for n in range(augm_order):
            s_noisy = apply_random_noise(S_train[i])
            r_noisy = get_r(fake_simulation, s_noisy, F_train[i])
            S_augm.append(s_noisy)
            F_augm.append(F_train[i])
            R_augm.append(r_noisy)

        if i%100 == 0:
            print('Iteration: ', i, 'of ', S_train.shape[0], '. Current length: ', len(S_augm))
    
    S_augm=np.array(S_augm)
    F_augm=np.array(F_augm)
    R_augm=np.array(R_augm)

    with open(dataset_path+"S_augm_train.npy", "wb") as f:
        np.save(f, S_augm)
    with open(dataset_path+"R_augm_train.npy", "wb") as f:
        np.save(f, R_augm)
    with open(dataset_path+"F_augm_train.npy", "wb") as f:
        np.save(f, F_augm)


if __name__ == "__main__":

    dataset_path='datasets_rommanager/'

    # generate_training_datasets(dataset_path)
    # generate_finetune_datasets(dataset_path)
    generate_augm_finetune_datasets(dataset_path, 3)