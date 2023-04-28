import os
import sys
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import h5py
import numpy as np

import keras
import tensorflow as tf

from keras import layers
from itertools import repeat

import importlib
import contextlib

import KratosMultiphysics
import KratosMultiphysics.RomApplication as KratosROM

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

from KratosMultiphysics.RomApplication.networks import gradient_shallow as gradient_shallow_ae
from KratosMultiphysics.RomApplication import kratos_io as kratos_io
from KratosMultiphysics.RomApplication import clustering as clustering
from KratosMultiphysics.RomApplication import python_solvers_wrapper_rom as solver_wrapper

from KratosMultiphysics.RomApplication import new_python_solvers_wrapper_rom
from KratosMultiphysics.RomApplication.hrom_training_utility import HRomTrainingUtility
from KratosMultiphysics.RomApplication.calculate_rom_basis_output_process import CalculateRomBasisOutputProcess

from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

def custom_loss(y_true, y_pred):
    y_diff = y_true-y_pred
    y_diff = y_diff ** 2

    return  y_diff

def CreateRomAnalysisInstance(cls, global_model, parameters):
    class RomAnalysis(cls):

        def __init__(self,global_model, parameters):
            super().__init__(global_model, parameters)
            self.override_first_iter = False
            self.snapshots_matrix = []

        ### NN ##
        def _CreateModel(self):

            S_train=np.load('datasets_rommanager/S_train.npy')
            S_test=np.load('datasets_rommanager/S_test.npy')

            self.U,sigma,_ = np.linalg.svd(S_train.T, full_matrices=True, compute_uv=True, hermitian=False)
            self.U = self.U[:,:2]
            print(self.U.shape)

            """ S_pred = self.U@self.U.T@S_train.T

            # Check the estimation error:
            s__norm = np.linalg.norm(S_train.T, axis=0)
            sp_norm = np.linalg.norm(S_pred-S_train.T, axis=0)
            print(s__norm.shape)
            print(sp_norm.shape)
            print('Train mean rel L2 norm:')
            print(np.mean(sp_norm/s__norm))

            s__norm = np.linalg.norm(S_train.T)
            sp_norm = np.linalg.norm(S_pred-S_train.T)
            print(s__norm.shape)
            print(sp_norm.shape)
            print('Train rel frob norm:')
            print(sp_norm/s__norm)


            # Check the prediction correctness:
            s__norm = np.linalg.norm(S_train[0])
            sp_norm = np.linalg.norm(S_pred[:,0]-S_train[0])
            print(s__norm)
            print(sp_norm)
            print("Norm error for this iteration", sp_norm/s__norm)

            S_test_pred = self.U@self.U.T@S_test.T

            # Check the estimation error:
            s__norm = np.linalg.norm(S_test.T, axis=0)
            sp_norm = np.linalg.norm(S_test_pred-S_test.T, axis=0)
            print(s__norm.shape)
            print(sp_norm.shape)
            print('Test mean rel L2 norm:')
            print(np.mean(sp_norm/s__norm))

            s__norm = np.linalg.norm(S_test.T)
            sp_norm = np.linalg.norm(S_test_pred-S_test.T)
            print(s__norm.shape)
            print(sp_norm.shape)
            print('Test rel frob norm:')
            print(sp_norm/s__norm)

            # Check the prediction correctness:
            s__norm = np.linalg.norm(S_test[0])
            sp_norm = np.linalg.norm(S_test_pred[:,0]-S_test[0])
            print(s__norm)
            print(sp_norm)
            print("Norm error for this iteration", sp_norm/s__norm) """

        def _CreateSolver(self):
            """ Create the Solver (and create and import the ModelPart if it is not alread in the model) """

            # Get the ROM settings from the RomParameters.json input file
            with open('RomParameters.json') as rom_parameters:
                self.rom_parameters = KratosMultiphysics.Parameters(rom_parameters.read())

            # Set the ROM settings in the "solver_settings" of the solver introducing the physics
            # self.project_parameters["solver_settings"].AddValue("rom_settings", self.rom_parameters["rom_settings"])

            # HROM operations flags
            self.rom_basis_process_list_check = False
            self.rom_basis_output_process_check = False
            self.run_hrom = False # self.rom_parameters["run_hrom"].GetBool() if self.rom_parameters.Has("run_hrom") else False
            self.train_hrom = False # self.rom_parameters["train_hrom"].GetBool() if self.rom_parameters.Has("train_hrom") else False
            if self.run_hrom and self.train_hrom:
                # Check that train an run HROM are not set at the same time
                err_msg = "\'run_hrom\' and \'train_hrom\' are both \'true\'. Select either training or running (if training has been already done)."
                raise Exception(err_msg)

            # Create the ROM solver
            return new_python_solvers_wrapper_rom.CreateSolver(
                self.model,
                self.project_parameters)

        def _GetListOfProcesses(self):
            # Get the already existent processes list
            list_of_processes = super()._GetListOfProcesses()

            # Check if there is any instance of ROM basis output
            if self.rom_basis_process_list_check:
                for process in list_of_processes:
                    if isinstance(process, KratosROM.calculate_rom_basis_output_process.CalculateRomBasisOutputProcess):
                        warn_msg = "\'CalculateRomBasisOutputProcess\' instance found in ROM stage. Basis must be already stored in \'RomParameters.json\'. Removing instance from processes list."
                        KratosMultiphysics.Logger.PrintWarning("RomAnalysis", warn_msg)
                        list_of_processes.remove(process)
                self.rom_basis_process_list_check = False

            return list_of_processes

        def _GetListOfOutputProcesses(self):
            # Get the already existent output processes list
            list_of_output_processes = super()._GetListOfOutputProcesses()

            # Check if there is any instance of ROM basis output
            if self.rom_basis_output_process_check:
                for process in list_of_output_processes:
                    if isinstance(process, KratosROM.calculate_rom_basis_output_process.CalculateRomBasisOutputProcess):
                        warn_msg = "\'CalculateRomBasisOutputProcess\' instance found in ROM stage. Basis must be already stored in \'RomParameters.json\'. Removing instance from output processes list."
                        KratosMultiphysics.Logger.PrintWarning("RomAnalysis", warn_msg)
                        list_of_output_processes.remove(process)
                self.rom_basis_output_process_check = False

            return list_of_output_processes

        def _GetSimulationName(self):
            return "::[ROM Simulation]:: "

        def InitializeSolutionStep(self):
            computing_model_part = self._GetSolver().GetComputingModelPart().GetRootModelPart()
            
            # Reset the diplacement for this iteration
            computing_model_part.SetValue(KratosROM.ROM_SOLUTION_INCREMENT, [0.0, 0.0])

            # Generate the input snapshot
            self.S = []
            for node in computing_model_part.Nodes:
                self.S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                self.S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))

            self.S = np.array(self.S)

            self.NSPredict_d = self.U@self.U.T@self.S


            self.q = self.U.T@self.S

            print("Shape of S:", self.S.shape)
            print("Shape of NSPredict_d:", self.NSPredict_d.shape)
            print("Shape of q:", self.q.shape)


            # Check the prediction correctness:
            s__norm = np.linalg.norm(self.S)
            sp_norm = np.linalg.norm(self.NSPredict_d-self.S)
            # print(s__norm)
            # print(sp_norm)

            if s__norm != 0:
                print("Norm error for this iteration", sp_norm/s__norm)
            else:
                print("Norm error  is 0")
                np.set_printoptions(threshold=sys.maxsize)
            
            nodal_modes_d = self.U

            nodal_dofs = 2
            rom_dofs = self.U.shape[1]

            # aux_e = KratosMultiphysics.Matrix(nodal_dofs, rom_dofs)
            aux_d = KratosMultiphysics.Matrix(nodal_dofs, rom_dofs)

            for node in computing_model_part.Nodes:
                node_id = node.Id-1
                for j in range(nodal_dofs):
                    for i in range(rom_dofs):
                        aux_d[j,i] = nodal_modes_d[node_id*nodal_dofs+j][i]

                node.SetValue(KratosROM.ROM_BASIS,     aux_d)

            super().InitializeSolutionStep()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            snapshot = []
            for node in self._GetSolver().GetComputingModelPart().Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
            self.snapshots_matrix.append(snapshot)

        def ModifyAfterSolverInitialize(self):
            """Here is where the ROM_BASIS is imposed to each node"""
            super().ModifyAfterSolverInitialize()

            # Get the model part where the ROM is to be applied
            computing_model_part = self._GetSolver().GetComputingModelPart().GetRootModelPart()
            # computing_model_part = self._GetSolver().GetComputingModelPart()

            # Initialize NN
            self._CreateModel()

        def Finalize(self):
            # This calls the physics Finalize
            super().Finalize()

            np.save("NN_ROM.npy",self.snapshots_matrix)

            # Once simulation is completed, calculate and save the HROM weights
            if self.train_hrom:
                self.__hrom_training_utility.CalculateAndSaveHRomWeights()
                self.__hrom_training_utility.CreateHRomModelParts()

    return RomAnalysis(global_model, parameters)

if __name__ == "__main__":

    with open("ProjectParameters_nnm.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KratosMultiphysics.Model()
    simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    
    # simulation = analysis_stage_class(global_model, parameters)
    simulation.Run()
