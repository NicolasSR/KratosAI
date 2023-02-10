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
            # List of files to read from
            data_inputs = [
                "hdf5_bases/result_80000.h5",
                "hdf5_bases/result_90000.h5",
                "hdf5_bases/result_100000.h5",
                "hdf5_bases/result_110000.h5",
                "hdf5_bases/result_120000.h5",
            ]

            # List of variables to run be used (VELOCITY, PRESSURE, etc...)
            S = kratos_io.build_snapshot_grid(
                data_inputs, 
                [
                    "DISPLACEMENT",
                ]
            )

            self.SORG = S

            # Some configuration
            self.config = {
                "load_model":       True,
                "train_model":      False,
                "save_model":       False,
                "print_results":    False,
                "use_reduced":      False,
            }

            # Select the netwrok to use
            self.kratos_network = gradient_shallow_ae.GradientShallow()

            print("=== Calculating Randomized Singular Value Decomposition ===")
            with contextlib.redirect_stdout(None):
            # Note: we redirect the output here because there is no option to reduce the log level of this method.
                # self.U,sigma,_,error = RandomizedSingularValueDecomposition().Calculate(S,1e-16)
                self.U,sigma,_ = np.linalg.svd(S, full_matrices=True, compute_uv=True, hermitian=False)
                self.U = self.U[:,:25]

                SPri = self.U @ self.U.T @ S

            # Select the reduced snapshot or the full input
            if self.config["use_reduced"]:
                SReduced = self.U.T @ S
            else:
                SReduced = S

            data_rows = SReduced.shape[0]
            data_cols = SReduced.shape[1]

            self.kratos_network.calculate_data_limits(SReduced)

            # Set the properties for the clusters
            num_clusters=1                      # Number of different bases chosen
            num_cluster_col=2                   # If I use num_cluster_col = num_variables result should be exact.
            num_encoding_var=num_cluster_col

            # Load the model or train a new one. TODO: Fix custom_loss not being saved correctly
            if self.config["load_model"]:
                self.autoencoder = tf.keras.models.load_model(self.kratos_network.model_name, custom_objects={
                    "custom_loss":custom_loss,
                    "set_m_grad":gradient_shallow_ae.GradModel2.set_m_grad
                })
                self.kratos_network.encoder_model = tf.keras.models.load_model(self.kratos_network.model_name+"_encoder", custom_objects={
                    "custom_loss":custom_loss,
                    "set_m_grad":gradient_shallow_ae.GradModel2.set_m_grad
                })
                self.kratos_network.decoder_model = tf.keras.models.load_model(self.kratos_network.model_name+"_decoder", custom_objects={
                    "custom_loss":custom_loss,
                    "set_m_grad":gradient_shallow_ae.GradModel2.set_m_grad
                })
            else:
                self.autoencoder, self.autoencoder_err = self.kratos_network.define_network(SReduced, custom_loss, num_encoding_var)

            # Prepare the data
            SReducedNormalized = (SReduced - self.kratos_network.data_min) / (self.kratos_network.data_max - self.kratos_network.data_min)
            SReducedNormalized = SReduced

            # Obtain Clusters
            # -- Currenctly this function is quite useless because we are always using the classical SVD with 1 cluster.
            print("=== Calculating Cluster Bases ===")
            with contextlib.redirect_stdout(None):
                CB, kmeans_object = clustering.calcualte_snapshots_with_columns(
                    snapshot_matrix=SReducedNormalized,
                    number_of_clusters=num_clusters,
                    number_of_columns_in_the_basis=num_cluster_col,
                    truncation_tolerance=1e-5
                )

            print(f"-> Generated {len(CB)} cluster bases with shapes:")
            for b in CB:
                print(f'-> {b=} has {CB[b].shape=}')
                SPri = CB[b] @ CB[b].T @ SReduced
                print(f"Check Norm for BASE (All  ): {np.linalg.norm(SPri-SReduced)/np.linalg.norm(SReduced)}")
                print(f"Check Norm for BASE (Col-X): {np.linalg.norm(SPri[2]-SReduced[2])/np.linalg.norm(SReduced[2])}")

            # Obtain the reduced representations (that would be our G's)
            q, s, c, b = {}, {}, {}, {}
            total_respresentations = 0

            for i in range(num_clusters):
                q[i] = CB[i] @ CB[i].T @ SReduced[:,kmeans_object.labels_==i]
                b[i] = CB[i] @ CB[i].T
                c[i] = np.array([i for _ in range(q[i].shape[1])])
                s[i] = SReduced[:,kmeans_object.labels_==i]

            print("Total number of representations:", total_respresentations)

            temp_size = num_cluster_col
            grad_size = num_cluster_col

            qc = np.empty(shape=(data_rows,0))
            sc = np.empty(shape=(data_rows,0))
            cc = np.empty(shape=(0))

            for i in range(num_clusters):
                qc = np.concatenate((qc, q[i]), axis=1)
                sc = np.concatenate((sc, s[i]), axis=1)
                cc = np.concatenate((cc, c[i]), axis=0)

            nqc = self.kratos_network.normalize_data(qc)
            nsc = self.kratos_network.normalize_data(sc)

            print(f"{cc.shape=}")

            if self.config["train_model"]:
                self.kratos_network.train_network(self.autoencoder, nsc, cc, len(data_inputs), 100)
                print("Model Initialized")

            if self.config["save_model"]:
                self.autoencoder.save(self.kratos_network.model_name)

            print("Trying results calling model directly:")
            print("nsc shape:", nsc.shape)

            fh = 1
            self.nscorg = nsc
            self.TI = nsc[:,fh:fh+1]

            print("TI shape:", self.TI.shape)

            TP = self.kratos_network.predict_snapshot(self.autoencoder, self.TI)
            print(self.TI, TP)

            print("TI norm error", np.linalg.norm((TP)-(self.TI))/np.linalg.norm(self.TI))

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
            if self.config["use_reduced"]:
                self.sc = self.U.T @ np.array([self.S]).T
            else:
                self.sc = np.array([self.S]).T

            self.nsc = self.kratos_network.normalize_data(self.sc)

            self.NSPredict_d = self.kratos_network.predict_snapshot(self.autoencoder, self.nsc)

            self.q = self.kratos_network.predict_snapshot(self.kratos_network.encoder_model, self.nsc)

            print("Shape of nsc:", self.nsc.shape)

            # Check the prediction correctness:
            s__norm = np.linalg.norm(self.nsc)
            sp_norm = np.linalg.norm((self.NSPredict_d)-(self.nsc))

            if s__norm != 0:
                print("Norm error for this iteration", sp_norm/s__norm)
            else:
                print("Norm error  is 0")
                np.set_printoptions(threshold=sys.maxsize)

            # Impose prediction into initial solution
            # iter = 0
            # for node in computing_model_part.Nodes:
            #     node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X, self.NSPredict_d[iter+0])
            #     node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y, self.NSPredict_d[iter+1])
            #     iter+=2
            
            grad_enc_input = np.array([self.nsc.T[0]], dtype="float32")
            grad_dec_input = np.array([self.q.T[0]])

            nn_nodal_modes_enc = self.kratos_network.get_gradients(
                self.kratos_network.encoder_model, 
                [self.kratos_network.encoder_model],
                grad_enc_input,
                None
            )

            nn_nodal_modes_dec = self.kratos_network.get_gradients(
                self.kratos_network.decoder_model, 
                [self.kratos_network.decoder_model],
                grad_dec_input,
                None
            )

            print("U        shapes:", self.U.shape)
            print("gradient shapes:", nn_nodal_modes_enc.T.shape)
            print("gradient shapes:", nn_nodal_modes_dec.T.shape)

            # Set the nodal ROM basis
            # nodal_modes_e = self.U @ nn_nodal_modes_enc.T 
            
            if self.config["use_reduced"]:
                nodal_modes_d = self.U @ nn_nodal_modes_dec    # This is the same as (nn_nodal_modes_dec.T @ self.U.T).T
            else:
                nodal_modes_d = nn_nodal_modes_dec

            nodal_dofs = 2
            rom_dofs = 2

            # aux_e = KratosMultiphysics.Matrix(nodal_dofs, rom_dofs)
            aux_d = KratosMultiphysics.Matrix(nodal_dofs, rom_dofs)

            for node in computing_model_part.Nodes:
                node_id = node.Id-1
                for j in range(nodal_dofs):
                    for i in range(rom_dofs):
                        # aux_e[j,i] = nodal_modes_e[node_id*nodal_dofs+j][i]
                        aux_d[j,i] = nodal_modes_d[node_id*nodal_dofs+j][i]

                node.SetValue(KratosROM.ROM_BASIS,     aux_d)
                # node.SetValue(KratosROM.ROM_BASIS_DEC, aux_d)

            super().InitializeSolutionStep()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            snapshot = []
            for node in self._GetSolver().GetComputingModelPart().Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
            self.snapshots_matrix.append(snapshot)

            # computing_model_part = self._GetSolver().GetComputingModelPart().GetRootModelPart()

            # # Get the diplacement for this iteration
            # dq = computing_model_part.GetValue(KratosROM.ROM_SOLUTION_INCREMENT)

            # composed = self.NSPredict_e
            # composed[0][0] += dq[0]
            # composed[1][0] += dq[1]

            # self.NSPredict_d = self.kratos_network.predict_snapshot(self.kratos_network.decoder_model, composed)
            # # self.NSPredict_d = self.kratos_network.predict_snapshot(self.autoencoder, self.nsc)
            # SPRED = self.U @ self.NSPredict_d

            # s__norm = np.linalg.norm(self.S)
            # sp_norm = np.linalg.norm(SPRED-self.S)

            # if s__norm != 0:
            #     print("Norm error for this iteration", sp_norm/s__norm)
            # else:
            #     print("Norm error  is 0")
            #     np.set_printoptions(threshold=sys.maxsize)
            #     print(SPRED)

            # iter = 0
            # for node in computing_model_part.Nodes:
            #     node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X, SPRED[iter+0])
            #     node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y, SPRED[iter+1])
            #     iter+=2

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

    with open("ProjectParameters.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    # analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    # analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KratosMultiphysics.Model()
    simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    
    # simulation = analysis_stage_class(global_model, parameters)
    simulation.Run()
