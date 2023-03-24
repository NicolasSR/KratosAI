import os
import sys
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import h5py
import numpy as np
import pandas as pd

import keras
import tensorflow as tf

from keras import layers
from itertools import repeat

import importlib
import contextlib

import KratosMultiphysics
import KratosMultiphysics.RomApplication as KratosROM

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

# from KratosMultiphysics.RomApplication.networks import gradient_shallow as gradient_shallow_ae
import networks.dense_residual_ae as dense_residual_ae
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
            self.first_iter=True

        def prepare_input(self, data_inputs_files):
            S = None
            for s in data_inputs_files:
                a = np.load(s)
                if S is None:
                    S = a
                else:
                    S = np.concatenate((S, a), axis=0)
            S=S.T
            S=S[4:,:]

            return S
        
        def normalize_snapshots_data(self, S, normalization_strategy):
            if normalization_strategy == 'per_feature':
                feat_means = []
                feat_stds = []
                print('Normalizing each feature in S')
                S_df = pd.DataFrame(S.T)
                for i in range(len(S_df.columns)):
                    feat_means.append(S_df[i].mean())
                    feat_stds.append(S_df[i].std())
                self.autoencoder.set_normalization_data(normalization_strategy, (feat_means, feat_stds))
            elif normalization_strategy == 'global':
                print('Applying global min-max normalization on S')
                data_min = np.min(S)
                data_max = np.max(S)
                self.autoencoder.set_normalization_data(normalization_strategy, (data_min, data_max))
            else:
                print('No normalization')
            SNorm = self.autoencoder.normalize_data(S.T)
            return SNorm.T

        ### NN ##
        def _CreateModel(self):
            # List of files to read from
            data_inputs_files = ["datasets/FOM.npy"]

            S = self.prepare_input(data_inputs_files)
            print('S shape:', S.shape)

            external_sets_length=int(S.shape[1]/6)
            S_test_low=S[:,:external_sets_length]
            S_test_high=S[:,-external_sets_length:]
            S=S[:,external_sets_length:-external_sets_length]
            print('Shape S: ', S.shape)
            print('Shape S_test_high: ', S_test_high.shape)
            print('Shape S_test_low: ', S_test_low.shape)
            
            # Some configuration
            self.config = {
                "load_model":       True,
                "train_model":      False,
                "save_model":       False,
                "print_results":    False,
                "use_reduced":      False,
            }

            # Select the netwrok to use
            self.kratos_network = dense_residual_ae.GradientShallow()

            print('======= Loading saved ae config =======')
            with open("saved_models/ae_config.npy", "rb") as ae_config_file:
                ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()

            # Load the autoencoder model
            print('======= Instantiating new autoencoder =======')
            self.autoencoder, self.encoder, self.decoder = self.kratos_network.define_network(S, ae_config)

            print('======= Loading saved weights =======')
            self.autoencoder.load_weights('saved_models/model_weights.h5')
            self.encoder.load_weights('saved_models/encoder_model_weights.h5')
            self.decoder.load_weights('saved_models/decoder_model_weights.h5')

            # Normalize the snapshots according to the desired normalization mode
            SNorm=S.copy()
            SNorm = self.normalize_snapshots_data(SNorm, ae_config["normalization_strategy"])


            """ # Obtain Clusters
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
                SPri = CB[b] @ CB[b].T @ S
                print(f"Check Norm for BASE (All  ): {np.linalg.norm(SPri-S)/np.linalg.norm(S)}")
                print(f"Check Norm for BASE (Col-X): {np.linalg.norm(SPri[2]-S[2])/np.linalg.norm(S[2])}")

            # Obtain the reduced representations (that would be our G's)
            q, s, c, b = {}, {}, {}, {}
            total_respresentations = 0

            for i in range(num_clusters):
                q[i] = CB[i] @ CB[i].T @ S[:,kmeans_object.labels_==i]
                b[i] = CB[i] @ CB[i].T
                c[i] = np.array([i for _ in range(q[i].shape[1])])
                s[i] = S[:,kmeans_object.labels_==i]

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

            print("TI norm error", np.linalg.norm((TP)-(self.TI))/np.linalg.norm(self.TI)) """

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
            
            if self.first_iter:
                S=[
                 7.41368540e-03, -1.58596070e-02, -1.17056207e-02, -1.39700617e-02,
                 8.72809942e-03, -6.10772234e-02, -2.76358538e-02, -5.55512169e-02,
                 -8.52063338e-04, -1.34367120e-01, -5.23636480e-02, -1.24024755e-01,
                 -2.50534323e-02, -2.33690737e-01, -8.96987097e-02, -2.18210254e-01,
                 -6.65904266e-02, -3.56930785e-01, -1.42580957e-01, -3.36713143e-01,
                 -1.27237001e-01, -5.02143015e-01, -2.13047333e-01, -4.78165671e-01,
                 -2.04748696e-01, -6.61898171e-01, -2.94903615e-01, -6.28291137e-01,
                 -2.90175914e-01, -8.19850190e-01, -3.82508741e-01, -7.77918421e-01,
                 -3.80033110e-01, -9.73827824e-01, -4.73120294e-01, -9.24972434e-01,
                 -4.71685222e-01, -1.12278060e+00, -5.64632088e-01, -1.06835024e+00,
                 -5.63213426e-01, -1.26629278e+00, -6.55473680e-01, -1.20752549e+00,
                 -6.53235019e-01, -1.40429161e+00, -7.45267213e-01, -1.34342165e+00]
                
                # S=[
                #  0.00257479, -0.00442512,
                #  -0.0029867, -0.00434256, 0.00453559, -0.01734126, -0.00632606, -0.01718823,
                #  0.00546122, -0.0391235, -0.01046927, -0.03897886, 0.00487612, -0.07012932,
                #  -0.01585522, -0.07014718, 0.00232162, -0.11068401, -0.02289253, -0.11109575,
                #  -0.0026172, -0.16104522, -0.03192278, -0.1621503, -0.00988217, -0.21917003,
                #  -0.04224098, -0.21852432, -0.01870639, -0.27902319, -0.05329543, -0.27660969,
                #  -0.02861475, -0.33921241, -0.06473496, -0.33513776, -0.03917904, -0.39873466,
                #  -0.07624921, -0.39318497, -0.05004142, -0.45688366, -0.08757942, -0.45008731,
                #  -0.06091876, -0.5131715, -0.09857205, -0.50580146]
                # Force = 2e6
                
                self.first_iter=False
            
            else:
                S = []
                for node in computing_model_part.Nodes:
                    # if not node.IsFixed(KratosMultiphysics.DISPLACEMENT_X):
                    #     S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                    # else:
                    #     print('Fixed x')
                    # if not node.IsFixed(KratosMultiphysics.DISPLACEMENT_Y):
                    #     S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
                    # else:
                    #     print('Fixed y')
                    if node.Id >= 3:
                        # S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                        # S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
                        S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X)*1e3)
                        S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y)*1e3)

            S = np.array([S])

            print('Shape of S', S.shape)
            print(S)

            SNorm=self.autoencoder.normalize_data(S.copy())
            SNPred = self.kratos_network.predict_snapshot(self.autoencoder, SNorm)
            SPred=self.autoencoder.denormalize_data(SNPred)

            q = self.kratos_network.predict_snapshot(self.encoder, SNorm)

            print("Shape of SNorm:", SNorm.shape)

            # Check the prediction correctness:
            # s__norm = np.linalg.norm(SNorm)
            # sp_norm = np.linalg.norm((SNPred)-(SNorm))

            s__norm = np.linalg.norm(S)
            sp_norm = np.linalg.norm((SPred)-(S))


            if s__norm != 0:
                print("Norm error for this iteration", sp_norm/s__norm)
            else:
                print("Norm error  is 0")
                np.set_printoptions(threshold=sys.maxsize)

            nn_nodal_modes = self.kratos_network.calculate_gradients(
                self.decoder, 
                self.autoencoder,
                q
            )

            # print("gradient shapes:", nn_nodal_modes.shape)
            # print(nn_nodal_modes)

            nn_nodal_modes=np.concatenate((np.zeros((4,nn_nodal_modes.shape[-1])),nn_nodal_modes),axis=0)

            print("gradient shapes:", nn_nodal_modes.shape)
            # print(nn_nodal_modes)
            
            nodal_dofs = 2
            rom_dofs = 1

            aux_d = KratosMultiphysics.Matrix(nodal_dofs, rom_dofs)

            for node in computing_model_part.Nodes:
                node_id = node.Id-1
                for j in range(nodal_dofs):
                    for i in range(rom_dofs):
                        # aux_e[j,i] = nodal_modes_e[node_id*nodal_dofs+j][i]
                        aux_d[j,i] = nn_nodal_modes[node_id*nodal_dofs+j][i]

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

    # analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    # analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KratosMultiphysics.Model()
    simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
    
    # simulation = analysis_stage_class(global_model, parameters)
    simulation.Run()
