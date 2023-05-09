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
from KratosMultiphysics.RomApplication import kratos_io as kratos_io
from KratosMultiphysics.RomApplication import clustering as clustering
from KratosMultiphysics.RomApplication import python_solvers_wrapper_rom as solver_wrapper

from KratosMultiphysics.RomApplication import new_python_solvers_wrapper_rom
from KratosMultiphysics.RomApplication.hrom_training_utility import HRomTrainingUtility
from KratosMultiphysics.RomApplication.calculate_rom_basis_output_process import CalculateRomBasisOutputProcess

from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

# import networks.dense_residual_ae as dense_residual_ae
from networks.conv2d_residual_ae import  Conv2D_Residual_AE
from utils.normalizers import Conv2D_AE_Normalizer_ChannelRange, Conv2D_AE_Normalizer_FeatureStand

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
            self.first_iter=False

        def prepare_input(self, data_inputs_files):
            S=np.load(data_inputs_files)[:,4:]
            return S
        
        def normalizer_selector(self, normalization_strategy):
            if normalization_strategy == 'channel_range':
                return Conv2D_AE_Normalizer_ChannelRange()
            elif normalization_strategy == 'feature_stand':
                return Conv2D_AE_Normalizer_FeatureStand()
            else:
                print('Normalization strategy is not valid')
                return None

        ### NN ##
        def _CreateModel(self):
            # List of files to read from
            data_inputs_files = "datasets_rommanager/FOM.npy"

            S = self.prepare_input(data_inputs_files)
            print('S shape:', S.shape)

            # external_sets_length=int(S.shape[1]/6)
            # S_test_low=S[:,:external_sets_length]
            # S_test_high=S[:,-external_sets_length:]
            # S=S[:,external_sets_length:-external_sets_length]
            # print('Shape S: ', S.shape)
            # print('Shape S_test_high: ', S_test_high.shape)
            # print('Shape S_test_low: ', S_test_low.shape)
            
            # Some configuration
            self.config = {
                "load_model":       True,
                "train_model":      False,
                "save_model":       False,
                "print_results":    False,
                "use_reduced":      False,
            }

            # Select the netwrok to use
            self.kratos_network = Conv2D_Residual_AE()

            model_path="saved_models_conv2d/"

            print('======= Loading saved ae config =======')
            with open(model_path+"ae_config.npy", "rb") as ae_config_file:
                ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()

            # Normalize the snapshots according to the desired normalization mode
            # SNorm=S.copy()
            # SNorm = self.normalize_snapshots_data(SNorm, ae_config["normalization_strategy"])
            data_normalizer=self.normalizer_selector(ae_config["normalization_strategy"])
            data_normalizer.configure_normalization_data(S)

            S_norm=data_normalizer.normalize_data(S)
            S_norm_2D=data_normalizer.reorganize_into_channels(S_norm)

            # Load the autoencoder model
            print('======= Instantiating new autoencoder =======')
            print(ae_config)
            self.autoencoder, self.encoder, self.decoder = self.kratos_network.define_network(S_norm_2D, ae_config)

            print('======= Loading saved weights =======')
            self.autoencoder.load_weights(model_path+'model_weights.h5')
            # self.encoder.load_weights(model_path+'encoder_model_weights.h5')
            # self.decoder.load_weights(model_path+'decoder_model_weights.h5')

            self.autoencoder.set_config_values_eval(data_normalizer)

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
                # S = [2.84322712e-06, -4.55168069e-06, -2.91819193e-06, -4.67132520e-06,
                #  5.50219914e-06, -1.77873238e-05, -5.73516507e-06, -1.82784288e-05,
                #  8.04057521e-06, -4.01316202e-05, -8.43360036e-06, -4.13274705e-05,
                #  1.04315097e-05, -7.20639649e-05, -1.09825867e-05, -7.43295393e-05,
                #  1.26398376e-05, -1.14051435e-04, -1.33402261e-05, -1.17773885e-04,
                #  1.46219209e-05, -1.66500193e-04, -1.54487345e-05, -1.72063521e-04,
                #  1.63808212e-05, -2.27363868e-04, -1.70845500e-05, -2.32125477e-04,
                #  1.77071633e-05, -2.90368808e-04, -1.83462357e-05, -2.94128369e-04,
                #  1.86508592e-05, -3.53999347e-04, -1.92738243e-05, -3.56664759e-04,
                #  1.92631152e-05, -4.17124598e-04, -1.99059630e-05, -4.18688927e-04,
                #  1.95893029e-05, -4.78920955e-04, -2.02778811e-05, -4.79435219e-04,
                #  1.96646313e-05, -5.38802923e-04, -2.03895760e-05, -5.38781761e-04]
                
                # Force = 2e3

                # S = [0.0008346 ,
                #  -0.00136451, -0.00088975, -0.00138119,  0.00157316, -0.00533982,
                #  -0.0017938 , -0.00542661,  0.0021886 , -0.01205465, -0.00275256,
                #  -0.01228803,  0.00262913, -0.02164873, -0.00380276, -0.02211904,
                #  0.00284205, -0.03425728, -0.00497628, -0.03506712,  0.00277643,
                #  -0.04999675, -0.00629531, -0.05125497,  0.00243684, -0.06825138,
                #  -0.00764496, -0.06917333,  0.00185296, -0.08713587, -0.00899139,
                #  -0.08767766,  0.00108142, -0.1061973 , -0.01030751, -0.1063499 ,
                #  0.00017523, -0.12509937, -0.01156975, -0.12487958, -0.00082028,
                #  -0.14359828, -0.01275892, -0.14303933, -0.0018692 , -0.16152126,
                #  -0.01385852, -0.16079624]

                # Force = 6e5 (index 299)

                S = [0.0013652 ,
                 -0.00226417, -0.00149219, -0.00227104,  0.00252582, -0.00886645,
                 -0.0030557 , -0.00894449,  0.00338624, -0.02001793, -0.00480691,
                 -0.02026828,  0.00381294, -0.03594093, -0.00685648, -0.03649229,
                 0.00367308, -0.0568484 , -0.00930528, -0.05785561,  0.00284185,
                 -0.08292122, -0.01223291, -0.0845571 ,  0.00133162, -0.11313267,
                 -0.01539762, -0.11410689, -0.00070657, -0.14435792, -0.01867491,
                 -0.14461606, -0.00313355, -0.17585217, -0.02197819, -0.17539989,
                 -0.00582079, -0.20706574, -0.02523098, -0.20595145, -0.0086593 ,
                 -0.23760242, -0.02836983, -0.23590054, -0.01156249, -0.26718278,
                 -0.03135038, -0.26519955]
                
                # Force = 6e5 (index 499)
                
                self.first_iter=False
            
            else:
                # S = []
                # for node in computing_model_part.Nodes:
                #     # if not node.IsFixed(KratosMultiphysics.DISPLACEMENT_X):
                #     #     S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                #     # else:
                #     #     print('Fixed x')
                #     # if not node.IsFixed(KratosMultiphysics.DISPLACEMENT_Y):
                #     #     S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
                #     # else:
                #     #     print('Fixed y')
                #     if node.Id >= 3:
                #         # S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                #         # S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
                #         S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X)*1e3)
                #         S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y)*1e3)

                S = []
                for node in computing_model_part.Nodes:
                    if node.Id >= 3:
                        S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                        S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
           

            S = np.array([S])

            print('Shape of S', S.shape)
            print(S)

            # SNorm=self.autoencoder.normalize_data(S.copy())
            SPred = self.autoencoder.predict_snapshot(S) # predict_snaptshot already normalizes and denormalizes
            # SPred=self.autoencoder.denormalize_data(SNPred)

            q = self.kratos_network.encode_snapshot(self.encoder, self.autoencoder, S)

            # print("Shape of SNorm:", SNorm.shape)

            # Check the prediction correctness:
            # s__norm = np.linalg.norm(SNorm)
            # sp_norm = np.linalg.norm((SNPred)-(SNorm))

            s__norm = np.linalg.norm(S, axis=1)
            sp_norm = np.linalg.norm(SPred-S, axis=1)

            # print(s__norm)
            # print(sp_norm)

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

            # print("gradient shapes:", nn_nodal_modes.shape)
            # print(nn_nodal_modes)
            
            nodal_dofs = 2
            rom_dofs = q.shape[1]

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
