import os
import sys
import time
import importlib
import numpy as np

import KratosMultiphysics

import tensorflow as tf

from utils.kratos_simulation import KratosSimulator
from networks.conv2d_ae_factory import  Conv2D_AE_Factory
from networks.dense_ae_factory import Dense_AE_Factory

def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters, autoencoder, data_normalizer, print_matrices):
    class AnalysisStageWithFlush(cls):

        def __init__(self, model,project_parameters, autoencoder, data_normalizer, print_matrices=False, flush_frequency=10.0):
            super().__init__(model, project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()

            self.autoencoder = autoencoder
            self.data_normalizer = data_normalizer
            self.modelpart = self._GetSolver().GetComputingModelPart()
            self.var_utils = KratosMultiphysics.VariableUtils()

            self.space = KratosMultiphysics.UblasSparseSpace()
            self.strategy = self._GetSolver()._GetSolutionStrategy()
            self.buildsol = self._GetSolver()._GetBuilderAndSolver()
            self.scheme = self._GetSolver()._GetScheme()

            self.print_matrices=print_matrices
            self.snapshots_nn_matrix=[]
            self.snapshots_fom_matrix=[]
            self.residuals_nn_matrix=[]
            self.residuals_fom_matrix=[]

        def Initialize(self):
            super().Initialize()
            sys.stdout.flush()

            self.initial_positions = self.var_utils.GetCurrentPositionsVector(self.modelpart.Nodes, 2)
            print(self.initial_positions[:10])



        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()

            # foo5 = self.var_utils.GetSolutionStepValuesVector(self.modelpart.Nodes, KratosMultiphysics.DISPLACEMENT, 0, 2)
            # foo6 = self.var_utils.GetCurrentPositionsVector(self.modelpart.Nodes, 2)
            # print(foo5[:10])
            # print(foo6[:10])


        def OutputSolutionStep(self):
            
            # Modify displacement and positions with those from the NN
            fom_displacement = self.var_utils.GetSolutionStepValuesVector(self.modelpart.Nodes, KratosMultiphysics.DISPLACEMENT, 0, 2)
            fom_positions = self.var_utils.GetCurrentPositionsVector(self.modelpart.Nodes, 2)
            fom_reactions = self.var_utils.GetSolutionStepValuesVector(self.modelpart.Nodes, KratosMultiphysics.REACTION, 0, 2)
            # print(fom_displacement[:10])
            # print(fom_positions[:10])

            b_fom = self.strategy.GetSystemVector()
            self.space.SetToZeroVector(b_fom)

            # self.buildsol.Build(self.scheme, self.modelpart, A, b)
            self.buildsol.BuildRHS(self.scheme, self.modelpart, b_fom)
            b_fom=-1*np.array(b_fom, copy=False)
            
            fom_displacement_np = np.expand_dims(np.array(fom_displacement, copy=True),axis=0)
            fom_displacement_normal = self.data_normalizer.process_raw_to_input_format(fom_displacement_np)
            nn_displacement_normal = self.autoencoder(fom_displacement_normal, training=False).numpy()
            nn_displacement = self.data_normalizer.process_input_to_raw_format(nn_displacement_normal)[0]
            nn_positions = self.initial_positions + nn_displacement

            self.var_utils.SetSolutionStepValuesVector(self.modelpart.Nodes, KratosMultiphysics.DISPLACEMENT, nn_displacement, 2)
            self.var_utils.SetCurrentPositionsVector(self.modelpart.Nodes, nn_positions)

            b = self.strategy.GetSystemVector()
            self.space.SetToZeroVector(b)

            # self.buildsol.Build(self.scheme, self.modelpart, A, b)
            self.buildsol.BuildRHS(self.scheme, self.modelpart, b)
            b=-1*np.array(b, copy=False)

            self.var_utils.SetSolutionStepValuesVector(self.modelpart.Nodes, KratosMultiphysics.REACTION, b, 2)

            # foo1 = self.var_utils.GetSolutionStepValuesVector(self.modelpart.Nodes, KratosMultiphysics.DISPLACEMENT, 0, 2)
            # foo2 = self.var_utils.GetCurrentPositionsVector(self.modelpart.Nodes, 2)
            # print(foo1[:10])
            # print(foo2[:10])

            # Call output processes (we are interested in GID output)
            super().OutputSolutionStep()

            # Put the originals back for the enxt step.
            self.var_utils.SetSolutionStepValuesVector(self.modelpart.Nodes, KratosMultiphysics.DISPLACEMENT, fom_displacement, 2)
            self.var_utils.SetCurrentPositionsVector(self.modelpart.Nodes, fom_positions)
            self.var_utils.SetSolutionStepValuesVector(self.modelpart.Nodes, KratosMultiphysics.REACTION, fom_reactions, 2)

            if self.print_matrices:
                self.snapshots_nn_matrix.append(fom_displacement)
                self.snapshots_fom_matrix.append(nn_displacement)
                self.residuals_nn_matrix.append(b)
                self.residuals_fom_matrix.append(b_fom)

            # foo3 = self.var_utils.GetSolutionStepValuesVector(self.modelpart.Nodes, KratosMultiphysics.DISPLACEMENT, 0, 2)
            # foo4 = self.var_utils.GetCurrentPositionsVector(self.modelpart.Nodes, 2)
            # print(foo3[:10])
            # print(foo4[:10])

        def FinalizeSolutionStep(self):

            super().FinalizeSolutionStep()

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now

        def Finalize(self):
            super().Finalize()

            if self.print_matrices:
                np.save("NN_snaps.npy", np.array(self.snapshots_nn_matrix, copy=False))
                np.save("FOM_snaps.npy", np.array(self.snapshots_fom_matrix, copy=False))
                np.save("NN_residuals.npy", np.array(self.residuals_nn_matrix, copy=False))
                np.save("FOM_residuals.npy", np.array(self.residuals_fom_matrix, copy=False))


    return AnalysisStageWithFlush(global_model, parameters, autoencoder, data_normalizer, print_matrices)


def CreateAnalysisStageBasic(cls, global_model, parameters):
    class AnalysisStageBasic(cls):

        def __init__(self, model,project_parameters, flush_frequency=10.0):
            super().__init__(model, project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()

        def Initialize(self):
            super().Initialize()
            sys.stdout.flush()

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now


    return AnalysisStageBasic(global_model, parameters)


class NN_FOM_Reconstructor():

    def __init__(self, working_path, model_path, best, print_matrices):
        self.working_path=working_path
        self.model_path=working_path+model_path
        if best=='x':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_x_')
        elif best=='r':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_r_')
        elif best is None:
            self.model_weights_path=self.model_path
            self.model_weights_filename='model_weights.h5'
        else:
            print('Value for --best argument is not recognized. Terminating')
            exit()

        with open(self.model_path+"ae_config.npy", "rb") as ae_config_file:
            self.ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()
        print(self.ae_config)
        self.dataset_path=working_path+self.ae_config['dataset_path']

        self.print_matrices=print_matrices

    def network_factory_selector(self, nn_type):
        if 'conv2d' in nn_type:
            return Conv2D_AE_Factory()
        elif 'dense' in nn_type:
            return Dense_AE_Factory()
        else:
            print('No valid network type was selected')
            return None
        
    def get_last_best_filename(self, model_weights_path, prefix):
        matching_files = [file for file in os.listdir(model_weights_path) if file.startswith(prefix)]
        highest_filename = sorted(matching_files, key=lambda x: int(x[len(prefix):][:-len('.h5')]))[-1]
        return highest_filename

    def execute_simulation(self):

        # Select the network to use
        network_factory = self.network_factory_selector(self.ae_config["nn_type"])

        # Select the type of preprocessimg (normalisation). This also decides if cropping the snapshot is needed
        self.data_normalizer=network_factory.normalizer_selector(self.working_path, self.ae_config)

        # Create a fake Analysis stage to calculate the predicted residuals
        self.residual_scale_factor=np.load(self.dataset_path+'residual_scale_factor.npy')
        self.kratos_simulation = KratosSimulator(self.working_path, self.ae_config, self.residual_scale_factor)
        crop_mat_tf, crop_mat_scp = self.kratos_simulation.get_crop_matrix()
        del self.kratos_simulation

        S_flat_orig = np.load(self.dataset_path+'FOM.npy')
        print('Shape S_flat_orig: ', S_flat_orig.shape)

        self.data_normalizer.configure_normalization_data(S_flat_orig, crop_mat_tf, crop_mat_scp)

        S = self.data_normalizer.process_raw_to_input_format(S_flat_orig)

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        self.autoencoder, encoder, decoder = network_factory.define_network(S, self.ae_config)
        self.autoencoder.load_weights(self.model_weights_path+self.model_weights_filename)
        self.autoencoder.set_config_values_eval(self.data_normalizer)

        with open(self.dataset_path+"ProjectParameters_nn_fom_reconstruction.json", 'r') as parameter_file:
            parameters = KratosMultiphysics.Parameters(parameter_file.read())

        analysis_stage_module_name = parameters["analysis_stage"].GetString()
        analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
        analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

        analysis_stage_module = importlib.import_module(analysis_stage_module_name)
        analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

        global_model = KratosMultiphysics.Model()
        simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters, self.autoencoder, self.data_normalizer, self.print_matrices)
        simulation.Run()


class FOM_Simulator():

    def __init__(self, working_path, model_path):
        self.working_path=working_path
        self.model_path=working_path+model_path

        with open(self.model_path+"ae_config.npy", "rb") as ae_config_file:
            self.ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()
        print(self.ae_config)
        self.dataset_path=working_path+self.ae_config['dataset_path']

    def execute_simulation(self):

        with open(self.dataset_path+"ProjectParameters_nn_fom_reconstruction.json", 'r') as parameter_file:
            parameters = KratosMultiphysics.Parameters(parameter_file.read())

        analysis_stage_module_name = parameters["analysis_stage"].GetString()
        analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
        analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

        analysis_stage_module = importlib.import_module(analysis_stage_module_name)
        analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

        global_model = KratosMultiphysics.Model()
        simulation = CreateAnalysisStageBasic(analysis_stage_class, global_model, parameters)
        simulation.Run()
