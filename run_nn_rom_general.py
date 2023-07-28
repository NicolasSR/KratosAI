import os
import sys

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import h5py
import numpy as np
import pandas as pd

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


def CreateRomAnalysisInstance(cls, global_model, parameters):
    class RomAnalysis(cls):

        def __init__(self,global_model, parameters):
            super().__init__(global_model, parameters)
            self.override_first_iter = False
            self.snapshots_matrix = []
            self.first_iter=False
            """ Initialize instance variables needed for the encoder and decoder.
            e.g. for POD: self.phi = None
            """
        
        def _ConfigureEncoderDecoder(self):
            """Define method to obtain the matrices or any other data to define the encoder and decoder
            e.g. for POD: self.phi=np.load('phi.npy')
            """

        def _EncodeSnapshot(self, u):
            """Define method to encode a given snapshot
            e.g. for POD: q=np.linalg.matmul(self.phi, u)
                          return q
            """
            
        def _DecodeSnapshot(self, q):
            """Define method to decode a given snapshot
            e.g. for POD: q=np.linalg.matmul(self.phi, u)
                          return u
            """

        def _GetDecoderGradient(self, q):
            """Define the gradient of the decoder given a certain q
            e.g. for POD: decoder_grad=self.phi
                          return decoder_grad
            """

        def _CreateSolver(self):
            """ Create the Solver (and create and import the ModelPart if it is not alread in the model) """

            # Get the ROM settings from the RomParameters.json input file
            with open('RomParameters.json') as rom_parameters:
                self.rom_parameters = KratosMultiphysics.Parameters(rom_parameters.read())

            # HROM operation flags
            self.rom_basis_process_list_check = False
            self.rom_basis_output_process_check = False

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

            # Get the input snapshot from the state after the last iteration (or default state)
            S = []
            for node in computing_model_part.Nodes:
                S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                S.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
           
            S = np.array([S])

            SPred = self.autoencoder.predict_snapshot(S) # predict_snaptshot already normalizes and denormalizes

            q = self._EncodeSnapshot(S)
            SPred = self._DecodeSnapshot(q)

            # Check the prediction correctness:
            s__norm = np.linalg.norm(S, axis=1)
            sp_norm = np.linalg.norm(SPred-S, axis=1)

            if s__norm != 0:
                print("Norm error for this iteration", sp_norm/s__norm)
            else:
                print("Norm error is 0")
                np.set_printoptions(threshold=sys.maxsize)

            nn_nodal_modes=self._GetDecoderGradient(q)

            print("Gradient shape:", nn_nodal_modes.shape)

            nodal_dofs = nn_nodal_modes.shape[0]
            rom_dofs = nn_nodal_modes.shape[1]

            aux_d = KratosMultiphysics.Matrix(nodal_dofs, rom_dofs)

            for node in computing_model_part.Nodes:
                node_id = node.Id-1
                for j in range(nodal_dofs):
                    for i in range(rom_dofs):
                        aux_d[j,i] = nn_nodal_modes[node_id*nodal_dofs+j][i]

                node.SetValue(KratosROM.ROM_BASIS, aux_d)

            super().InitializeSolutionStep()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            snapshot = []
            for node in self._GetSolver().GetComputingModelPart().Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
            self.snapshots_matrix.append(snapshot)

        def ModifyAfterSolverInitialize(self):
            super().ModifyAfterSolverInitialize()

            self._ConfigureEncoderDecoder()

        def Finalize(self):
            # This calls the physics Finalize
            super().Finalize()

            np.save("Custom_ROM_Snapshots.npy",self.snapshots_matrix)

    return RomAnalysis(global_model, parameters)

if __name__ == "__main__":

    with open("ProjectParameters_custom_rom.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    # analysis_stage_class = StructuralMechanicsAnalysis

    global_model = KratosMultiphysics.Model()
    simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
  
    simulation.Run()
