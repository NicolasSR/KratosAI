import numpy as np

import KratosMultiphysics
import KratosMultiphysics.RomApplication as KratosROM

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

from KratosMultiphysics.RomApplication.networks import gradient_shallow as gradient_shallow_ae
from KratosMultiphysics.RomApplication import python_solvers_wrapper_rom as solver_wrapper

from KratosMultiphysics.RomApplication import new_python_solvers_wrapper_rom
from KratosMultiphysics.RomApplication.hrom_training_utility import HRomTrainingUtility
from KratosMultiphysics.RomApplication.calculate_rom_basis_output_process import CalculateRomBasisOutputProcess

from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition


def CreateRomAnalysisInstance(cls, global_model, parameters):
    class RomAnalysis(cls):

        def __init__(self,global_model, parameters):
            super().__init__(global_model, parameters)

        def PrintAnalysisStageProgressInformation(self):
            pass

        def _CreateSolver(self):
            """ Create the Solver (and create and import the ModelPart if it is not alread in the model) """

            # Get the ROM settings from the RomParameters.json input file
            with open('RomParameters.json') as rom_parameters:
                self.rom_parameters = KratosMultiphysics.Parameters(rom_parameters.read())

            # Set the ROM settings in the "solver_settings" of the solver introducing the physics
            self.project_parameters["solver_settings"].AddValue("rom_settings", self.rom_parameters["rom_settings"])

            # HROM operations flags
            self.rom_basis_process_list_check = True
            self.rom_basis_output_process_check = True
            self.run_hrom = self.rom_parameters["run_hrom"].GetBool() if self.rom_parameters.Has("run_hrom") else False
            self.train_hrom = self.rom_parameters["train_hrom"].GetBool() if self.rom_parameters.Has("train_hrom") else False
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

        def ModifyAfterSolverInitialize(self):
            """Here is where the ROM_BASIS is imposed to each node"""
            super().ModifyAfterSolverInitialize()

            # Get the model part where the ROM is to be applied
            computing_model_part = self._GetSolver().GetComputingModelPart().GetRootModelPart()

            # Matrix to compare results
            self.snapshots_matrix = []

            # Store 1st Level ROM residuals
            self.residuals_matrix = []

            # Set ROM basis
            nodal_modes = self.rom_parameters["nodal_modes"]
            nodal_dofs = len(self.project_parameters["solver_settings"]["rom_settings"]["nodal_unknowns"].GetStringArray())
            rom_dofs = self.project_parameters["solver_settings"]["rom_settings"]["number_of_rom_dofs"].GetInt()

            # Set the nodal ROM basis
            aux = KratosMultiphysics.Matrix(nodal_dofs, rom_dofs)
            for node in computing_model_part.Nodes:
                node_id = str(node.Id)
                for j in range(nodal_dofs):
                    for i in range(rom_dofs):
                        aux[j,i] = nodal_modes[node_id][j][i].GetDouble()
                node.SetValue(KratosROM.ROM_BASIS, aux)

            # Check for HROM stages
            if self.train_hrom:
                # Create the training utility to calculate the HROM weights
                self.__hrom_training_utility = HRomTrainingUtility(
                    self._GetSolver(),
                    self.rom_parameters
                )

            elif self.run_hrom:
                # Set the HROM weights in elements and conditions
                hrom_weights_elements = self.rom_parameters["elements_and_weights"]["Elements"]
                for key,value in zip(hrom_weights_elements.keys(), hrom_weights_elements.values()):
                    computing_model_part.GetElement(int(key)+1).SetValue(KratosROM.HROM_WEIGHT, value.GetDouble()) #FIXME: FIX THE +1

                hrom_weights_condtions = self.rom_parameters["elements_and_weights"]["Conditions"]
                for key,value in zip(hrom_weights_condtions.keys(), hrom_weights_condtions.values()):
                    computing_model_part.GetCondition(int(key)+1).SetValue(KratosROM.HROM_WEIGHT, value.GetDouble()) #FIXME: FIX THE +1

        def FinalizeSolutionStep(self):
            # Call the HROM training utility to append the current step residuals
            # Note that this needs to be done prior to the other processes to avoid unfixing the BCs
            if self.train_hrom:
                self.__hrom_training_utility.AppendCurrentStepResiduals()

            # Write the snapshot to compare results
            snapshot = []
            for node in self._GetSolver().GetComputingModelPart().GetRootModelPart().Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
            self.snapshots_matrix.append(snapshot)

            # Store 1st Level ROM residuals
            self.residuals_matrix.append(self.model.GetModelPart("Structure").GetValue(KratosROM.ROM_RESIDUAL))

            # #FIXME: Make this optional. This must be a process
            # # Project the ROM solution onto the visualization modelparts
            # if self.run_hrom:
            #     model_part_name = self._GetSolver().settings["model_part_name"].GetString()
            #     visualization_model_part = self.model.GetModelPart("{}.{}Visualization".format(model_part_name, model_part_name))
            #     KratosROM.RomAuxiliaryUtilities.ProjectRomSolutionIncrementToNodes(
            #         self.rom_parameters["rom_settings"]["nodal_unknowns"].GetStringArray(),
            #         visualization_model_part)

            # This calls the physics FinalizeSolutionStep (e.g. BCs)
            super().FinalizeSolutionStep()

        def Finalize(self):
            # This calls the physics Finalize
            super().Finalize()

            # Once simulation is completed, calculate and save the HROM weights
            if self.train_hrom:
                self.__hrom_training_utility.CalculateAndSaveHRomWeights()
                self.__hrom_training_utility.CreateHRomModelParts()

            # Write the snapshot to compare results
            np.save("ROM.npy",self.snapshots_matrix)

            # Store 1st Level ROM residuals
            print(f"Saving residuals {np.array(self.residuals_matrix).shape=}")
            np.save("ROM_RESIDUALS.npy",self.residuals_matrix)

    return RomAnalysis(global_model, parameters)
