import numpy as np

import KratosMultiphysics
import KratosMultiphysics.RomApplication as KratosROM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis


def CreateRomAnalysisInstance(cls, global_model, parameters):
    class StructuralMechanicsAnalysisFOM(StructuralMechanicsAnalysis):
        def __init__(self,model,project_parameters):
            super().__init__(model,project_parameters)
        
        def ModifyInitialGeometry(self):
            super().ModifyInitialGeometry()
            # self.inapshots_matrix = []
            # self.snapshots_matrix = []
            # self.residuals_matrix = []
            # self.pointload_matrix = []
            # self.fixedNodes = []
            # self.main_model_part = self.model.GetModelPart("Structure")

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()
            # snapshot = []
            # for node in self.main_model_part.Nodes:
            #     snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
            #     snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
            # self.inapshots_matrix.append(snapshot)
        
        def FinalizeSolutionStep(self):
            # pointload = []
            # for condition in self.main_model_part.Conditions:
            #     pointload.append(condition.GetValue(SMA.POINT_LOAD))
            # self.pointload_matrix.append(pointload)

            super().FinalizeSolutionStep()
            # snapshot = []
            # for node in self.main_model_part.Nodes:
            #     snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
            #     snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
            # self.snapshots_matrix.append(snapshot)

            # strategy  = self._GetSolver()._GetSolutionStrategy()
            # buildsol  = self._GetSolver()._GetBuilderAndSolver()
            # scheme    = KratosMultiphysics.ResidualBasedIncrementalUpdateStaticScheme()

            # A = strategy.GetSystemMatrix()
            # b = strategy.GetSystemVector()

            # space = KratosMultiphysics.UblasSparseSpace()

            # space.SetToZeroMatrix(A)
            # space.SetToZeroVector(b)

            # buildsol.Build(scheme, self.main_model_part, A, b)

            # self.residuals_matrix.append([x for x in b])

        def Finalize(self):
            super().Finalize()
            # np.save("FOM.npy",           self.snapshots_matrix)
            # np.save("FOM_INPUTSFOM.npy", self.inapshots_matrix)
            # np.save("FOM_RESIDUALS.npy", self.residuals_matrix)
            # np.save("FOM_POINLOADS.npy", self.pointload_matrix)

            # self.testerro = np.load("FOM.npy")

            # print(self.snapshots_matrix - self.testerro)

    return StructuralMechanicsAnalysisFOM(global_model, parameters)
