import numpy as np
import scipy

import KratosMultiphysics as KMP
import KratosMultiphysics.StructuralMechanicsApplication as SMA

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from fom_analysis import CreateRomAnalysisInstance

class KratosSimulator():

    def __init__(self, ae_config, needs_cropping):
        with open("ProjectParameters_fom.json", 'r') as parameter_file:
            parameters = KMP.Parameters(parameter_file.read())

        analysis_stage_class = StructuralMechanicsAnalysis

        global_model = KMP.Model()
        self.fake_simulation = CreateRomAnalysisInstance(analysis_stage_class, global_model, parameters)
        self.fake_simulation.Initialize()
        self.fake_simulation.InitializeSolutionStep()

        self.space = KMP.UblasSparseSpace()
        self.strategy = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        self.buildsol = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        self.scheme = KMP.ResidualBasedIncrementalUpdateStaticScheme()
        # self.scheme = self.fake_simulation._GetScheme()
        print(self.scheme)
        self.modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()

        self.is_cropped = needs_cropping

        self.cropped_dof_ids=[]
        if self.is_cropped:
            for node in self.modelpart.Nodes:
                if node.IsFixed(KMP.DISPLACEMENT_X):
                    self.cropped_dof_ids.append((node.Id-1)*2)
                if node.IsFixed(KMP.DISPLACEMENT_Y):
                    self.cropped_dof_ids.append(node.Id*2-1)
        print(self.cropped_dof_ids)

        self.use_force = False
        if "use_force" in ae_config:
            self.use_force=ae_config["use_force"]

    def get_cropped_dof_ids(self):
        return self.cropped_dof_ids

    def project_prediction(self, y_pred, f_true):
        values = y_pred[0]

        itr = 0
        for node in self.modelpart.Nodes:
            if not node.IsFixed(KMP.DISPLACEMENT_X):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_X, values[itr+0])
                node.X = node.X0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_X)

            if not node.IsFixed(KMP.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_Y, values[itr+1])
                node.Y = node.Y0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_Y)

            if self.is_cropped and not node.IsFixed(KMP.DISPLACEMENT_X):
                itr += 2
            
        if self.use_force:
            print('Using force')
            f_value=f_true[0]
            for condition in self.modelpart.Conditions:
                condition.SetValue(SMA.POINT_LOAD, f_value)

    def get_r(self, y_pred, f_true):

        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        self.project_prediction(y_pred, f_true)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)

        # self.buildsol.ApplyDirichletConditions(scheme, modelpart, A, b, b)

        b=np.array(b)

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))

        raw_A = -Ascipy.todense()

        cropped_A = np.delete(raw_A, self.cropped_dof_ids, axis=1)
        
        return cropped_A/1e9, b/1e9
    
    def get_r_array(self, samples, F_true):
        b_list=[]
        for i, sample in enumerate(samples):
            _, b = self.kratos_simulation.get_r(np.expand_dims(sample, axis=0), F_true[i])
            b_list.append(b)
        b_array=np.array(b_list)
        return b_array