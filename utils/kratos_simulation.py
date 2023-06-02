import numpy as np
import scipy

import tensorflow as tf

import KratosMultiphysics as KMP
import KratosMultiphysics.StructuralMechanicsApplication as SMA

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from fom_analysis import CreateRomAnalysisInstance

import time

class KratosSimulator():

    def __init__(self, working_path, ae_config, needs_cropping, residual_scale_factor):
        if "project_parameters_file" in ae_config:
            project_parameters_path=ae_config["dataset_path"]+ae_config["project_parameters_file"]
        else: 
            project_parameters_path='ProjectParameters_fom.json'
            print('LOADED DEFAULT PROJECT PARAMETERS FILE')
        with open(working_path+project_parameters_path, 'r') as parameter_file:
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
        self.var_utils = KMP.VariableUtils()

        self.is_cropped = needs_cropping

        self.cropped_dof_ids=[]
        self.non_cropped_dof_ids=[]
        if self.is_cropped:
            for node in self.modelpart.Nodes:
                if node.IsFixed(KMP.DISPLACEMENT_X):
                    self.cropped_dof_ids.append((node.Id-1)*2)
                else:
                    self.non_cropped_dof_ids.append((node.Id-1)*2)
                if node.IsFixed(KMP.DISPLACEMENT_Y):
                    self.cropped_dof_ids.append(node.Id*2-1)
                else:
                    self.non_cropped_dof_ids.append(node.Id*2-1)
        else:
            self.non_cropped_dof_ids=np.arange(self.modelpart.NumberOfNodes()*2)
        print(self.cropped_dof_ids)
        print(self.non_cropped_dof_ids)

        self.use_force = False
        if "use_force" in ae_config:
            self.use_force=ae_config["use_force"]

        self.residual_scale_factor = residual_scale_factor

    def get_cropped_dof_ids(self):
        return self.cropped_dof_ids
    
    def project_prediction_vectorial(self, y_pred, f_true):
        values = y_pred[0]
        values_full=np.zeros(self.modelpart.NumberOfNodes()*2)
        values_full[self.non_cropped_dof_ids]+=values
        # print(np.all(values_full==values))
        values_x=values_full[0::2].copy()
        values_y=values_full[1::2].copy()

        # print(np.all(values[0::2]==values_x))
        # print(np.all(values[1::2]==values_y))

        dim = 2
        nodes_array=self.modelpart.Nodes
        x0_vec = self.var_utils.GetInitialPositionsVector(nodes_array,dim)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT_X, values_x, 0)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT_Y, values_y, 0)
        x_vec=x0_vec+values_full
        self.var_utils.SetCurrentPositionsVector(nodes_array,x_vec)

        # compar_1_x=self.var_utils.GetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT_X, 0)
        # compar_1_y=self.var_utils.GetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT_Y, 0)
        # print(np.all(np.abs(compar_1_x-values[0::2])<1e-13))
        # print(np.all(np.abs(compar_1_y-values[1::2])<1e-13))

        # return compar_1_x, compar_1_y, x_vec


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

            if not self.is_cropped:
                itr += 2
            elif self.is_cropped and not node.IsFixed(KMP.DISPLACEMENT_X):
                itr += 2
            
        if self.use_force:
            print('Using force')
            f_value=f_true[0]
            for condition in self.modelpart.Conditions:
                condition.SetValue(SMA.POINT_LOAD, f_value)

        # nodes_array=self.modelpart.Nodes
        # compar_2_x=self.var_utils.GetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT_X, 0)
        # compar_2_y=self.var_utils.GetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT_Y, 0)
        # print(np.all(compar_2_x==values[0::2]))
        # print(np.all(compar_2_y==values[1::2]))

        # x_vec=self.var_utils.GetCurrentPositionsVector(nodes_array,2)

        # return compar_2_x, compar_2_y, x_vec

    def get_r(self, y_pred, f_true):

        # if self.is_cropped:
        #     print('NO IMPLEMENTATION FOR CROPPED SNAPSHODS YET')
        #     exit()

        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        self.project_prediction_vectorial(y_pred, f_true)
        # self.project_prediction(y_pred, f_true)

        # print('')
        # print(np.all(np.abs(comp1_x-comp2_x)<1e-9))
        # print(np.all(np.abs(comp1_y-comp2_y)<1e-9))
        # print(np.all(np.abs(comp1_all-comp2_all)<1e-13))

        self.buildsol.Build(self.scheme, self.modelpart, A, b)
        # self.buildsol.ApplyDirichletConditions(scheme, modelpart, A, b, b)
        
        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))
        Ascipy=-Ascipy[:,self.non_cropped_dof_ids]/self.residual_scale_factor
        
        # cropped_A=Ascipy.todense()

        b=np.array(b)
        b=b/self.residual_scale_factor

        # return cropped_A, b
        return Ascipy, b
    
    def get_r_array(self, samples, F_true):
        b_list=[]
        for i, sample in enumerate(samples):
            _, b = self.get_r(np.expand_dims(sample, axis=0), F_true[i])
            b_list.append(b)
        b_array=np.array(b_list)
        return b_array