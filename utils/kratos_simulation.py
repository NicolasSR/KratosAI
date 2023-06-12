import numpy as np
import scipy

import tensorflow as tf

import KratosMultiphysics as KMP
import KratosMultiphysics.StructuralMechanicsApplication as SMA

from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from fom_analysis import CreateRomAnalysisInstance

import time

class KratosSimulator():

    def __init__(self, working_path, ae_config, residual_scale_factor):
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

        self.use_force = False
        if "use_force" in ae_config:
            self.use_force=ae_config["use_force"]

        self.residual_scale_factor = residual_scale_factor

        # self.Agraph=None
        # self.define_connectivity_and_graph()

    def get_crop_matrix(self):
        indices=[]
        col=0
        for i, node in enumerate(self.modelpart.Nodes):
            if not node.IsFixed(KMP.DISPLACEMENT_X):
                indices.append([2*i,col])
                col+=1
            if not node.IsFixed(KMP.DISPLACEMENT_Y):
                indices.append([2*i+1,col])
                col+=1
        num_cols=col
        num_rows=self.modelpart.NumberOfNodes()*2
        values=np.ones(num_cols)
        crop_mat_tf = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[num_rows,num_cols])
        indices=np.asarray(indices)
        crop_mat_scp = scipy.sparse.coo_array((values, (indices[:,0], indices[:,1])), shape=[num_rows,num_cols]).tocsr()
        
        return crop_mat_tf, crop_mat_scp


    def get_cropped_dof_ids(self):
        indices=[]
        for i, node in enumerate(self.modelpart.Nodes):
            if node.IsFixed(KMP.DISPLACEMENT_X):
                indices.append(2*i)
            if node.IsFixed(KMP.DISPLACEMENT_Y):
                indices.append(2*i+1)
        return indices
    
    def project_prediction_vectorial(self, y_pred, f_true):
        values = y_pred[0]
        values_full=np.zeros(self.modelpart.NumberOfNodes()*2)
        values_full[self.non_cropped_dof_ids]+=values
        values_x=values_full[0::2].copy()
        values_y=values_full[1::2].copy()

        dim = 2
        nodes_array=self.modelpart.Nodes
        x0_vec = self.var_utils.GetInitialPositionsVector(nodes_array,dim)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT_X, values_x, 0)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT_Y, values_y, 0)
        x_vec=x0_vec+values_full
        self.var_utils.SetCurrentPositionsVector(nodes_array,x_vec)

    def project_prediction_vectorial_optim(self, y_pred):
        values = y_pred[0]
        values_full=np.zeros(self.modelpart.NumberOfNodes()*2)
        values_full+=values

        dim = 2
        nodes_array=self.modelpart.Nodes
        x0_vec = self.var_utils.GetInitialPositionsVector(nodes_array,dim)
        self.var_utils.SetSolutionStepValuesVector(nodes_array, KMP.DISPLACEMENT, values_full, 0)
        x_vec=x0_vec+values_full
        self.var_utils.SetCurrentPositionsVector(nodes_array,x_vec)

    def get_v_loss_r_(self, y_pred, b_true):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()
        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        self.project_prediction_vectorial_optim(y_pred)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)
        b=b/self.residual_scale_factor

        err_r=KMP.Vector(b_true[0]-b)

        v_loss_r = self.space.CreateEmptyVectorPointer()
        self.space.ResizeVector(v_loss_r, self.space.Size(b))
        self.space.SetToZeroVector(v_loss_r)

        self.space.TransposeMult(A,err_r,v_loss_r)
        
        err_r=np.expand_dims(np.array(err_r, copy=False),axis=0)
        v_loss_r=np.expand_dims(np.array(v_loss_r, copy=False),axis=0)

        v_loss_r=-v_loss_r/self.residual_scale_factor  # This negation and scaling would be better applied on A, instead on here. But it will work anyways
        
        return err_r, v_loss_r
    
    def get_r_(self, y_pred):
        
        A = self.strategy.GetSystemMatrix()
        b = self.strategy.GetSystemVector()

        self.space.SetToZeroMatrix(A)
        self.space.SetToZeroVector(b)

        self.project_prediction_vectorial_optim(y_pred)

        self.buildsol.Build(self.scheme, self.modelpart, A, b)
        
        b=np.expand_dims(np.array(b, copy=False),axis=0)
        b=b/self.residual_scale_factor
        
        return b

    @tf.function(input_signature=(tf.TensorSpec(None, tf.float64), tf.TensorSpec(None, tf.float64)))
    def get_v_loss_r(self, y_pred, b_true):
        y,w = tf.numpy_function(self.get_v_loss_r_, [y_pred, b_true], (tf.float64, tf.float64))
        return y,w

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
    def get_r(self, y_pred):
        y = tf.numpy_function(self.get_r_, [y_pred], (tf.float64))
        return y
    
    def get_r_array(self, samples, F_true):
        b_list=[]
        for i, sample in enumerate(samples):
            b = self.get_r_(np.expand_dims(sample, axis=0))
            # We may need to remove the outer dimension
            b_list.append(b)
        b_array=np.array(b_list)
        return b_array