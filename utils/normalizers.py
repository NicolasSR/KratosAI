import numpy as np
import pandas as pd
import tensorflow as tf
import abc

from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
import matplotlib.pyplot as plt


class Decoder_Normalizer_Base(abc.ABC):

    def __init__(self):
        super().__init__()

    def process_raw_to_input_format(self, data):
        data_norm = self.normalize_data(data)
        return data_norm
    
    def process_input_to_raw_format(self, data, q):
        data_denorm = self.denormalize_data(data, q)
        return data_denorm
    
    @tf.function
    def process_raw_to_input_format_tf(self, tensor):
        tensor_norm = self.normalize_data_tf(tensor)
        return tensor_norm
    
    @tf.function
    def process_input_to_raw_format_tf(self, tensor, tensor_q):
        tensor_denorm = self.denormalize_data_tf(tensor, tensor_q)
        return tensor_denorm
    
    @abc.abstractmethod
    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def normalize_data(self, data):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def denormalize_data(self, data, q):
        """ Define in subclass"""

    @abc.abstractmethod
    def normalize_data_tf(self, tensor):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def denormalize_data_tf(self, tensor, tensor_q):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def get_q(self, data):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def get_q_tf(self, tensor):
        """ Define in subclass"""

class Decoder_Normalizer_SVD_Whitening_NoStand(Decoder_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi = None
        self.phi_tf = None
        self.phi_inf = None
        self.phi_inf_tf = None
        self.phi_sgs = None
        self.phi_sgs_tf = None
        self.phi = None
        self.phi_tf = None
        self.sigma = None
        self.sigma_tf = None
        self.sigma_inf = None
        self.sigma_inf_tf = None
        self.sigma_sgs = None
        self.sigma_sgs_tf = None
        self.dataset_path=working_path+dataset_path
        self.crop_mat_tf=None
        self.crop_mat_scp=None

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        print('Applying SVD-whitening without prior standartization')

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp

        S_crop=np.transpose(self.crop_mat_scp.transpose().dot(S.T))


        try:
            self.phi=np.load(self.dataset_path+'svd_phi_raw_white_nostand.npy')
            self.sigma=np.load(self.dataset_path+'svd_sigma_raw_white_nostand.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            Corr=S/np.sqrt(S.shape[0])
            self.phi,self.sigma, _ = np.linalg.svd(Corr.T)
            np.save(self.dataset_path+'svd_phi_raw_white_nostand.npy', self.phi)
            np.save(self.dataset_path+'svd_sigma_raw_white_nostand.npy', self.sigma)

        # try:
        #     self.phi=np.load(self.dataset_path+'svd_phi_raw.npy')
        # except IOError:
        #     print("No precomputed phi matrix found. Computing a new one")
        #     self.phi, self.sigma, _ = np.linalg.svd(S.T)
        #     np.save(self.dataset_path+'svd_phi_raw.npy', self.phi)
        # self.sigma=np.ones(20)

        self.phi=self.phi[:,:20]
        self.sigma=self.sigma[:20]
        self.phi_inf=self.phi[:,:2].copy()
        self.sigma_inf=self.sigma[:2].copy()
        self.phi_sgs=self.phi[:,2:20].copy()
        self.sigma_sgs=self.sigma[2:20].copy()
        print('Phi matrix shape: ', self.phi.shape)
        print('Sigma array shape: ', self.sigma.shape)
        print('Phi_inf matrix shape: ', self.phi_inf.shape)
        print('Sigma_inf array shape: ', self.sigma_inf.shape)
        print('Phi_sgs matrix shape: ', self.phi_sgs.shape)
        print('Sigma_sgs array shape: ', self.sigma_sgs.shape)
        self.phi_tf=tf.constant(self.phi)
        self.sigma_tf=tf.constant(self.sigma)
        self.phi_inf_tf=tf.constant(self.phi_inf)
        self.sigma_inf_tf=tf.constant(self.sigma_inf)
        self.phi_sgs_tf=tf.constant(self.phi_sgs)
        self.sigma_sgs_tf=tf.constant(self.sigma_sgs)

        # S_norm=np.divide(np.matmul(self.phi.T,S_crop.T).T,self.sigma)
        # plt.boxplot(S_norm)
        # plt.show()
        # exit()

        S_recons = np.matmul(self.phi,np.multiply(np.divide(np.matmul(self.phi.T,S.T).T,self.sigma),self.sigma).T).T
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons-S)/np.linalg.norm(S))
        err_aux=np.linalg.norm(S-S_recons, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])

    def normalize_data(self, data):
        output_data=data.copy()
        output_data=np.divide(np.matmul(self.phi_sgs.T,output_data.T).T,self.sigma_sgs)
        plt.boxplot(output_data)
        plt.show()
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_sgs_tf,tensor,transpose_a=True,transpose_b=True))/self.sigma_sgs_tf
        return output_tensor

    def denormalize_data(self, data, q):
        output_data=np.matmul(self.phi,np.multiply(data, self.sigma).T).T
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        print('NOT IMPLEMENTED YET')
        exit()
        return output_data
    
    def denormalize_data_tf(self,tensor, q_tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,tensor*self.sigma_tf,transpose_b=True))
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=False, adjoint_b=True))
        print('NOT IMPLEMENTED YET')
        exit()
        return output_tensor
    
    def get_q(self, data):
        output_data=data.copy()
        output_data=np.divide(np.matmul(self.phi_inf.T,output_data.T).T,self.sigma_inf)
        plt.boxplot(output_data)
        plt.show()
        return output_data
    
    def get_q_tf(self, tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,tensor,transpose_a=True,transpose_b=True))/self.sigma_inf_tf
        return output_tensor
    
class Decoder_Normalizer_SVD_Range(Decoder_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi = None
        self.phi_tf = None
        self.phi_inf = None
        self.phi_inf_tf = None
        self.phi_sgs = None
        self.phi_sgs_tf = None
        self.phi = None
        self.phi_tf = None
        self.max_q_inf=None
        self.min_q_inf=None
        self.max_q_sgs=None
        self.min_q_sgs=None
        self.max_q_inf_tf=None
        self.min_q_inf_tf=None
        self.max_q_sgs_tf=None
        self.min_q_sgs_tf=None
        self.dataset_path=working_path+dataset_path
        self.crop_mat_tf=None
        self.crop_mat_scp=None

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        print('Applying SVD-whitening without prior standartization')

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp

        S_crop=np.transpose(self.crop_mat_scp.transpose().dot(S.T))


        # try:
        #     self.phi=np.load(self.dataset_path+'svd_phi_raw_white_nostand.npy')
        #     self.sigma=np.load(self.dataset_path+'svd_sigma_raw_white_nostand.npy')
        # except IOError:
        #     print("No precomputed phi matrix found. Computing a new one")
        #     Corr=S/np.sqrt(S.shape[0])
        #     self.phi,self.sigma, _ = np.linalg.svd(Corr.T)
        #     np.save(self.dataset_path+'svd_phi_raw_white_nostand.npy', self.phi)
        #     np.save(self.dataset_path+'svd_sigma_raw_white_nostand.npy', self.sigma)

        try:
            self.phi=np.load(self.dataset_path+'svd_phi_raw.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            self.phi, self.sigma, _ = np.linalg.svd(S.T)
            np.save(self.dataset_path+'svd_phi_raw.npy', self.phi)

        self.phi=self.phi[:,:20]
        self.phi_inf=self.phi[:,:2].copy()
        self.phi_sgs=self.phi[:,2:20].copy()
        print('Phi matrix shape: ', self.phi.shape)
        print('Phi_inf matrix shape: ', self.phi_inf.shape)
        print('Phi_sgs matrix shape: ', self.phi_sgs.shape)
        self.phi_tf=tf.constant(self.phi)
        self.phi_inf_tf=tf.constant(self.phi_inf)
        self.phi_sgs_tf=tf.constant(self.phi_sgs)

        q_inf=np.matmul(self.phi_inf.T,S.T).T
        q_sgs=np.matmul(self.phi_sgs.T,S.T).T

        self.max_q_inf=np.expand_dims(np.max(q_inf, axis=0), axis=0)
        self.min_q_inf=np.expand_dims(np.min(q_inf, axis=0), axis=0)
        self.max_q_sgs=np.expand_dims(np.max(q_sgs, axis=0), axis=0)
        self.min_q_sgs=np.expand_dims(np.min(q_sgs, axis=0), axis=0)
        
        self.max_q_inf_tf=tf.constant(self.max_q_inf)
        self.min_q_inf_tf=tf.constant(self.min_q_inf)
        self.max_q_sgs_tf=tf.constant(self.max_q_sgs)
        self.min_q_sgs_tf=tf.constant(self.min_q_sgs)

        # S_norm=np.divide(np.matmul(self.phi.T,S_crop.T).T,self.sigma)
        # plt.boxplot(S_norm)
        # plt.show()
        # exit()

        # S_recons = np.matmul(self.phi,np.multiply(np.divide(np.matmul(self.phi.T,S.T).T,self.sigma),self.sigma).T).T
        # print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_recons-S)/np.linalg.norm(S))
        # err_aux=np.linalg.norm(S-S_recons, ord=2, axis=1)/np.linalg.norm(S, ord=2, axis=1)
        # print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S.shape[0])

    def normalize_data(self, data):
        output_data=data.copy()
        output_data=np.matmul(self.phi_sgs.T,output_data.T).T
        output_data=(output_data-self.min_q_sgs)/(self.max_q_sgs-self.min_q_sgs)
        plt.boxplot(output_data)
        plt.show()
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_sgs_tf,tensor,transpose_a=True,transpose_b=True))
        output_data=(output_tensor-self.min_q_sgs_tf)/(self.max_q_sgs_tf-self.min_q_sgs_tf)
        return output_tensor

    def denormalize_data(self, data, q):
        output_data=np.matmul(self.phi,np.multiply(data, self.sigma).T).T
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        print('NOT IMPLEMENTED YET')
        exit()
        return output_data
    
    def denormalize_data_tf(self,tensor, q_tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,tensor*self.sigma_tf,transpose_b=True))
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=False, adjoint_b=True))
        print('NOT IMPLEMENTED YET')
        exit()
        return output_tensor
    
    def get_q(self, data):
        output_data=data.copy()
        output_data=np.matmul(self.phi_inf.T,output_data.T).T
        output_data=(output_data-self.min_q_inf)/(self.max_q_inf-self.min_q_inf)
        plt.boxplot(output_data)
        plt.show()
        return output_data
    
    def get_q_tf(self, tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_inf_tf,tensor,transpose_a=True,transpose_b=True))
        output_tensor=(output_tensor-self.min_q_inf_tf)/(self.max_q_inf_tf-self.min_q_inf_tf)
        return output_tensor

class DecCorr_Normalizer_FeatureScale(Decoder_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.scale_factor=None
        self.scale_factor_tf=None
        self.phi = None
        self.phi_tf = None
        self.crop_mat_tf=None
        self.crop_mat_scp=None
        self.dataset_path=working_path+dataset_path

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        try:
            self.phi=np.load(self.dataset_path+'svd_phi_raw.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            self.phi, _, _ = np.linalg.svd(S.T)
            np.save(self.dataset_path+'svd_phi_raw.npy', self.phi)

        self.phi=self.phi[:,:2]

        S_err=S-np.matmul(self.phi,np.matmul(self.phi.T,S.T)).T
        max_arg_vec=np.argmax(np.abs(S_err), axis=0)
        scale_factor=[]
        for i, id in enumerate(max_arg_vec):
            scale_factor.append(S_err[id,i])
        scale_factor=np.expand_dims(np.array(scale_factor, copy=False), axis=0)
        self.scale_factor=scale_factor
        self.scale_factor_tf=tf.constant(scale_factor)
        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp
    
    def normalize_data(self, data):
        output_data=data.copy()
        output_data=output_data-np.matmul(self.phi,np.matmul(self.phi.T,output_data.T)).T
        output_data=output_data/self.scale_factor
        output_data=np.transpose(self.crop_mat_scp.transpose().dot(output_data.T))
        return output_data
    
    def normalize_data_tf(self,tensor):
        aux=tf.linalg.matmul(self.phi_tf,tensor,transpose_a=True,transpose_b=True)
        output_tensor=tensor-tf.transpose(tf.linalg.matmul(self.phi_tf,aux,transpose_a=False,transpose_b=False))
        output_tensor=output_tensor/self.scale_factor_tf
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=True, adjoint_b=True))
        return output_tensor

    def denormalize_data(self, data, q):
        output_data=data.copy()
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        output_data=output_data*self.scale_factor
        output_data=output_data+np.matmul(self.phi,q.T).T
        return output_data
    
    def denormalize_data_tf(self, tensor, tensor_q):
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, tensor, adjoint_a=False, adjoint_b=True))
        output_tensor=output_tensor*self.scale_factor_tf
        output_tensor=output_tensor+tf.transpose(tf.linalg.matmul(self.phi_tf,tensor_q,transpose_b=True))
        return output_tensor
    
    def get_q(self, data):
        output_data=data.copy()
        output_data=np.matmul(self.phi.T,output_data.T).T
        return output_data
    
    def get_q_tf(self, tensor):
        output_tensor=tf.linalg.matmul(self.phi_tf,tensor,transpose_a=True,transpose_b=True)
        return output_tensor
    

class DecCorr_Normalizer_ChannelScale(Decoder_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.scale_factor=None
        self.scale_factor_tf=None
        self.phi = None
        self.phi_tf = None
        self.crop_mat_tf=None
        self.crop_mat_scp=None
        self.dataset_path=working_path+dataset_path

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        try:
            self.phi=np.load(self.dataset_path+'svd_phi_raw.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            self.phi, _, _ = np.linalg.svd(S.T)
            np.save(self.dataset_path+'svd_phi_raw.npy', self.phi)

        self.phi=self.phi[:,:2]

        S_err=S-np.matmul(self.phi,np.matmul(self.phi.T,S.T)).T
        S_err_x=S_err[:,0::2]
        S_err_y=S_err[:,1::2]
        max_arg_vec_x=np.unravel_index(np.argmax(np.abs(S_err_x)), S_err_x.shape)
        max_arg_vec_y=np.unravel_index(np.argmax(np.abs(S_err_y)), S_err_y.shape)
        scale_factor=[]
        for i in range(S_err.shape[1]//2):
            scale_factor.append(S_err_x[max_arg_vec_x])
            scale_factor.append(S_err_y[max_arg_vec_y])
        scale_factor=np.expand_dims(np.array(scale_factor, copy=False), axis=0)
        self.scale_factor=scale_factor
        self.scale_factor_tf=tf.constant(scale_factor)
        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp
    
    def normalize_data(self, data):
        output_data=data.copy()
        output_data=output_data-np.matmul(self.phi,np.matmul(self.phi.T,output_data.T)).T
        output_data=output_data/self.scale_factor
        output_data=np.transpose(self.crop_mat_scp.transpose().dot(output_data.T))
        return output_data
    
    def normalize_data_tf(self,tensor):
        aux=tf.linalg.matmul(self.phi_tf,tensor,transpose_a=True,transpose_b=True)
        output_tensor=tensor-tf.transpose(tf.linalg.matmul(self.phi_tf,aux,transpose_a=False,transpose_b=False))
        output_tensor=output_tensor/self.scale_factor_tf
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=True, adjoint_b=True))
        return output_tensor

    def denormalize_data(self, data, q):
        output_data=data.copy()
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        output_data=output_data*self.scale_factor
        output_data=output_data+np.matmul(self.phi,q.T).T
        return output_data
    
    def denormalize_data_tf(self, tensor, tensor_q):
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, tensor, adjoint_a=False, adjoint_b=True))
        output_tensor=output_tensor*self.scale_factor_tf
        output_tensor=output_tensor+tf.transpose(tf.linalg.matmul(self.phi_tf,tensor_q,transpose_b=True))
        return output_tensor
    
    def get_q(self, data):
        output_data=data.copy()
        output_data=np.matmul(self.phi.T,output_data.T).T
        return output_data
    
    def get_q_tf(self, tensor):
        output_tensor=tf.linalg.matmul(self.phi_tf,tensor,transpose_a=True,transpose_b=True)
        return output_tensor
        


class AE_Normalizer_Base(abc.ABC):

    def __init__(self):
        super().__init__()

    def process_raw_to_input_format(self, data):
        data_norm = self.normalize_data(data)
        return data_norm
    
    def process_input_to_raw_format(self, data):
        data_denorm = self.denormalize_data(data)
        return data_denorm
    
    @tf.function
    def process_raw_to_input_format_tf(self, tensor):
        tensor_norm = self.normalize_data_tf(tensor)
        return tensor_norm
    
    @tf.function
    def process_input_to_raw_format_tf(self, tensor):
        tensor_denorm = self.denormalize_data_tf(tensor)
        return tensor_denorm
    
    @abc.abstractmethod
    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def normalize_data(self, data):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def denormalize_data(self, data):
        """ Define in subclass"""

    @abc.abstractmethod
    def normalize_data_tf(self, tensor):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def denormalize_data_tf(self, tensor):
        """ Define in subclass"""

""" class AE_Normalizer_SVD_Whitening_Test(AE_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi = None
        self.phi_tf = None
        self.means = None
        self.means_tf = None
        self.std = None
        self.std_tf = None
        self.diag_vals = None
        self.diag_vals_tf = None
        self.dataset_path=working_path+dataset_path
        self.crop_mat_tf=None
        self.crop_mat_scp=None

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        print('Scaling each feature in S, then applying SVD')
        self.feat_factors = []
        abs_S_df = pd.DataFrame(np.abs(S))
        for i in range(len(abs_S_df.columns)):
            self.feat_factors.append(abs_S_df[i].max())
        self.feat_factors_tf = tf.constant(self.feat_factors)

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp

        S_crop=np.transpose(self.crop_mat_scp.transpose().dot(S.T))
        # print('Shape S_crop: ', S_crop.shape)
        # self.means=np.mean(S_crop, axis=0)
        # print('Shape means: ', self.means.shape)
        # self.means_tf=tf.constant(self.means)
        # self.std=np.std(S_crop, axis=0)
        # print('Shape std: ', self.std.shape)
        # self.std_tf=tf.constant(self.std)
        # S_stand=np.divide(np.subtract(S_crop,self.means),self.std)
        # S_stand=np.subtract(S_crop,self.means)
        S_stand=S_crop
        print('Shape S_stand: ', S_stand.shape)
        print('Means: ', np.all(np.abs(np.mean(S_stand, axis=0))<1e-15))

        # try:
        #     self.phi=np.load(self.dataset_path+'svd_phi_prenorm.npy')
        # except IOError:
        print("No precomputed phi matrix found. Computing a new one")
        # Corr=np.matmul(S_stand.T,S_stand)/S.shape[0]
        Corr=S_stand/np.sqrt(S.shape[0])
        # self.phi,sigma,_ = np.linalg.svd(Corr)
        self.phi,sigma,_,error = RandomizedSingularValueDecomposition().Calculate(Corr.T,1e-16)
        # np.save(self.dataset_path+'svd_phi_prenorm.npy', self.phi)
        # self.phi=self.phi[:,:50]
        print('Phi matrix shape: ', self.phi.shape)
        self.phi_tf=tf.constant(self.phi)
        # self.diag_vals=np.sqrt(sigma[:50])
        # self.diag_vals=np.sqrt(sigma)
        self.diag_vals=sigma
        print('Sigma array shape: ', self.diag_vals.shape)
        self.diag_vals=tf.constant(self.diag_vals)

        S_white = np.divide(np.matmul(self.phi.T,S_stand.T).T,self.diag_vals)
        print('Shape S_white: ', S_white.shape)

        plt.boxplot(S_white)
        plt.show()

        print(np.matmul(S_white.T,S_white)/S.shape[0])

        S_stand_recons = np.matmul(self.phi,np.multiply(np.divide(np.matmul(self.phi.T,S_stand.T).T,self.diag_vals),self.diag_vals).T).T
        print('Shape S_stand_recons: ', S_stand_recons.shape)
        print('Reconstruction error SVD: ', np.linalg.norm(S_stand_recons-S_stand)/np.linalg.norm(S_stand))

        exit()


        
        S_recons=np.matmul(self.phi,np.matmul(self.phi.T,S_norm.T)).T
        print('Reconstruction error SVD: ', np.linalg.norm(S_recons-S_norm)/np.linalg.norm(S_norm))
        
        

    def normalize_data(self, data):
        output_data=np.divide(data, self.feat_factors)
        output_data=np.transpose(self.crop_mat_scp.transpose().dot(output_data.T))
        output_data=np.matmul(self.phi.T,output_data.T).T

        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor/self.feat_factors_tf
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=True, adjoint_b=True))
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,output_tensor,transpose_a=True,transpose_b=True))
        return output_tensor

    def denormalize_data(self, data):
        output_data=np.matmul(self.phi,data.T).T
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        output_data=np.multiply(output_data, self.feat_factors)
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,tensor,transpose_b=True))
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=False, adjoint_b=True))
        output_tensor=output_tensor*self.feat_factors_tf
        return output_tensor """

class AE_Normalizer_SVD_Whitening(AE_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi = None
        self.phi_tf = None
        self.means = None
        self.means_tf = None
        self.std = None
        self.std_tf = None
        self.sigma = None
        self.sigma_tf = None
        self.dataset_path=working_path+dataset_path
        self.crop_mat_tf=None
        self.crop_mat_scp=None

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        print('Standardizing each feature in S_crop, then applying SVD-whitening')

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp

        S_crop=np.transpose(self.crop_mat_scp.transpose().dot(S.T))

        print('Shape S_crop: ', S_crop.shape)
        self.means=np.mean(S_crop, axis=0)
        print('Shape means: ', self.means.shape)
        self.means_tf=tf.constant(self.means)
        self.std=np.std(S_crop, axis=0)
        print('Shape std: ', self.std.shape)
        self.std_tf=tf.constant(self.std)
        S_stand=np.divide(np.subtract(S_crop,self.means),self.std)
        print('Shape S_stand: ', S_stand.shape)
        print('Means: ', np.all(np.abs(np.mean(S_stand, axis=0))<1e-15))

        try:
            self.phi=np.load(self.dataset_path+'svd_phi_white.npy')
            self.sigma=np.load(self.dataset_path+'svd_sigma_white.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            Corr=S_stand/np.sqrt(S.shape[0])
            self.phi,self.sigma, _ = np.linalg.svd(Corr.T)
            # self.phi,self.sigma,_,error = RandomizedSingularValueDecomposition().Calculate(Corr.T,1e-16)
            np.save(self.dataset_path+'svd_phi_white.npy', self.phi)
            np.save(self.dataset_path+'svd_sigma_white.npy', self.sigma)

        self.phi=self.phi[:,:20]
        self.sigma=self.sigma[:20]
        print('Phi matrix shape: ', self.phi.shape)
        print('Sigma array shape: ', self.sigma.shape)
        self.phi_tf=tf.constant(self.phi)
        self.sigma_tf=tf.constant(self.sigma)

        # S_norm=np.divide(np.matmul(self.phi.T,S_stand.T).T,self.sigma)
        # plt.boxplot(S_norm)
        # plt.show()
        # exit()

        S_stand_recons = np.matmul(self.phi,np.multiply(np.divide(np.matmul(self.phi.T,S_stand.T).T,self.sigma),self.sigma).T).T
        print('Reconstruction error SVD: ', np.linalg.norm(S_stand_recons-S_stand)/np.linalg.norm(S_stand))

    def normalize_data(self, data):
        output_data=np.transpose(self.crop_mat_scp.transpose().dot(data.T))
        output_data=np.divide(np.subtract(output_data,self.means),self.std)
        output_data=np.divide(np.matmul(self.phi.T,output_data.T).T,self.sigma)
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, tensor, adjoint_a=True, adjoint_b=True))
        output_tensor=(output_tensor-self.means_tf)/self.std_tf
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,output_tensor,transpose_a=True,transpose_b=True))/self.sigma_tf
        return output_tensor

    def denormalize_data(self, data):
        output_data=np.matmul(self.phi,np.multiply(data, self.sigma).T).T
        output_data=np.add(np.multiply(output_data,self.std),self.means)
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,tensor*self.sigma_tf,transpose_b=True))
        output_tensor=(output_tensor*self.std_tf)+self.means_tf
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=False, adjoint_b=True))
        return output_tensor
    

class AE_Normalizer_SVD_Whitening_NoStand(AE_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi = None
        self.phi_tf = None
        self.sigma = None
        self.sigma_tf = None
        self.dataset_path=working_path+dataset_path
        self.crop_mat_tf=None
        self.crop_mat_scp=None

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        print('Applying SVD-whitening without prior standartization')

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp

        S_crop=np.transpose(self.crop_mat_scp.transpose().dot(S.T))

        try:
            self.phi=np.load(self.dataset_path+'svd_phi_white_nostand.npy')
            self.sigma=np.load(self.dataset_path+'svd_sigma_white_nostand.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            Corr=S_crop/np.sqrt(S.shape[0])
            self.phi,self.sigma, _ = np.linalg.svd(Corr.T)
            np.save(self.dataset_path+'svd_phi_white_nostand.npy', self.phi)
            np.save(self.dataset_path+'svd_sigma_white_nostand.npy', self.sigma)

        self.phi=self.phi[:,:20]
        self.sigma=self.sigma[:20]
        print('Phi matrix shape: ', self.phi.shape)
        print('Sigma array shape: ', self.sigma.shape)
        self.phi_tf=tf.constant(self.phi)
        self.sigma_tf=tf.constant(self.sigma)

        # S_norm=np.divide(np.matmul(self.phi.T,S_crop.T).T,self.sigma)
        # plt.boxplot(S_norm)
        # plt.show()
        # exit()

        S_crop_recons = np.matmul(self.phi,np.multiply(np.divide(np.matmul(self.phi.T,S_crop.T).T,self.sigma),self.sigma).T).T
        print('Reconstruction error SVD (Frob): ', np.linalg.norm(S_crop_recons-S_crop)/np.linalg.norm(S_crop))
        err_aux=np.linalg.norm(S_crop-S_crop_recons, ord=2, axis=1)/np.linalg.norm(S_crop, ord=2, axis=1)
        print('Reconstruction error SVD (Mean L2): ', np.sum(err_aux)/S_crop.shape[0])

    def normalize_data(self, data):
        output_data=np.transpose(self.crop_mat_scp.transpose().dot(data.T))
        output_data=np.divide(np.matmul(self.phi.T,output_data.T).T,self.sigma)
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, tensor, adjoint_a=True, adjoint_b=True))
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,output_tensor,transpose_a=True,transpose_b=True))/self.sigma_tf
        return output_tensor

    def denormalize_data(self, data):
        output_data=np.matmul(self.phi,np.multiply(data, self.sigma).T).T
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,tensor*self.sigma_tf,transpose_b=True))
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=False, adjoint_b=True))
        return output_tensor


class AE_Normalizer_SVD_Prenorm(AE_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi = None
        self.phi_tf = None
        self.feat_factors = None
        self.feat_factors_tf = None
        self.dataset_path=working_path+dataset_path
        self.crop_mat_tf=None
        self.crop_mat_scp=None

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        print('Scaling each feature in S, then applying SVD')
        self.feat_factors = []
        abs_S_df = pd.DataFrame(np.abs(S))
        for i in range(len(abs_S_df.columns)):
            self.feat_factors.append(abs_S_df[i].max())
        self.feat_factors_tf = tf.constant(self.feat_factors)

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp

        S_norm=S/np.array(self.feat_factors)[None,:]
        S_norm=np.transpose(self.crop_mat_scp.transpose().dot(S_norm.T))

        try:
            self.phi=np.load(self.dataset_path+'svd_phi_prenorm.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            self.phi,sigma,_,error = RandomizedSingularValueDecomposition().Calculate(S_norm.T,1e-16)
            print(self.phi)
            np.save(self.dataset_path+'svd_phi_prenorm.npy', self.phi)
        
        S_recons=np.matmul(self.phi,np.matmul(self.phi.T,S_norm.T)).T
        print('Reconstruction error SVD: ', np.linalg.norm(S_recons-S_norm)/np.linalg.norm(S_norm))
        
        print('Phi matrix shape: ', self.phi.shape)
        self.phi_tf=tf.constant(self.phi)

    def normalize_data(self, data):
        output_data=np.divide(data, self.feat_factors)
        output_data=np.transpose(self.crop_mat_scp.transpose().dot(output_data.T))
        output_data=np.matmul(self.phi.T,output_data.T).T

        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor/self.feat_factors_tf
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=True, adjoint_b=True))
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,output_tensor,transpose_a=True,transpose_b=True))
        return output_tensor

    def denormalize_data(self, data):
        output_data=np.matmul(self.phi,data.T).T
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        output_data=np.multiply(output_data, self.feat_factors)
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,tensor,transpose_b=True))
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=False, adjoint_b=True))
        output_tensor=output_tensor*self.feat_factors_tf
        return output_tensor
    

class AE_Normalizer_SVD_PrenormChan(AE_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi = None
        self.phi_tf = None
        self.feat_factors = None
        self.feat_factors_tf = None
        self.dataset_path=working_path+dataset_path
        self.crop_mat_tf=None
        self.crop_mat_scp=None

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        print('Scaling each channel in S, then applying SVD')

        ch1_factor=np.max(abs(S[:,0::2]))
        ch2_factor=np.max(abs(S[:,1::2]))
        print(ch1_factor)
        print(ch2_factor)

        aux=np.arange(S.shape[1])
        self.feat_factors=np.where(aux%2==0, ch1_factor, ch2_factor)
        self.feat_factors_tf=tf.constant(np.expand_dims(np.where(aux%2==0, ch1_factor, ch2_factor),axis=0))

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp

        S_norm=S/self.feat_factors[None,:]
        S_norm=np.transpose(self.crop_mat_scp.transpose().dot(S_norm.T))

        try:
            self.phi=np.load(self.dataset_path+'svd_phi_prenorm_chan.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            self.phi,sigma,_,error = RandomizedSingularValueDecomposition().Calculate(S_norm.T,1e-16)
            np.save(self.dataset_path+'svd_phi_prenorm_chan.npy', self.phi)
        
        S_recons=np.matmul(self.phi,np.matmul(self.phi.T,S_norm.T)).T
        print('Reconstruction error SVD: ', np.linalg.norm(S_recons-S_norm)/np.linalg.norm(S_norm))
        
        print('Phi matrix shape: ', self.phi.shape)
        self.phi_tf=tf.constant(self.phi)

    def normalize_data(self, data):
        output_data=np.divide(data, self.feat_factors)
        output_data=np.transpose(self.crop_mat_scp.transpose().dot(output_data.T))
        output_data=np.matmul(self.phi.T,output_data.T).T

        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor/self.feat_factors_tf
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=True, adjoint_b=True))
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,output_tensor,transpose_a=True,transpose_b=True))
        return output_tensor

    def denormalize_data(self, data):
        output_data=np.matmul(self.phi,data.T).T
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        output_data=np.multiply(output_data, self.feat_factors)
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,tensor,transpose_b=True))
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=False, adjoint_b=True))
        output_tensor=output_tensor*self.feat_factors_tf
        return output_tensor
    

class AE_Normalizer_SVD(AE_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi=None
        self.phi_tf=None
        self.feat_factors = None
        self.feat_factors_tf = None
        self.dataset_path=working_path+dataset_path

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        try:
            self.phi=np.load(self.dataset_path+'svd_phi.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            self.phi,sigma,_,error = RandomizedSingularValueDecomposition().Calculate(S.T,1e-16)
            np.save(self.dataset_path+'svd_phi.npy', self.phi)

        S_recons=np.matmul(self.phi,np.matmul(self.phi.T,S.T)).T
        print('Reconstruction error SVD: ', np.linalg.norm(S_recons-S)/np.linalg.norm(S))
        
        print('Phi matrix shape: ', self.phi.shape)
        self.phi_tf=tf.constant(self.phi)
        
        S_svd=np.matmul(self.phi.T,S.T).T
        plt.boxplot(S_svd)
        plt.show()

        self.feat_factors = []
        print('Scaling each feature in S_svd')
        max_ids=np.argmax(np.abs(S_svd), axis=0)
        for i in range(max_ids.shape[0]):
            self.feat_factors.append(S_svd[max_ids[i],i])

        S_input=np.divide(S_svd, self.feat_factors)
        plt.boxplot(S_input)
        plt.show()
        exit()

    def normalize_data(self, data):
        output_data=np.matmul(self.phi.T,data.T).T
        output_data=np.divide(output_data, self.feat_factors)

        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,tensor,transpose_a=True,transpose_b=True))
        output_tensor=output_tensor/self.feat_factors_tf
        return output_tensor

    def denormalize_data(self, data):
        output_data=np.multiply(data, self.feat_factors)
        output_data=np.matmul(self.phi,output_data.T).T
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*self.feat_factors_tf
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,output_tensor,transpose_b=True))
        return output_tensor
    
    
class AE_Normalizer_SVD_Uniform(AE_Normalizer_Base):
    def __init__(self, working_path, dataset_path):
        super().__init__()
        self.phi=None
        self.phi_tf=None
        self.feat_factor = None
        self.feat_factor_tf = None
        self.dataset_path=working_path+dataset_path

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        try:
            self.phi=np.load(self.dataset_path+'svd_phi.npy')
        except IOError:
            print("No precomputed phi matrix found. Computing a new one")
            self.phi,sigma,_,error = RandomizedSingularValueDecomposition().Calculate(S.T,1e-16)
            np.save(self.dataset_path+'svd_phi.npy', self.phi)

        S_recons=np.matmul(self.phi,np.matmul(self.phi.T,S.T)).T
        print('Reconstruction error SVD: ', np.linalg.norm(S_recons-S)/np.linalg.norm(S))
        
        print('Phi matrix shape: ', self.phi.shape)
        self.phi_tf=tf.constant(self.phi)
        
        S_svd=np.matmul(self.phi.T,S.T).T
        self.feat_factor = np.max(abs(S_svd))
        self.feat_factor_tf = tf.constant(self.feat_factor)

    def normalize_data(self, data):
        output_data=np.matmul(self.phi.T,data.T).T
        output_data=output_data/self.feat_factor

        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,tensor,transpose_a=True,transpose_b=True))
        output_tensor=output_tensor/self.feat_factor_tf
        return output_tensor

    def denormalize_data(self, data):
        output_data=data*self.feat_factor
        output_data=np.matmul(self.phi,output_data.T).T
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*self.feat_factor_tf
        output_tensor=tf.transpose(tf.linalg.matmul(self.phi_tf,output_tensor,transpose_b=True))
        return output_tensor
    
class AE_Normalizer_ChannelScale(AE_Normalizer_Base):
    def __init__(self):
        super().__init__()
        self.ch1_factor=None
        self.ch2_factor=None
        self.div_coeff=None
        self.div_coeff_tf=None
        self.crop_mat_tf=None
        self.crop_mat_scp=None

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        ch1_max=np.max(S[:,0::2])
        ch1_min=np.min(S[:,0::2])
        if abs(ch1_max)>abs(ch1_min):
            self.ch1_factor=ch1_max
        else:
            self.ch1_factor=ch1_min
        ch2_max=np.max(S[:,1::2])
        ch2_min=np.min(S[:,1::2])
        if abs(ch2_max)>abs(ch2_min):
            self.ch2_factor=ch2_max
        else:
            self.ch2_factor=ch2_min
        print(self.ch1_factor)
        print(self.ch2_factor)

        aux=np.arange(S.shape[1])
        self.div_coeff=np.expand_dims(np.where(aux%2==0, self.ch1_factor, self.ch2_factor),axis=0)
        self.div_coeff_tf=tf.constant(np.expand_dims(np.where(aux%2==0, self.ch1_factor, self.ch2_factor),axis=0))

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp
    
    def normalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]/self.ch1_factor
        output_data[:,1::2]=output_data[:,1::2]/self.ch2_factor
        output_data=np.transpose(self.crop_mat_scp.transpose().dot(output_data.T))
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor/self.div_coeff_tf
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=True, adjoint_b=True))
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        output_data[:,0::2]=output_data[:,0::2]*self.ch1_factor
        output_data[:,1::2]=output_data[:,1::2]*self.ch2_factor
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, tensor, adjoint_a=False, adjoint_b=True))
        output_tensor=output_tensor*self.div_coeff_tf
        return output_tensor
    
""" class AE_Normalizer_ChannelRange(AE_Normalizer_Base):
    def __init__(self):
        super().__init__()
        self.ch1_max=None
        self.ch1_min=None
        self.ch2_max=None
        self.ch2_min=None
        self.subt_term=None
        self.div_coeff=None

        print('CROPPING NOT IMPLEMENTED PROPERLY FOR THIS NORMALISATION. ABORING')
        exit()

    def configure_normalization_data(self, S):
        self.ch1_max=np.max(S[:,0::2])
        self.ch1_min=np.min(S[:,0::2])
        self.ch2_max=np.max(S[:,1::2])
        self.ch2_min=np.min(S[:,1::2])
        aux=np.arange(S.shape[1])
        self.subt_term=np.expand_dims(np.where(aux%2==0, self.ch1_min, self.ch2_min),axis=0)
        self.div_coeff=np.expand_dims(np.where(aux%2==0, self.ch1_max-self.ch1_min, self.ch2_max-self.ch2_min),axis=0)

    def normalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=(output_data[:,0::2]-self.ch1_min)/(self.ch1_max-self.ch1_min)
        output_data[:,1::2]=(output_data[:,1::2]-self.ch2_min)/(self.ch2_max-self.ch2_min)
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor-self.subt_term
        output_tensor=output_tensor/self.div_coeff
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]*(self.ch1_max-self.ch1_min)+self.ch1_min
        output_data[:,1::2]=output_data[:,1::2]*(self.ch2_max-self.ch2_min)+self.ch2_min
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*self.div_coeff
        output_tensor=output_tensor+self.subt_term
        return output_tensor """
  
  
""" class AE_Normalizer_FeatureStand(AE_Normalizer_Base):
    def __init__(self):
        super().__init__()
        self.feat_means = None   
        self.feat_stds = None
        print('CROPPING NOT IMPLEMENTED PROPERLY FOR THIS NORMALISATION. ABORING')
        exit()

    def configure_normalization_data(self, S):
        feat_means = []
        feat_stds = []
        print('Normalizing each feature in S')
        S_df = pd.DataFrame(S)
        for i in range(len(S_df.columns)):
            feat_means.append(S_df[i].mean())
            feat_stds.append(S_df[i].std())
        self.feat_means = feat_means
        self.feat_stds = feat_stds
    
    def normalize_data(self, data):
        output_data=data.copy()
        for i in range(output_data.shape[1]):
            output_data[:,i]=(output_data[:,i]-self.feat_means[i])/(self.feat_stds[i]+0.00000001)
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor-np.array([self.feat_means])
        output_tensor=output_tensor/(np.array([self.feat_stds])+0.00000001)
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        for i in range(output_data.shape[1]):
            output_data[:,i]= (output_data[:,i]*self.feat_stds[i])+self.feat_means[i]
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*np.array([self.feat_stds])
        output_tensor=output_tensor+np.array([self.feat_means])
        return output_tensor """
    
    
class AE_Normalizer_None(AE_Normalizer_Base):
    def __init__(self):
        super().__init__()

    def configure_normalization_data(self, S):
        return

    def normalize_data(self, data):
        output_data=data.copy()
        return output_data
    
    def normalize_data_tf(self, tensor):
        output_tensor=tensor
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        return output_data
    
    def denormalize_data_tf(self, tensor):
        output_tensor=tensor
        return output_tensor
    

class Conv2D_AE_Normalizer_Base(abc.ABC):

    def __init__(self):
        super().__init__()

    def process_raw_to_input_format(self, data):
        data_norm = self.normalize_data(data)
        data_norm_2d = self.reorganize_into_channels(data_norm)
        return data_norm_2d
    
    def process_input_to_raw_format(self, data):
        data_flat = self.reorganize_into_original(data)
        data_flat_denorm = self.denormalize_data(data_flat)
        return data_flat_denorm
    
    @tf.function
    def process_raw_to_input_format_tf(self, tensor):
        tensor_norm = self.normalize_data_tf(tensor)
        tensor_norm_2d = self.reorganize_into_channels_tf(tensor_norm)
        return tensor_norm_2d
    
    @tf.function
    def process_input_to_raw_format_tf(self, tensor):
        tensor_flat = self.reorganize_into_original_tf(tensor)
        tensor_flat_denorm = self.denormalize_data_tf(tensor_flat)
        return tensor_flat_denorm
    
    @abc.abstractmethod
    def configure_normalization_data(self, S):
        """ Define in subclass"""

    @abc.abstractmethod
    def reorganize_into_original(self, S):
        """ Define in subclass"""

    @abc.abstractmethod
    def reorganize_into_channels(self, S):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def normalize_data(self, data):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def denormalize_data(self, data):
        """ Define in subclass"""

    @abc.abstractmethod
    def reorganize_into_original_tf(self, tensor):
        """ Define in subclass"""

    @abc.abstractmethod
    def reorganize_into_channels_tf(self, tensor):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def normalize_data_tf(self, tensor):
        """ Define in subclass"""
    
    @abc.abstractmethod
    def denormalize_data_tf(self, tensor):
        """ Define in subclass"""
    

class Conv2D_AE_Normalizer_ChannelScale(Conv2D_AE_Normalizer_Base):

    def __init__(self):
        super().__init__()

        self.ch1_factor=None
        self.ch2_factor=None
        self.div_coeff=None
        self.div_coeff_tf=None
        self.crop_mat_tf=None
        self.crop_mat_scp=None

        self.ids_order_to_orig=[]
        for i in range(48//4):
            self.ids_order_to_orig.append(i*2)
            self.ids_order_to_orig.append((i*2)+1)
            self.ids_order_to_orig.append(24+i*2)
            self.ids_order_to_orig.append((24+i*2)+1)

        self.ids_order_to_chan=[]
        for i in range(48//4):
            self.ids_order_to_chan.append(i*4)
            self.ids_order_to_chan.append((i*4)+1)
        for i in range(48//4):
            self.ids_order_to_chan.append((i*4)+2)
            self.ids_order_to_chan.append((i*4)+3)

    def configure_normalization_data(self, S, crop_mat_tf, crop_mat_scp):
        ch1_max=np.max(S[:,0::2])
        ch1_min=np.min(S[:,0::2])
        if abs(ch1_max)>abs(ch1_min):
            self.ch1_factor=ch1_max
        else:
            self.ch1_factor=ch1_min
        ch2_max=np.max(S[:,1::2])
        ch2_min=np.min(S[:,1::2])
        if abs(ch2_max)>abs(ch2_min):
            self.ch2_factor=ch2_max
        else:
            self.ch2_factor=ch2_min

        aux=np.arange(S.shape[1])
        self.div_coeff=np.expand_dims(np.where(aux%2==0, self.ch1_factor, self.ch2_factor),axis=0)
        self.div_coeff_tf=tf.constant(np.expand_dims(np.where(aux%2==0, self.ch1_factor, self.ch2_factor),axis=0))

        self.crop_mat_tf=crop_mat_tf
        self.crop_mat_scp=crop_mat_scp

    def reorganize_into_original(self, S):
        S_aux=S.copy().reshape((S.shape[0],48)) # Not the prettiest, with hard-coded dimensions
        S_rearr=[]
        for s in S_aux:
            S_rearr.append(s[self.ids_order_to_orig])
        S_rearr=np.array(S_rearr)
        return S_rearr

    def reorganize_into_original_tf(self, tensor):
        output_tensor=tf.reshape(tensor, (tensor.shape[0],48)) # Not the prettiest, with hard-coded dimensions
        output_tensor=tf.gather(output_tensor, indices=self.ids_order_to_orig, axis=1)
        return output_tensor
    
    def reorganize_into_channels(self, S):
        S_rearr=[]
        for s in S:
            S_rearr.append(s[self.ids_order_to_chan])
        S_rearr=np.array(S_rearr)
        S_rearr=S_rearr.reshape((S_rearr.shape[0],2,12,2)) # Not the prettiest, with hard-coded dimensions
        return S_rearr
    
    def reorganize_into_channels_tf(self, tensor):
        output_tensor=tf.gather(tensor, indices=self.ids_order_to_chan, axis=1)
        output_tensor=tf.reshape(output_tensor, (output_tensor.shape[0],2,12,2)) # Not the prettiest, with hard-coded dimensions
        return output_tensor

    def normalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]/self.ch1_factor
        output_data[:,1::2]=output_data[:,1::2]/self.ch2_factor
        output_data=np.transpose(self.crop_mat_scp.transpose().dot(output_data.T))
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor/self.div_coeff
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, output_tensor, adjoint_a=True, adjoint_b=True))
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        output_data=np.transpose(self.crop_mat_scp.dot(output_data.T))
        output_data[:,0::2]=output_data[:,0::2]*self.ch1_factor
        output_data[:,1::2]=output_data[:,1::2]*self.ch2_factor
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tf.transpose(tf.sparse.sparse_dense_matmul(self.crop_mat_tf, tensor, adjoint_a=False, adjoint_b=True))
        output_tensor=output_tensor*self.div_coeff
        return output_tensor


""" class Conv2D_AE_Normalizer_ChannelRange(Conv2D_AE_Normalizer_Base):

    def __init__(self):
        super().__init__()
        self.ch1_max=None
        self.ch1_min=None
        self.ch2_max=None
        self.ch2_min=None
        self.subt_term=None
        self.div_coeff=None

        self.ids_order_to_orig=[]
        for i in range(48//4):
            self.ids_order_to_orig.append(i*2)
            self.ids_order_to_orig.append((i*2)+1)
            self.ids_order_to_orig.append(24+i*2)
            self.ids_order_to_orig.append((24+i*2)+1)

        self.ids_order_to_chan=[]
        for i in range(48//4):
            self.ids_order_to_chan.append(i*4)
            self.ids_order_to_chan.append((i*4)+1)
        for i in range(48//4):
            self.ids_order_to_chan.append((i*4)+2)
            self.ids_order_to_chan.append((i*4)+3)
        
    def configure_normalization_data(self, S):
        self.ch1_max=np.max(S[:,0::2])
        self.ch1_min=np.min(S[:,0::2])
        self.ch2_max=np.max(S[:,1::2])
        self.ch2_min=np.min(S[:,1::2])
        aux=np.arange(S.shape[1])
        self.subt_term=np.expand_dims(np.where(aux%2==0, self.ch1_min, self.ch2_min),axis=0)
        self.div_coeff=np.expand_dims(np.where(aux%2==0, self.ch1_max-self.ch1_min, self.ch2_max-self.ch2_min),axis=0)
    
    def reorganize_into_original(self, S):
        S_aux=S.copy().reshape((S.shape[0],48)) # Not the prettiest, with hard-coded dimensions
        S_rearr=[]
        for s in S_aux:
            S_rearr.append(s[self.ids_order_to_orig])
        S_rearr=np.array(S_rearr)
        return S_rearr

    def reorganize_into_original_tf(self, tensor):
        output_tensor=tf.reshape(tensor, (tensor.shape[0],48)) # Not the prettiest, with hard-coded dimensions
        output_tensor=tf.gather(output_tensor, indices=self.ids_order_to_orig, axis=1)
        return output_tensor
    
    def reorganize_into_channels(self, S):
        S_rearr=[]
        for s in S:
            S_rearr.append(s[self.ids_order_to_chan])
        S_rearr=np.array(S_rearr)
        S_rearr=S_rearr.reshape((S_rearr.shape[0],2,12,2)) # Not the prettiest, with hard-coded dimensions
        return S_rearr
    
    def reorganize_into_channels_tf(self, tensor):
        output_tensor=tf.gather(tensor, indices=self.ids_order_to_chan, axis=1)
        output_tensor=tf.reshape(output_tensor, (output_tensor.shape[0],2,12,2)) # Not the prettiest, with hard-coded dimensions
        return output_tensor

    def normalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=(output_data[:,0::2]-self.ch1_min)/(self.ch1_max-self.ch1_min)
        output_data[:,1::2]=(output_data[:,1::2]-self.ch2_min)/(self.ch2_max-self.ch2_min)
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor-self.subt_term
        output_tensor=output_tensor/self.div_coeff
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]*(self.ch1_max-self.ch1_min)+self.ch1_min
        output_data[:,1::2]=output_data[:,1::2]*(self.ch2_max-self.ch2_min)+self.ch2_min
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*self.div_coeff
        output_tensor=output_tensor+self.subt_term
        return output_tensor """
    
    
""" class Conv2D_AE_Normalizer_FeatureStand(Conv2D_AE_Normalizer_Base):

    def __init__(self):
        super().__init__()

        self.feat_means = None
        self.feat_stds = None

        self.ids_order_to_orig=[]
        for i in range(48//4):
            self.ids_order_to_orig.append(i*2)
            self.ids_order_to_orig.append((i*2)+1)
            self.ids_order_to_orig.append(24+i*2)
            self.ids_order_to_orig.append((24+i*2)+1)

        self.ids_order_to_chan=[]
        for i in range(48//4):
            self.ids_order_to_chan.append(i*4)
            self.ids_order_to_chan.append((i*4)+1)
        for i in range(48//4):
            self.ids_order_to_chan.append((i*4)+2)
            self.ids_order_to_chan.append((i*4)+3)

    def configure_normalization_data(self, S):
        feat_means = []
        feat_stds = []
        print('Normalizing each feature in S')
        S_df = pd.DataFrame(S)
        for i in range(len(S_df.columns)):
            feat_means.append(S_df[i].mean())
            feat_stds.append(S_df[i].std())
        self.feat_means = feat_means
        self.feat_stds = feat_stds

    def reorganize_into_original(self, S):
        S_aux=S.copy().reshape((S.shape[0],48)) # Not the prettiest, with hard-coded dimensions
        S_rearr=[]
        for s in S_aux:
            S_rearr.append(s[self.ids_order_to_orig])
        S_rearr=np.array(S_rearr)
        return S_rearr

    def reorganize_into_original_tf(self, tensor):
        output_tensor=tf.reshape(tensor, (tensor.shape[0],48)) # Not the prettiest, with hard-coded dimensions
        output_tensor=tf.gather(output_tensor, indices=self.ids_order_to_orig, axis=1)
        return output_tensor
    
    def reorganize_into_channels(self, S):
        S_rearr=[]
        for s in S:
            S_rearr.append(s[self.ids_order_to_chan])
        S_rearr=np.array(S_rearr)
        S_rearr=S_rearr.reshape((S_rearr.shape[0],2,12,2)) # Not the prettiest, with hard-coded dimensions
        return S_rearr
    
    def reorganize_into_channels_tf(self, tensor):
        output_tensor=tf.gather(tensor, indices=self.ids_order_to_chan, axis=1)
        output_tensor=tf.reshape(output_tensor, (output_tensor.shape[0],2,12,2)) # Not the prettiest, with hard-coded dimensions
        return output_tensor
    
    def normalize_data(self, data):
        output_data=data.copy()
        for i in range(output_data.shape[1]):
            output_data[:,i]=(output_data[:,i]-self.feat_means[i])/(self.feat_stds[i]+0.00000001)
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor-np.array([self.feat_means])
        output_tensor=output_tensor/(np.array([self.feat_stds])+0.00000001)
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        for i in range(output_data.shape[1]):
            output_data[:,i]= (output_data[:,i]*self.feat_stds[i])+self.feat_means[i]
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*np.array([self.feat_stds])
        output_tensor=output_tensor+np.array([self.feat_means])
        return output_tensor """