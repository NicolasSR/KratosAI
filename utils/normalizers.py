import numpy as np
import pandas as pd
import tensorflow as tf
import abc

from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
import matplotlib.pyplot as plt


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
        
        print('Phi matrix shape: ', self.phi.shape)
        self.phi_tf=tf.constant(self.phi)
        
        S_svd=np.matmul(self.phi.T,S.T).T
        self.feat_factors = []
        print('Scaling each feature in S_svd')
        abs_S_df = pd.DataFrame(np.abs(S_svd))
        for i in range(len(abs_S_df.columns)):
            self.feat_factors.append(abs_S_df[i].max())
        self.feat_factors_tf = tf.constant(self.feat_factors)

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
        self.factor_ch1=None
        self.factor_ch2=None
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
    
class AE_Normalizer_ChannelRange(AE_Normalizer_Base):
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
        return output_tensor
  
  
class AE_Normalizer_FeatureStand(AE_Normalizer_Base):
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
        return output_tensor
    
    
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
        print('CROPPING NOT IMPLEMENTED PROPERLY FOR THIS NORMALISATION. ABORING')
        exit()

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


class Conv2D_AE_Normalizer_ChannelRange(Conv2D_AE_Normalizer_Base):

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
        return output_tensor
    

class Conv2D_AE_Normalizer_ChannelScale(Conv2D_AE_Normalizer_Base):

    def __init__(self):
        super().__init__()

        self.factor_ch1=None
        self.factor_ch2=None
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
        return output_data
    
    def normalize_data_tf(self,tensor):
        output_tensor=tensor/self.div_coeff
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]*self.ch1_factor
        output_data[:,1::2]=output_data[:,1::2]*self.ch2_factor
        return output_data
    
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*self.div_coeff
        return output_tensor
    
    
class Conv2D_AE_Normalizer_FeatureStand(Conv2D_AE_Normalizer_Base):

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
        return output_tensor