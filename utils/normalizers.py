import numpy as np
import pandas as pd
import tensorflow as tf

class AE_Normalizer_ChannelRange():
    def __init__(self):
        self.ch1_max=None
        self.ch1_min=None
        self.ch2_max=None
        self.ch2_min=None
        self.subt_term=None
        self.div_coeff=None

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
    
    @tf.function
    def normalize_data_tf(self,tensor):
        output_tensor=tensor-self.subt_term
        output_tensor=output_tensor/self.div_coeff
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]*(self.ch1_max-self.ch1_min)+self.ch1_min
        output_data[:,1::2]=output_data[:,1::2]*(self.ch2_max-self.ch2_min)+self.ch2_min
        return output_data
    
    @tf.function
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*self.div_coeff
        output_tensor=output_tensor+self.subt_term
        return output_tensor
    
    
class AE_Normalizer_ChannelScale():
    def __init__(self):
        self.factor_ch1=None
        self.factor_ch2=None
        self.subt_term=None
        self.div_coeff=None

    def configure_normalization_data(self, S):
        ch1_max=np.max(S[:,:,:,1])
        ch1_min=np.min(S[:,:,:,1])
        if abs(ch1_max)>abs(ch1_min):
            self.ch1_factor=ch1_max
        else:
            self.ch1_factor=ch1_min
        ch2_max=np.max(S[:,:,:,1])
        ch2_min=np.min(S[:,:,:,1])
        if abs(ch2_max)>abs(ch2_min):
            self.ch2_factor=ch2_max
        else:
            self.ch2_factor=ch2_min

        aux=np.arange(S.shape[1])
        self.div_coeff=np.expand_dims(np.where(aux%2==0, self.ch1_factor, self.ch2_factor),axis=0)


    def normalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]/self.ch1_factor
        output_data[:,1::2]=output_data[:,1::2]/self.ch2_factor
        return output_data
    
    @tf.function
    def normalize_data_tf(self,tensor):
        output_tensor=tensor/self.div_coeff
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]*self.ch1_factor
        output_data[:,1::2]=output_data[:,1::2]*self.ch2_factor
        return output_data
    
    @tf.function
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*self.div_coeff
        return output_tensor
  
  
class AE_Normalizer_FeatureStand():
    def __init__(self):
        self.feat_means = None   
        self.feat_stds = None

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
    
    @tf.function
    def normalize_data_tf(self,tensor):
        output_tensor=tensor-np.array([self.feat_means])
        output_tensor=output_tensor/(np.array([self.feat_stds])+0.00000001)
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        for i in range(output_data.shape[1]):
            output_data[:,i]= (output_data[:,i]*self.feat_stds[i])+self.feat_means[i]
        return output_data
    
    @tf.function
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*np.array([self.feat_stds])
        output_tensor=output_tensor+np.array([self.feat_means])
        return output_tensor
    
    
class AE_Normalizer_None():
    def __init__(self):
        return

    def configure_normalization_data(self, S):
        return

    def normalize_data(self, data):
        output_data=data.copy()
        return output_data
    
    @tf.function
    def normalize_data_tf(self, tensor):
        output_tensor=tensor
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        return output_data
    
    @tf.function
    def denormalize_data_tf(self, tensor):
        output_tensor=tensor
        return output_tensor


class Conv2D_AE_Normalizer_ChannelRange():
    def __init__(self):
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

    @tf.function
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
    
    @tf.function
    def reorganize_into_channels_tf(self, tensor):
        output_tensor=tf.gather(tensor, indices=self.ids_order_to_chan, axis=1)
        output_tensor=tf.reshape(output_tensor, (output_tensor.shape[0],2,12,2)) # Not the prettiest, with hard-coded dimensions
        return output_tensor

    def normalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=(output_data[:,0::2]-self.ch1_min)/(self.ch1_max-self.ch1_min)
        output_data[:,1::2]=(output_data[:,1::2]-self.ch2_min)/(self.ch2_max-self.ch2_min)
        return output_data
    
    @tf.function
    def normalize_data_tf(self,tensor):
        output_tensor=tensor-self.subt_term
        output_tensor=output_tensor/self.div_coeff
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]*(self.ch1_max-self.ch1_min)+self.ch1_min
        output_data[:,1::2]=output_data[:,1::2]*(self.ch2_max-self.ch2_min)+self.ch2_min
        return output_data
    
    @tf.function
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*self.div_coeff
        output_tensor=output_tensor+self.subt_term
        return output_tensor
    

class Conv2D_AE_Normalizer_ChannelScale():
    def __init__(self):
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

    @tf.function
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
    
    @tf.function
    def reorganize_into_channels_tf(self, tensor):
        output_tensor=tf.gather(tensor, indices=self.ids_order_to_chan, axis=1)
        output_tensor=tf.reshape(output_tensor, (output_tensor.shape[0],2,12,2)) # Not the prettiest, with hard-coded dimensions
        return output_tensor

    def normalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]/self.ch1_factor
        output_data[:,1::2]=output_data[:,1::2]/self.ch2_factor
        return output_data
    
    @tf.function
    def normalize_data_tf(self,tensor):
        output_tensor=tensor/self.div_coeff
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        output_data[:,0::2]=output_data[:,0::2]*self.ch1_factor
        output_data[:,1::2]=output_data[:,1::2]*self.ch2_factor
        return output_data
    
    @tf.function
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*self.div_coeff
        return output_tensor
    
    
class Conv2D_AE_Normalizer_FeatureStand():
    def __init__(self):
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

    @tf.function
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
    
    @tf.function
    def reorganize_into_channels_tf(self, tensor):
        output_tensor=tf.gather(tensor, indices=self.ids_order_to_chan, axis=1)
        output_tensor=tf.reshape(output_tensor, (output_tensor.shape[0],2,12,2)) # Not the prettiest, with hard-coded dimensions
        return output_tensor
    
    def normalize_data(self, data):
        output_data=data.copy()
        for i in range(output_data.shape[1]):
            output_data[:,i]=(output_data[:,i]-self.feat_means[i])/(self.feat_stds[i]+0.00000001)
        return output_data
    
    @tf.function
    def normalize_data_tf(self,tensor):
        output_tensor=tensor-np.array([self.feat_means])
        output_tensor=output_tensor/(np.array([self.feat_stds])+0.00000001)
        return output_tensor

    def denormalize_data(self, data):
        output_data=data.copy()
        for i in range(output_data.shape[1]):
            output_data[:,i]= (output_data[:,i]*self.feat_stds[i])+self.feat_means[i]
        return output_data
    
    @tf.function
    def denormalize_data_tf(self,tensor):
        output_tensor=tensor*np.array([self.feat_stds])
        output_tensor=output_tensor+np.array([self.feat_means])
        return output_tensor