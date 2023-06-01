import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from utils.kratos_simulation import KratosSimulator

from networks.conv2d_ae_factory import  Conv2D_AE_Factory
from networks.dense_ae_factory import Dense_AE_Factory
from utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error

tf.keras.backend.set_floatx('float64')

class NN_Evaluator():

    def prepare_input_finetune(self, dataset_path, use_force, cropped_ids):
        S_flat_orig=np.load(dataset_path+'FOM.npy')
        S_flat_orig_train=np.load(dataset_path+'S_finetune_train.npy')
        S_flat_orig_test=np.load(dataset_path+'S_finetune_test.npy')

        if use_force is None or use_force == True:
            R_train=np.load(dataset_path+'R_finetune_train.npy')
            R_test=np.load(dataset_path+'R_finetune_test.npy')
        else:
            print('Not using Force')
            R_train=np.load(dataset_path+'R_finetune_noF_train.npy')
            R_test=np.load(dataset_path+'R_finetune_noF_test.npy')
        F_train=np.load(dataset_path+'F_finetune_train.npy')[:,0,:]
        F_test=np.load(dataset_path+'F_finetune_test.npy')[:,0,:]

        S_flat_orig=np.delete(S_flat_orig, cropped_ids, axis=1)
        S_flat_orig_train=np.delete(S_flat_orig_train, cropped_ids, axis=1)
        S_flat_orig_test=np.delete(S_flat_orig_test, cropped_ids, axis=1)

        return S_flat_orig, S_flat_orig_train, S_flat_orig_test, R_train, R_test, F_train, F_test
    
    def network_factory_selector(self, nn_type):
        if 'conv2d' in nn_type:
            return Conv2D_AE_Factory()
        elif 'dense' in nn_type:
            return Dense_AE_Factory()
        else:
            print('No valid network type was selected')
            return None
        
    def calculate_R_norm_error(self, S_input, R_true, F_true):
        print('Scale factor:', self.residual_scale_factor)
        R_true=R_true/self.residual_scale_factor
        S_pred_norm=self.autoencoder(S_input).numpy()
        S_pred_denorm = self.data_normalizer.process_input_to_raw_format(S_pred_norm)
        R_pred=self.kratos_simulation.get_r_array(S_pred_denorm, F_true)
        l2_error=mean_relative_l2_error(R_true,R_pred)
        forb_error=relative_forbenius_error(R_true,R_pred)
        print('Residual. Mean rel L2 error l:', l2_error)
        print('Residual. Rel Forb. error l:', forb_error)
        r_error = R_pred-R_true
        return r_error

    def calculate_X_norm_error(self, S_true, S_input):
        S_pred_norm=self.autoencoder(S_input).numpy()
        S_pred_denorm = self.data_normalizer.process_input_to_raw_format(S_pred_norm)
        l2_error=mean_relative_l2_error(S_true,S_pred_denorm)
        forb_error=relative_forbenius_error(S_true,S_pred_denorm)
        print('X. Mean rel L2 error:', l2_error)
        print('X. Rel Forb. error:', forb_error)

    def draw_x_error_image(self, S, S_flat_orig, F):
        S_pred = self.autoencoder(S).numpy()
        S_pred_denorm = self.data_normalizer.process_input_to_raw_format(S_pred)
        # err_df = pd.DataFrame(S_pred_flat-S_flat)
        err_df = pd.DataFrame(S_pred_denorm-S_flat_orig)
        err_df['force']=np.abs(F[:,1])
        err_df=err_df.sort_values('force')
        print(err_df)
        print(err_df.describe())
        fig, (ax1) = plt.subplots(ncols=1)
        image=err_df.iloc[:,:-1].to_numpy()
        im1 = ax1.imshow(image, extent=[1,S_flat_orig.shape[1],np.min(np.abs(F[:,1])),np.max(np.abs(F[:,1]))], interpolation='none')
        # im1 = ax1.imshow(image, extent=[1,S_flat_orig.shape[1],np.min(np.abs(F[:,1])),np.max(np.abs(F[:,1]))], interpolation='none', vmin=-0.001, vmax=0.0015)
        ax1.set_aspect(1/2e5)
        cbar1 = plt.colorbar(im1)
        plt.xlabel('index')
        plt.ylabel('force')
        plt.title('Displacement Abs Error')
        plt.show()

    def draw_r_error_image(self, R_err, F):
        err_df = pd.DataFrame(R_err)
        err_df['force']=np.abs(F[:,1])
        err_df=err_df.sort_values('force')
        print(err_df)
        print(err_df.describe())
        fig, (ax1) = plt.subplots(ncols=1)
        image=err_df.iloc[:,:-1].to_numpy()
        im1 = ax1.imshow(image, extent=[1,R_err.shape[1],np.min(np.abs(F[:,1])),np.max(np.abs(F[:,1]))], interpolation='none', vmin=-0.02, vmax=0.02)
        ax1.set_aspect(1/2e5)
        cbar1 = plt.colorbar(im1)
        plt.xlabel('index')
        plt.ylabel('force')
        plt.title('Residual Abs Error')
        plt.show()

    def print_x_and_r_vecs(self, S_flat_orig, S_flat, S, F, R, sample_id):
        x_orig=np.expand_dims(S_flat_orig[sample_id], axis=0)
        x_norm=np.expand_dims(S_flat[sample_id], axis=0)
        x_pred_norm=self.autoencoder(np.expand_dims(S[sample_id], axis=0)).numpy()
        x_pred_flat=self.data_normalizer.process_input_to_raw_format(x_pred_norm)
        r_orig=np.expand_dims(R[sample_id]/self.residual_scale_factor, axis=0)
        x_pred_denorm = self.data_normalizer.process_input_to_raw_format(x_pred_norm)
        r_pred=self.kratos_simulation.get_r_array(x_pred_denorm,np.expand_dims(F[sample_id], axis=0))
        print('x_orig:', x_orig)
        print('x_pred:', x_pred_flat)
        print('x_norm:', x_norm)
        print('x_pred_norm:', x_pred_norm)
        print('r_orig:', r_orig)
        print('r_pred:', r_pred)

        plt.plot(x_orig[0], '.')
        plt.plot(x_pred_flat[0], '.')
        plt.legend(['x_true','x_pred'])
        plt.ylabel('displacement')
        plt.xlabel('degree of freedom')
        plt.title('F = '+str(F[sample_id,1]))
        plt.tight_layout()
        plt.show()

        plt.plot(x_norm[0], '.')
        plt.plot(x_norm[0], '.')
        plt.legend(['x_true','x_pred'])
        plt.ylabel('normalized displacement')
        plt.xlabel('degree of freedom')
        plt.title('F = '+str(F[sample_id,1]))
        plt.tight_layout()
        plt.show()

        plt.plot(r_orig[0])
        plt.plot(r_pred[0])
        plt.legend(['r_true','r_pred'])
        plt.ylabel('residual value')
        plt.xlabel('index')
        plt.title('F = '+str(F[sample_id,1]))
        plt.tight_layout()
        plt.show()

    def plot_embeddings(self, encoder, S, F):
        embeddings=tf.transpose(encoder(S))

        for i in range(embeddings.shape[0]):
            plt.scatter(F[:,1],embeddings[i])
        plt.show()
    
    def execute_evaluation(self):

        data_path='saved_models_newexample/'
        with open(data_path+"ae_config.npy", "rb") as ae_config_file:
            self.ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()
        # self.ae_config=np.load(data_path+'ae_config.npy',allow_pickle=True)
        print(self.ae_config)

        # Select the network to use
        network_factory = self.network_factory_selector(self.ae_config["nn_type"])

        # Select the type of preprocessimg (normalisation). This also decides if cropping the snapshot is needed
        self.data_normalizer=network_factory.normalizer_selector(self.ae_config)

        # Create a fake Analysis stage to calculate the predicted residuals
        self.residual_scale_factor=np.load(self.ae_config['dataset_path']+'residual_scale_factor.npy')
        self.kratos_simulation = KratosSimulator(self.ae_config, self.data_normalizer.needs_truncation, self.residual_scale_factor)

        S_flat_orig, S_flat_orig_train, S_flat_orig_test, R_train, R_test, F_train, F_test = self.prepare_input_finetune(self.ae_config['dataset_path'], self.ae_config["use_force"], self.kratos_simulation.get_cropped_dof_ids())
        print('Shape S_flat_orig: ', S_flat_orig.shape)
        print('Shape S_flat_orig_train:', S_flat_orig_train.shape)
        print('Shape S_flat_orig_test:', S_flat_orig_test.shape)
        print('Shape R_train: ', R_train.shape)
        print('Shape R_est: ', R_test.shape)
        print('Shape F_train: ', F_train.shape)
        print('Shape F_test: ', F_test.shape)

        self.data_normalizer.configure_normalization_data(S_flat_orig)

        S_flat_orig=np.concatenate((S_flat_orig_train, S_flat_orig_test), axis=0)
        F=np.concatenate((F_train, F_test), axis=0)
        R=np.concatenate((R_train, R_test), axis=0)

        S = self.data_normalizer.process_raw_to_input_format(S_flat_orig)
        S_train = self.data_normalizer.process_raw_to_input_format(S_flat_orig_train)
        S_test = self.data_normalizer.process_raw_to_input_format(S_flat_orig_test)
        print('Shape S: ', S.shape)
        print('Shape S_train: ', S_train.shape)
        print('Shape S_test: ', S_test.shape)

        S_flat = self.data_normalizer.normalize_data(S_flat_orig)

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        self.autoencoder, encoder, decoder = network_factory.define_network(S, self.ae_config)
        self.autoencoder.load_weights(data_path+'model_weights.h5')
        self.autoencoder.set_config_values_eval(self.data_normalizer)

        print(self.ae_config)

        print(F.shape)
        print(S.shape)

        self.plot_embeddings(encoder, S, F)

        print('Test errors')
        self.calculate_X_norm_error(S_flat_orig_test, S_test)
        print('Train errors')
        self.calculate_X_norm_error(S_flat_orig_train, S_train)

        self.draw_x_error_image(S, S_flat_orig, F)

        max_F=np.max(abs(F[:,1]))
        min_F=np.min(abs(F[:,1]))

        print(max_F)
        print(min_F)

        sample_min = np.argmin(abs(F[:,1]))
        print(F[sample_min])
        self.print_x_and_r_vecs(S_flat_orig, S_flat, S, F, R, sample_min)

        sample_id = np.argmin(abs(F[:,1]+max_F/2))
        print(sample_id)
        print(F[sample_id])
        self.print_x_and_r_vecs(S_flat_orig, S_flat, S, F, R, sample_id)

        sample_max = np.argmax(abs(F[:,1]))
        print(sample_max)
        print(F[sample_max])
        self.print_x_and_r_vecs(S_flat_orig, S_flat, S, F, R, sample_max)

        print('Test errors')
        r_error_test = self.calculate_R_norm_error(S_test,R_test,F_test)
        print('Train errors')
        r_error_train = self.calculate_R_norm_error(S_train,R_train,F_train)

        r_error_to_draw = np.concatenate([r_error_train, r_error_test], axis=0)
        f_to_draw = np.concatenate([F_train, F_test], axis=0)

        self.draw_r_error_image(r_error_to_draw,f_to_draw)

        print(self.ae_config)

        
if __name__ == "__main__":

    nn_evaluator = NN_Evaluator()

    nn_evaluator.execute_evaluation()