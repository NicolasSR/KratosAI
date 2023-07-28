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

class NN_Features_Evaluator():

    def __init__(self, working_path, model_path, best, test_large=False):
        self.working_path=working_path
        self.model_path=working_path+model_path
        if best=='x':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_x_')
            self.best_name_part='_bestx_'
        elif best=='r':
            self.model_weights_path=self.model_path+'best/'
            self.model_weights_filename=self.get_last_best_filename(self.model_weights_path, 'weights_r_')
            self.best_name_part='_bestr_'
        elif best is None:
            self.model_weights_path=self.model_path
            self.model_weights_filename='model_weights.h5'
            self.best_name_part=''
        else:
            print('Value for --best argument is not recognized. Terminating')
            exit()

        with open(self.model_path+"ae_config.npy", "rb") as ae_config_file:
            self.ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()
        print(self.ae_config)
        self.dataset_path=working_path+self.ae_config['dataset_path']

        self.test_large=test_large
        if self.test_large:
            self.name_complement='_test_large_'
        else:
            self.name_complement = ''

    def get_last_best_filename(self, model_weights_path, prefix):
        matching_files = [file for file in os.listdir(model_weights_path) if file.startswith(prefix)]
        highest_filename = sorted(matching_files, key=lambda x: int(x[len(prefix):][:-len('.h5')]))[-1]
        return highest_filename

    def prepare_input_finetune(self):
        S_flat_orig=np.load(self.dataset_path+'FOM.npy')

        S_flat_orig_test=np.load(self.dataset_path+'S_finetune_test.npy')

        F_test=np.load(self.dataset_path+'F_finetune_test.npy')

        return S_flat_orig, S_flat_orig_test, F_test
    
    def prepare_input_test_large(self):
        S_flat_orig=np.load(self.dataset_path+'FOM.npy')

        S_flat_orig_test=np.load(self.dataset_path+'S_test_large.npy')

        F_test=np.load(self.dataset_path+'F_test_large.npy')

        return S_flat_orig, S_flat_orig_test, F_test
    
    def network_factory_selector(self, nn_type):
        if 'conv2d' in nn_type:
            return Conv2D_AE_Factory()
        elif 'dense' in nn_type:
            return Dense_AE_Factory()
        else:
            print('No valid network type was selected')
            return None

    def calculate_X_norm_error(self, S_input):
        plt.boxplot(S_input)
        plt.show()
        S_pred_norm=self.autoencoder(S_input).numpy()
        plt.boxplot(np.abs((S_input-S_pred_norm)/S_input))
        plt.show()
        l2_error=mean_relative_l2_error(S_input,S_pred_norm)
        forb_error=relative_forbenius_error(S_input,S_pred_norm)
        return [l2_error, forb_error]
    
    def get_F_parsed(self, F):
        num_forces=np.unique(F[0],axis=0).shape[0]
        F_parsed=[]
        for snap in range(F.shape[0]):
            F_snap_unique=np.unique(F[snap],axis=0)
            F_parsed.append(np.diagonal(F_snap_unique))
        force_names=[]
        for i in range(num_forces):
            force_names.append('force_'+str(i))
        F_parsed=np.array(F_parsed, copy=False)
        return F_parsed, force_names

    def get_x_error_image(self, S, S_flat_orig, F_parsed, force_names):
        
        S_pred = self.autoencoder(S).numpy()
        S_pred_denorm = self.data_normalizer.process_input_to_raw_format(S_pred)
        err_df = pd.DataFrame(np.abs(S_pred_denorm-S_flat_orig)/np.abs(S_flat_orig+1e-14))
        forces_df = pd.DataFrame(F_parsed, columns=force_names)
        err_df = pd.concat([err_df, forces_df], axis=1)
        err_df=err_df.sort_values(by=force_names)
        print(err_df.describe())

        print('MAX ERROR ON DISPLACEMENT', np.max(np.abs(S_pred_denorm-S_flat_orig)))

        S_err_image=err_df.iloc[:,:-len(force_names)].to_numpy()
        # im1 = ax1.imshow(image, extent=[1,S_flat_orig.shape[1],np.min(np.abs(F[:,1])),np.max(np.abs(F[:,1]))], interpolation='none')
        # # im1 = ax1.imshow(image, extent=[1,S_flat_orig.shape[1],np.min(np.abs(F[:,1])),np.max(np.abs(F[:,1]))], interpolation='none', vmin=-0.001, vmax=0.0015)
        # ax1.set_aspect(1/2e4)
        # cbar1 = plt.colorbar(im1)
        # plt.xlabel('index')
        # plt.ylabel('force')
        # plt.title('Displacement Abs Error')
        # plt.show()
        F_reordered=err_df.iloc[:,-len(force_names):].to_numpy()

        return S_err_image, F_reordered


    def plot_embeddings(self, encoder, S, F):
        embeddings=tf.transpose(encoder(S))

        for i in range(embeddings.shape[0]):
            plt.scatter(F[:,1],embeddings[i])
        plt.show()

    
    def execute_evaluation(self):

        # Select the network to use
        network_factory = self.network_factory_selector(self.ae_config["nn_type"])

        # Select the type of preprocessimg (normalisation). This also decides if cropping the snapshot is needed
        self.data_normalizer=network_factory.normalizer_selector(self.working_path, self.ae_config)

        # Create a fake Analysis stage to calculate the predicted residuals
        self.residual_scale_factor=np.load(self.working_path+self.ae_config['dataset_path']+'residual_scale_factor.npy')
        self.kratos_simulation = KratosSimulator(self.working_path, self.ae_config, self.residual_scale_factor)
        crop_mat_tf, crop_mat_scp = self.kratos_simulation.get_crop_matrix()

        if self.test_large==False:
            S_flat_orig_FOM, S_flat_orig_test, F_test = self.prepare_input_finetune()
        else:
            S_flat_orig_FOM, S_flat_orig_test, F_test = self.prepare_input_test_large()
        print('Shape S_flat_orig_test:', S_flat_orig_test.shape)
        print('Shape F_test: ', F_test.shape)

        self.data_normalizer.configure_normalization_data(S_flat_orig_FOM, crop_mat_tf, crop_mat_scp)

        S_test = self.data_normalizer.process_raw_to_input_format(S_flat_orig_test)
        print('Shape S_test: ', S_test.shape)

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        self.autoencoder, encoder, decoder = network_factory.define_network(S_test, self.ae_config)
        self.autoencoder.load_weights(self.model_weights_path+self.model_weights_filename)
        self.autoencoder.set_config_values_eval(self.data_normalizer)

        # encoder_bin = tf.keras.models.load_model(self.model_path+'encoder_model_bestr/')
        # decoder_bin = tf.keras.models.load_model(self.model_path+'decoder_model_bestr/')
        # S_test_encoded = encoder(S_test)
        # S_test_encoded_bin = encoder_bin(S_test)
        # print('Comparing encoders')
        # print(np.all(S_test_encoded==S_test_encoded_bin))

        F_parsed, force_names = self.get_F_parsed(F_test)
        # F_parsed, force_names = self.get_F_parsed(F_test)

        # Test errors
        x_norm_err_matrix=self.calculate_X_norm_error(S_test)
        print(x_norm_err_matrix)
        exit()
        np.save(self.model_path+'x_norm_errs'+self.best_name_part+self.name_complement+'.npy',x_norm_err_matrix)

        s_err_image, F_parsed_reordered = self.get_x_error_image(S, S_flat_orig, F_parsed, force_names)
        # s_err_image, F_parsed_reordered = self.get_x_error_image(S_test, S_flat_orig_test, F_parsed, force_names)
        np.save(self.model_path+'s_err_image'+self.best_name_part+self.name_complement+'.npy',s_err_image)


        """ comp_inf_norm = np.argmax(np.sum(np.abs(s_err_image), axis=1))
        print('comp_inf_norm', comp_inf_norm)

        dot_value = np.sum(np.abs(s_err_image), axis=1)
        # idxs=np.argwhere(F_parsed_reordered[:,1]<-2.5e7)
        # idxs=np.argwhere(dot_value>500)
        # idxs=3485
        # sc=plt.scatter(F_parsed_reordered[idxs,0], F_parsed_reordered[idxs,1], c=dot_value[idxs], s=4, cmap='jet')
        sc=plt.scatter(F_parsed_reordered[:,0], F_parsed_reordered[:,1], c=dot_value, s=4, cmap='jet')
        plt.colorbar(sc)
        plt.show()

        exit() """

        R_true_scaled_test, R_true_scaled_train, R_pred_scaled_test, R_pred_scaled_train, R_err = self.get_r_matrices(S, S_test, S_train, R, R_test, R_train)

        """ print(R[4200])
        R_true_scaled = np.concatenate((R_true_scaled_train, R_true_scaled_test), axis=0)
        R_pred_scaled = np.concatenate((R_pred_scaled_train, R_pred_scaled_test), axis=0)
        print(np.sum(R_true_scaled[4200]))
        print(np.sum(R_pred_scaled[4200]))
        exit() """


        r_norm_err_matrix=[]
        # Test errors
        r_norm_err_matrix.append(self.calculate_R_norm_error(R_true_scaled_test,R_pred_scaled_test))
        # Train errors
        r_norm_err_matrix.append(self.calculate_R_norm_error(R_true_scaled_train,R_pred_scaled_train))
        r_norm_err_matrix=np.array(r_norm_err_matrix)

        print(r_norm_err_matrix)
        np.save(self.model_path+'r_norm_errs'+self.best_name_part+self.name_complement+'.npy',r_norm_err_matrix)

        r_err_image = self.get_r_error_image(R_err, F_parsed, force_names)
        np.save(self.model_path+'r_err_image'+self.best_name_part+self.name_complement+'.npy',r_err_image)

        """ # dot_value = np.sum(np.abs(r_err_image), axis=1)
        dot_value = np.sum(R_err, axis=1)
        # idxs=np.argwhere(F_parsed_reordered[:,1]<-2.5e7)
        # idxs=np.argwhere(dot_value>500)
        # sc=plt.scatter(F_parsed_reordered[idxs,0], F_parsed_reordered[idxs,1], c=dot_value[idxs], s=4, cmap='jet')
        sc=plt.scatter(F_parsed_reordered[:,0], F_parsed_reordered[:,1], c=dot_value, s=4, cmap='jet')
        plt.colorbar(sc)
        plt.show() """


        
        # max_F=np.max(abs(F[:,1]))
        # min_F=np.min(abs(F[:,1]))

        # print(max_F)
        # print(min_F)

        # sample_min = np.argmin(abs(F[:,1]))
        # print(F[sample_min])
        # self.print_x_and_r_vecs(S_flat_orig, S_flat, S, F, R, sample_min)

        # sample_id = np.argmin(abs(F[:,1]+max_F/2))
        # print(sample_id)
        # print(F[sample_id])
        # self.print_x_and_r_vecs(S_flat_orig, S_flat, S, F, R, sample_id)

        # sample_max = np.argmax(abs(F[:,1]))
        # print(sample_max)
        # print(F[sample_max])
        # self.print_x_and_r_vecs(S_flat_orig, S_flat, S, F, R, sample_max)

        

        # r_error_to_draw = np.concatenate([r_error_train, r_error_test], axis=0)
        # f_to_draw = np.concatenate([F_train, F_test], axis=0)

        # self.draw_r_error_image(r_error_to_draw,f_to_draw)

        # print(self.ae_config)

        
if __name__ == "__main__":

    nn_evaluator = NN_Features_Evaluator()

    nn_evaluator.execute_evaluation()