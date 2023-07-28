import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import scipy

import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import logging
tf.get_logger().setLevel(logging.ERROR)

from utils.kratos_simulation import KratosSimulator

from networks.conv2d_ae_factory import  Conv2D_AE_Factory
from networks.dense_ae_factory import Dense_AE_Factory
from networks.linear_ae_factory import Linear_AE_Factory
from utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error

tf.keras.backend.set_floatx('float64')

class Scipy_Optimiser():

    def __init__(self, working_path, model_path, F_matrix_path, best):
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
        self.F_matrix_path=working_path+F_matrix_path

    def get_last_best_filename(self, model_weights_path, prefix):
        matching_files = [file for file in os.listdir(model_weights_path) if file.startswith(prefix)]
        highest_filename = sorted(matching_files, key=lambda x: int(x[len(prefix):][:-len('.h5')]))[-1]
        return highest_filename

    def prepare_input(self):
        S_flat_orig=np.load(self.dataset_path+'FOM.npy')

        F=np.load(self.F_matrix_path)

        return S_flat_orig, F
    
    def network_factory_selector(self, nn_type):
        if 'conv2d' in nn_type:
            return Conv2D_AE_Factory()
        elif 'dense' in nn_type:
            return Dense_AE_Factory()
        elif 'linear' in nn_type:
            return Linear_AE_Factory()
        else:
            print('No valid network type was selected')
            return None
        

    def optimisation_routine(self, q0, f_vectors, s_true):
        f_vectors=np.expand_dims(f_vectors, axis=0)
        snapshot_true=np.expand_dims(s_true, axis=0)
        f_vectors_0=np.zeros(f_vectors.shape)

        # condition_dofs_list = self.kratos_simulation.get_dofs_with_conditions()
        
        snapshot_0 = self.decoder(np.expand_dims(q0, axis=0))
        snapshot_0 = self.data_normalizer.process_input_to_raw_format(snapshot_0)
        r_vector_0 = self.kratos_simulation.get_r_forces_(snapshot_0, f_vectors)[0]
        total_forces = np.mean(f_vectors[0], axis=1)*0.6

        # plt.plot(r_vector_0)
        # plt.show()
        # r_vector_0=r_vector_0[condition_dofs_list]
        # plt.plot(r_vector_0)
        # plt.show()
        # exit()

        r_norm_0 = np.linalg.norm(r_vector_0)
        print('Init residual norm: ', r_norm_0)

        def opt_function(x):
            snapshot = self.decoder(np.expand_dims(x, axis=0))
            snapshot = self.data_normalizer.process_input_to_raw_format(snapshot)
            r_vector = self.kratos_simulation.get_r_forces_(snapshot, f_vectors)[0]
            # r_vector = np.delete(r_vector, self.crop_indices)
            r_norm = np.linalg.norm(r_vector)
            return r_norm
        
        def opt_function_r_diff(x):
            snapshot_app = self.decoder(np.expand_dims(x, axis=0))
            snapshot_app = self.data_normalizer.process_input_to_raw_format(snapshot_app)
            r_vector_app = self.kratos_simulation.get_r_forces_withDirich_(snapshot_app, f_vectors_0)[0]

            r_vector_true = self.kratos_simulation.get_r_forces_withDirich_(snapshot_true, f_vectors_0)[0]

            loss_norm = np.linalg.norm(r_vector_app-r_vector_true)
            return loss_norm
        
        def opt_function_r_comps(x):
            snapshot_app = self.decoder(np.expand_dims(x, axis=0))
            snapshot_app = self.data_normalizer.process_input_to_raw_format(snapshot_app)
            r_vector_app = self.kratos_simulation.get_r_forces_withDirich_(snapshot_app, f_vectors)[0]
            r_comps=[np.sum(r_vector_app[0::2]), np.sum(r_vector_app[1::2])]
            r_vector_app_noridich = self.kratos_simulation.get_r_forces_(snapshot_app, f_vectors)[0]
            r_norm = np.linalg.norm(r_comps)+0.05*np.linalg.norm(r_vector_app_noridich)
            return r_norm
        
        def opt_function_2(x):
            snapshot_app = self.decoder(np.expand_dims(x, axis=0))
            snapshot_app = self.data_normalizer.process_input_to_raw_format(snapshot_app)
            r_vector_app = self.kratos_simulation.get_r_forces_(snapshot_app, f_vectors)[0]
            r_comps=[np.sum(r_vector_app[0::2])+total_forces[0], np.sum(r_vector_app[1::2])+total_forces[1]]
            r_norm = np.linalg.norm(r_comps)
            return r_norm
        
        """ def opt_function_r_loads(x):
            snapshot = self.decoder(np.expand_dims(x, axis=0))
            snapshot = self.data_normalizer.process_input_to_raw_format(snapshot)
            r_vector = self.kratos_simulation.get_r_forces_(snapshot, f_vectors)[0][condition_dofs_list]
            # r_vector = np.delete(r_vector, self.crop_indices)
            r_norm = np.linalg.norm(r_vector)
            return r_norm """

        # q_optim = scipy.optimize.minimize(opt_function, q0, method='L-BFGS-B')

        # q_optim = scipy.optimize.minimize(opt_function_r_diff, q0, method='L-BFGS-B')

        q_optim = scipy.optimize.minimize(opt_function_2, q0, method='L-BFGS-B')


        snapshot_final = self.decoder(np.expand_dims(q_optim.x, axis=0))
        snapshot_final = self.data_normalizer.process_input_to_raw_format(snapshot_final)
        r_vector_final = self.kratos_simulation.get_r_forces_(snapshot_final, f_vectors)
        final_r_norm = np.linalg.norm(r_vector_final)

        reactions = -1*self.kratos_simulation.get_r_forces_withDirich_(snapshot_final, f_vectors)
        
        return r_norm_0, reactions, q_optim, final_r_norm

    
    def execute_optimisation(self):

        # Select the network to use
        network_factory = self.network_factory_selector(self.ae_config["nn_type"])

        # Select the type of preprocessimg (normalisation). This also decides if cropping the snapshot is needed
        self.data_normalizer=network_factory.normalizer_selector(self.working_path, self.ae_config)

        # Create a fake Analysis stage to calculate the predicted residuals
        # self.residual_scale_factor=np.load(self.working_path+self.ae_config['dataset_path']+'residual_scale_factor.npy')
        self.residual_scale_factor=1.0
        self.kratos_simulation = KratosSimulator(self.working_path, self.ae_config, self.residual_scale_factor)
        crop_mat_tf, crop_mat_scp = self.kratos_simulation.get_crop_matrix()

        S_flat_orig_FOM, F = self.prepare_input()

        print('Shape S_flat_orig_FOM: ', S_flat_orig_FOM.shape)
        print('Shape F: ', F.shape)

        self.data_normalizer.configure_normalization_data(S_flat_orig_FOM, crop_mat_tf, crop_mat_scp)

        S = self.data_normalizer.process_raw_to_input_format(S_flat_orig_FOM)
        print('Shape S: ', S.shape)

        # Load the autoencoder model
        print('======= Instantiating new autoencoder =======')
        self.autoencoder, self.encoder, self.decoder = network_factory.define_network(S, self.ae_config)
        self.autoencoder.load_weights(self.model_weights_path+self.model_weights_filename)
        self.autoencoder.set_config_values_eval(self.data_normalizer)

        # F_parsed, force_names = self.get_F_parsed(F)

        S_true = np.load('Scipy_FOM_x_snapshots.npy')
        S_init = np.load('FOM_init_snaps_30steps.npy')

        # self.crop_indices=np.array([0,1,5,10,16,24,34,46,61,77,94,113,133,155,182,209,238,267,298,329,362])
        # self.crop_indices=np.concatenate([self.crop_indices*2,self.crop_indices*2+1])
        # print(self.crop_indices)

        # for i, forces in enumerate(F):
        #     r_vector = self.kratos_simulation.get_r_forces_(np.expand_dims(S_true[i],axis=0), forces)
        #     print(r_vector.shape)
        #     r_vector = np.delete(r_vector, self.crop_indices, axis=1)
        #     print(r_vector.shape)
        #     print(r_vector)
        #     plt.plot(r_vector[0])
        #     plt.show()
        #     exit()

        # q0 = np.array([0,0])
        # q0 = np.zeros(20)
        # q0 = self.data_normalizer.process_raw_to_input_format(np.expand_dims(S_init[0], axis=0))
        # q0 = self.encoder(q0)[0]
        reduced_snapshots_matrix=[]
        residual_norms_list=[]
        reactions_matrix=[]
        for i, forces in enumerate(F):

            # r_vector_0 = self.kratos_simulation.get_r_forces_(np.expand_dims(S_init[i],axis=0), forces)
            # r_vector_0 = np.delete(r_vector_0, self.crop_indices)
            # r_norm_0 = np.linalg.norm(r_vector_0)
            # print(r_norm_0)

            q0 = self.data_normalizer.process_raw_to_input_format(np.expand_dims(S_init[i], axis=0))
            q0 = self.encoder(q0)[0]
            # q0 = np.zeros(2)
            print(forces.shape)
            print(q0.shape)
            res_norm_init, reactions, opt_result, final_r_norm  = self.optimisation_routine(q0, forces, S_true[i])
            print(opt_result)
            print('Final residual norm: ', final_r_norm)
            print('Relative residual norms', abs(final_r_norm/res_norm_init))
            q=np.array(opt_result.x)
            print(q)
            reduced_snapshots_matrix.append(q)
            residual_norms_list.append([res_norm_init,final_r_norm])
            reactions_matrix.append(reactions)
            # q0=q

        reduced_snapshots_matrix=np.array(reduced_snapshots_matrix, copy=False)
        print(reduced_snapshots_matrix.shape)
        np.save('Scipy_q_snapshots.npy', reduced_snapshots_matrix)

        snapshots_matrix = self.decoder(reduced_snapshots_matrix)
        snapshots_matrix = self.data_normalizer.process_input_to_raw_format(snapshots_matrix)
        print(snapshots_matrix.shape)
        np.save('Scipy_x_snapshots.npy', snapshots_matrix)

        np.save('Scipy_residual_norms.npy', residual_norms_list)
        np.save('Scipy_reactions_snapshots.npy', reactions_matrix)
        
if __name__ == "__main__":

    model_path = 'Finetune_Dense_extended_RMain_w0.1_lay40_LRe5_svd_white_nostand_1000ep'
    F_matrix_path = 'FOM_POINTLOADS_30steps.npy'

    scipy_optimiser = Scipy_Optimiser('', 'saved_models_newexample/'+model_path+'/', F_matrix_path, 'r')

    scipy_optimiser.execute_optimisation()