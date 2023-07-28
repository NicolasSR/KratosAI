import numpy as np

from utils.kratos_simulation import KratosSimulator
from utils.custom_metrics import mean_relative_l2_error, relative_forbenius_error
from networks.dense_ae_factory import Dense_AE_Factory

def normalize_data(data, crop_mat_scp, phi):
    output_data=np.transpose(crop_mat_scp.transpose().dot(data.T))
    output_data=np.matmul(phi.T,output_data.T).T
    return output_data

def denormalize_data(data, crop_mat_scp, phi):
    output_data=np.matmul(phi, data.T).T
    output_data=np.transpose(crop_mat_scp.dot(output_data.T))
    return output_data

if __name__ == "__main__":

    S_test=np.load('datasets_two_forces_dense_lowforce/S_test_large.npy')
    Phi=np.load('datasets_two_forces_dense_lowforce/svd_phi_white_nostand.npy')[:,:10]

    # Select the network to use
    network_factory = Dense_AE_Factory()

    ae_config={
        "normalization_strategy": 'svd_white_nostand',
        "dataset_path": 'datasets_two_forces_dense_lowforce/',
        "project_parameters_file":'ProjectParameters_fom.json'
   }

    # Select the type of preprocessimg (normalisation). This also decides if cropping the snapshot is needed
    data_normalizer=network_factory.normalizer_selector('', ae_config)

    # Create a fake Analysis stage to calculate the predicted residuals
    kratos_simulation = KratosSimulator('', ae_config, 1)
    crop_mat_tf, crop_mat_scp = kratos_simulation.get_crop_matrix()

    print(S_test.shape)
    print(Phi.shape)
    print(crop_mat_scp.shape)

    S_reduced_test = normalize_data(S_test, crop_mat_scp, Phi)
    S_reconstructed_test = denormalize_data(S_reduced_test, crop_mat_scp, Phi)


    l2_error=mean_relative_l2_error(S_test,S_reconstructed_test)
    forb_error=relative_forbenius_error(S_test,S_reconstructed_test)

    print([l2_error, forb_error])

    

