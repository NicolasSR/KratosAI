from nn_trainer import NN_Trainer
# from residual_convergence_test import ResidualConvergenceTest
from sys import argv
import json

# Import pycompss
""" from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.parameter import *
from pycompss.api.constraint import constraint """

""" @constraint(computing_units="8")"""
""" @task() """
def train(working_path, ae_config):
    training_routine=NN_Trainer(working_path, ae_config)
    training_routine.execute_training()
    del training_routine

def prepare_files(orig_project_parameters_file, dataset_path, working_path):
    """pre-pending the absolut path of the files in the Project Parameters"""
    output_path=orig_project_parameters_file[:-5]+'_workflow.json'

    with open(working_path+dataset_path+orig_project_parameters_file,'r') as f:
        updated_project_parameters = json.load(f)
        file_input_name = updated_project_parameters["solver_settings"]["model_import_settings"]["input_filename"]
        materials_filename = updated_project_parameters["solver_settings"]["material_import_settings"]["materials_filename"]
        updated_project_parameters["solver_settings"]["model_import_settings"]["input_filename"] = working_path + file_input_name
        updated_project_parameters["solver_settings"]["material_import_settings"]["materials_filename"] = working_path + materials_filename

    with open(working_path+dataset_path+output_path,'w') as f:
        json.dump(updated_project_parameters, f, indent = 4)
    
    return output_path

if __name__ == "__main__":

    ae_config_list = [
   #   {
   #      "nn_type": 'conv2d_smain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
   #      "name": 'Test_finetune_small_case_2D_smain_0.1__',
   #      "encoding_size": 1,
   #      # "hidden_layers": ((16,(3,5),(1,2)),
   #      #                   (32,(3,5),(1,2))
   #      #                   ),
   #      "hidden_layers": [[16, [3, 5], [1, 2]], [32, [3, 5], [1, 2]]],
   #      "batch_size": 1,
   #      "epochs": 100,
   #      "normalization_strategy": 'channel_scale',  # ['svd_prenorm', 'svd', 'svd_unif', 'channel_scale']
   #      "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
   #      "orthogonal_loss_ratio": ('const', 0.0),
   #      "learning_rate": ('const', 1e-5), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
   #      "dataset_path": 'datasets_rommanager/',
   #      "models_path": 'saved_models_newexample/',
   #      "finetune_from": 'saved_models_newexample/Test_small_case_2D_smain/',
   #    #   "finetune_from": 'saved_models_conv2d/W0_Colab_RandomDatabase/',
   #      "augmented": True,
   #      "pretrain": False,
   #      "use_bias": False,
   #      "svd_tolerance": None,
   #      "project_parameters_file":'ProjectParameters_fom.json'
   # {
   #      "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
   #      "name": 'test',
   #      "encoding_size": 2,
   #      # "hidden_layers": ((16,(3,5),(1,2)),
   #      #                   (32,(3,5),(1,2))
   #      #                   ),
   #      "hidden_layers": (80,80),
   #      "batch_size": 1,
   #      "epochs": 100,
   #      "normalization_strategy": 'channel_scale',  # ['svd_prenorm', 'svd', 'svd_unif', 'channel_scale']
   #      "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
   #      "orthogonal_loss_ratio": ('const', 0.0),
   #      "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
   #      "dataset_path": 'datasets_two_forces_dense/',
   #      # "dataset_path": 'datasets_two_forces/',
   #      "models_path": 'saved_models_newexample/',
   #      "finetune_from": None,
   #      "augmented": False,
   #      "use_force": False,
   #      "use_bias": False,
   #      "svd_tolerance": None,
   #      "project_parameters_file":'ProjectParameters_fom.json'
   #   },
#    {
#         "nn_type": 'dense_sonly', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'test',
#         "encoding_size": 15,
#         "hidden_layers": [20],
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'svd_white_nostand',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_extended/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": None,
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    },
#    {
#         "nn_type": 'dense_sonly', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'Cont_Dense_lowforce_SOnly_emb10_lay20_LRe6_svd_white_nostand_1000ep',
#         "encoding_size": 10,
#         "hidden_layers": [20],
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'svd_white_nostand',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.000001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_lowforce/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": 'saved_models_newexample/Cont_Dense_lowforce_SOnly_emb10_lay20_LRe5_svd_white_nostand_1000ep/',
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    },
#    {
#         "nn_type": 'dense_sonly', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'Dense_extended_SOnly_emb10_lay40_LRe4_channel_scale_1000ep',
#         "encoding_size": 10,
#         "hidden_layers": [40,40],
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'channel_scale',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_extended/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": None,
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    },
# {
#         "nn_type": 'dec_correct_correctsonly', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'Cont_CorrDecoder_Dense_extended_ChanScale_emb2_LR4_test_1000ep',
#         "encoding_size": 2,
#         "hidden_layers": [10,100],
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'channel_scale',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_extended/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": 'saved_models_newexample/CorrDecoder_Dense_extended_ChanScale_emb2_LR3-4_test_1000ep/',
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    },
   {
        "nn_type": 'dec_correct_correctsonly', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'CorrDecoder_Dense_extended_SVDWhiteNoStand_emb2_LR3_test_1000ep',
        "encoding_size": 2,
        "hidden_layers": [40],
        "batch_size": 1,
        "epochs": 1000,
        "normalization_strategy": 'svd_range',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('steps', 0.01, 10, 1e-6, 20), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense_extended/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "pretrain": False,
        "use_force": False,
        "use_bias": True,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
   },
#    {
#         "nn_type": 'dense_svdmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'Cont_Dense_extended_SVDMain_emb5_lay40_LRe5_white_nostand_3000ep',
#         "encoding_size": 5,
#         "hidden_layers": [40,40],
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'svd_white_nostand',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_extended/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": 'saved_models_newexample/Dense_extended_SVDMain_emb5_lay40_LRe4_white_nostand_2000ep/',
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    },
#    {
#         "nn_type": 'dense_svdmainmae', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'Cont_Dense_extended_SVDMainMAE_emb5_lay40_LRe4_white_nostand_2000ep',
#         "encoding_size": 5,
#         "hidden_layers": [40,40],
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'svd_white_nostand',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_extended/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": 'saved_models_newexample/Dense_extended_SVDMainMAE_emb5_lay40_LRe3_white_nostand_1000ep/',
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    },
#    {
#         "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'Finetune_Dense_extended_RMain_w0.01_lay40_LRe6_origLRe6ep6000_svd_white_nostand_1000ep',
#         "encoding_size": 2,
#         "hidden_layers": (40,40),
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'svd_white_nostand',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 0.01), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.000001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_extended/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": 'saved_models_newexample/Cont_Dense_extended_SOnly_w0_lay40_LRe6_svd_white_nostand_6000ep/',
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    },
#    {
#         "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'Finetune_Dense_extended_RMain_w0_lay40_LRe5_svd_white_nostand_1000ep',
#         "encoding_size": 2,
#         "hidden_layers": (40,40),
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'svd_white_nostand',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_extended/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": 'saved_models_newexample/Cont_Dense_extended_SOnly_w0_lay40_LRe5_svd_white_nostand_2000ep/',
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    },
#    {
#         "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'Finetune_Dense_extended_RMain_w1_lay40_LRe5_svd_white_nostand_1000ep',
#         "encoding_size": 2,
#         "hidden_layers": (40,40),
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'svd_white_nostand',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 1.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_extended/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": 'saved_models_newexample/Cont_Dense_extended_SOnly_w0_lay40_LRe5_svd_white_nostand_2000ep/',
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    },
#    {
#         "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
#         "name": 'Finetune_Dense_extended_RMain_w10_lay40_LRe5_svd_white_nostand_1000ep',
#         "encoding_size": 2,
#         "hidden_layers": (40,40),
#         "batch_size": 1,
#         "epochs": 1000,
#         "normalization_strategy": 'svd_white_nostand',  # ['svd_white', 'svd_white_nostand', 'svd_prenorm', svd_prenorm_chan, 'svd', 'svd_unif', 'channel_scale']
#         "residual_loss_ratio": ('const', 10), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
#         "orthogonal_loss_ratio": ('const', 0.0),
#         "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
#         "dataset_path": 'datasets_two_forces_dense_extended/',
#         # "dataset_path": 'datasets_rommanager/',
#         "models_path": 'saved_models_newexample/',
#         "finetune_from": 'saved_models_newexample/Cont_Dense_extended_SOnly_w0_lay40_LRe5_svd_white_nostand_2000ep/',
#         "augmented": False,
#         "pretrain": False,
#         "use_force": False,
#         "use_bias": False,
#         "svd_tolerance": None,
#         "project_parameters_file":'ProjectParameters_fom.json'
#    }
   ]
    
    """ Notes on SVD training
    Using embedding: 2
    For now the best has been to use LR e-4 for at least 200 epochs (maybe try something higher to see if we get the same faster)
    Then, layers (80,80) worked better than (40,40). Try higher, or same number of neurons with one more layer, etc

    If this still doesn't work, we should try a different normalisation of the data. Maybe scaling all features by the same factor
    effectively gives the first feature more importance. Otherwise we can try to use a loss function that weights the outputs

    Best until now: ScaledSVD_ThickCantilever_try2
    
    """

    # residual_test = ResidualConvergenceTest(ae_config_list[0])
    # residual_test.execute_test()

    working_path=argv[1]+"/"
    
    for i, ae_config in enumerate(ae_config_list):
        
        print('----------  Training case ', i+1, ' of ', len(ae_config_list), '  ----------')
        output_path=prepare_files(ae_config["project_parameters_file"], ae_config["dataset_path"], working_path)
        ae_config["project_parameters_file"]=output_path
        train(working_path, ae_config)
    
    # compss_barrier()
    print('FINISHED TRAINING')

        
