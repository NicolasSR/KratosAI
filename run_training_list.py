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
   #      "name": 'NewCode_NoForce_AugmFinetune_RandomDatabase_finetuneColab_w0.1_lam0.0_lr0.00001',
   #      "encoding_size": 1,
   #      # "hidden_layers": ((16,(3,5),(1,2)),
   #      #                   (32,(3,5),(1,2))
   #      #                   ),
   #      "hidden_layers": [[16, [3, 5], [1, 2]], [32, [3, 5], [1, 2]]],
   #      "batch_size": 1,
   #      "epochs": 100,
   #      "normalization_strategy": 'channel_range',  # ['feature_stand','channel_range', 'channel_scale', 'svd]
   #      "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
   #      "orthogonal_loss_ratio": ('const', 0.0),
   #      "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
   #      "dataset_path": 'datasets_rommanager/',
   #      "models_path": 'saved_models_newexample/',
   #      "finetune_from": 'saved_models_conv2d/W0_Colab_RandomDatabase/',
   #      "augmented": True,
   #      "use_force": False,
   #      "use_bias": True,
   #      "svd_tolerance": None,
   #      "project_parameters_file":'ProjectParameters_fom.json'
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay80_LRe5_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay80_LRe4_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay80_LRe3_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay80_LRe5_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay80_LRe4_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay80_LRe3_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay40_LRe5_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay40_LRe4_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay40_LRe3_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay40_LRe5_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0),
        "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay40_LRe4_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay40_LRe3_svdunif',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd_unif',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay80_LRe5_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay80_LRe4_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay80_LRe3_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay80_LRe5_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay80_LRe4_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay80_LRe3_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (80,80),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay40_LRe5_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay40_LRe4_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0_lay40_LRe3_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.0), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay40_LRe5_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.00001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay40_LRe4_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.0001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     },
     {
        "nn_type": 'dense_rmain', # ['dense_umain','conv2d_umain','dense_rmain','conv2d_rmain']
        "name": 'Dense_RMain_w0.1_lay40_LRe3_svd',
        "encoding_size": 2,
        # "hidden_layers": ((16,(3,5),(1,2)),
        #                   (32,(3,5),(1,2))
        #                   ),
        "hidden_layers": (40,40),
        "batch_size": 1,
        "epochs": 100,
        "normalization_strategy": 'svd',  # ['feature_stand','channel_range', 'channel_scale', 'svd', 'svd_unif']
        "residual_loss_ratio": ('const', 0.1), # ('linear', 0.99999, 0.1, 100), ('const', 1.0), ('binary', 0.99999, 0.0, 2)
        "orthogonal_loss_ratio": ('const', 0.0),
        "learning_rate": ('const', 0.001), # ('steps', 0.001, 10, 1e-6, 100), ('const', 0.001)
        "dataset_path": 'datasets_two_forces_dense/',
        # "dataset_path": 'datasets_rommanager/',
        "models_path": 'saved_models_newexample/',
        "finetune_from": None,
        "augmented": False,
        "use_force": False,
        "use_bias": False,
        "svd_tolerance": None,
        "project_parameters_file":'ProjectParameters_fom.json'
     }    ]
    
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

        
