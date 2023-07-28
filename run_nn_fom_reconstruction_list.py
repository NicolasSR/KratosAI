import json
import argparse
import numpy as np

from nn_fom_reconstructor import NN_FOM_Reconstructor, FOM_Simulator

def simulate(working_path, model_path, best, print_matrices, original):
    if original:
        simulation_routine=FOM_Simulator(working_path, model_path)
    else:
        simulation_routine=NN_FOM_Reconstructor(working_path, model_path, best, print_matrices)
    simulation_routine.execute_simulation()

def prepare_files(orig_project_parameters_file, dataset_path, working_path, output_filename):
    """pre-pending the absolut path of the files in the Project Parameters"""
    """also changing the output path and names for the gid output files"""

    with open(working_path+dataset_path+orig_project_parameters_file,'r') as f:
        updated_project_parameters = json.load(f)
        updated_project_parameters["output_processes"]["gid_output"][0]["Parameters"]["output_name"] = output_filename

    with open(working_path+dataset_path+orig_project_parameters_file,'w') as f:
        json.dump(updated_project_parameters, f, indent = 4)

if __name__ == "__main__":
    paths_list=[
            'Dense_lowforce_SOnly_emb10_lay20_LRe4_svd_white_nostand_1000ep'
            ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    parser.add_argument('--best', type=str, help='Evaluate the best epoch instead of last. can be set to x or r.')
    parser.add_argument('--printmat', action='store_true')
    parser.add_argument('-o', '--original', action='store_true')
    args = parser.parse_args()
    working_path = args.working_path+'/'
    best=args.best
    print_matrices = args.printmat
    original = args.original
    print('Original: ', original)
    
    for i, model_path in enumerate(paths_list):
        
        model_path='saved_models_newexample/'+model_path+'/'

        print('----------  Processing case ', i+1, ' of ', len(paths_list), '  ----------')
        with open(model_path+"ae_config.npy", "rb") as ae_config_file:
            ae_config = np.load(ae_config_file,allow_pickle='TRUE').item()

        if original:
            output_filename = working_path+model_path+'fom_coarse'
        else:
            output_filename = working_path+model_path+'nn_fom_reconstruction_coarse'
            if best=='x':
                output_filename+='_bestx'
            elif best=='r':
                output_filename+='_bestr'
    
        prepare_files('ProjectParameters_nn_fom_reconstruction.json', ae_config["dataset_path"], working_path, output_filename)
        simulate(working_path, model_path, best, print_matrices, original)
    
    # compss_barrier()
    print('FINISHED PROCESSING')
