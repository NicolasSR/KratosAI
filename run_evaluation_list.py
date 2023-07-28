from sys import argv
import getopt
import argparse

from nn_evaluator import NN_Evaluator
from nn_features_evaluator import NN_Features_Evaluator

def evaluate(working_path, model_path, best, test_large=False):
    training_routine=NN_Evaluator(working_path, model_path, best, test_large=test_large)
    training_routine.execute_evaluation()

if __name__ == "__main__":
    paths_list=[
            'Finetune_Dense_extended_RMain_w0.1_lay40_LRe6_origLRe6ep6000_svd_white_nostand_1000ep',
            ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    parser.add_argument('--best', type=str, help='Evaluate the best epoch instead of last. can be set to x or r.')
    parser.add_argument('--test_large', action='store_true')
    args = parser.parse_args()
    working_path = args.working_path+'/'
    best=args.best
    test_large = args.test_large
    
    for i, model_path in enumerate(paths_list):
        
        print('----------  Evaluating case ', i+1, ' of ', len(paths_list), '  ----------')
        evaluate(working_path, 'saved_models_newexample/'+model_path+'/', best, test_large=test_large)
    
    # compss_barrier()
    print('FINISHED EVALUATING')