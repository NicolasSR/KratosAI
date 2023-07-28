import argparse

from nn_serializer import NN_Serializer

def evaluate(working_path, model_path, best):
    serialization_routine=NN_Serializer(working_path, model_path, best)
    serialization_routine.execute_serialization()

if __name__ == "__main__":
    paths_list=[
            'LinearNet_LR5'
            ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('working_path', type=str, help='Root directory to work from.')
    parser.add_argument('--best', type=str, help='Evaluate the best epoch instead of last. can be set to x or r.')
    args = parser.parse_args()
    working_path = args.working_path+'/'
    best=args.best
    
    for i, model_path in enumerate(paths_list):
        
        print('----------  Evaluating case ', i+1, ' of ', len(paths_list), '  ----------')
        evaluate(working_path, 'saved_models_newexample/'+model_path+'/', best)
    
    # compss_barrier()
    print('FINISHED SERIALIZING')
