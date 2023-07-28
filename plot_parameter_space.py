import os
import logging

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


class Plotter():

    def __init__(self, working_path, dataset_path, F_test_file_name, F_train_filename):
        self.dataset_path=working_path+dataset_path

        F_test = np.load(self.dataset_path+F_test_file_name)
        self.F_test_parsed, _ = self.get_F_parsed(F_test)
        print('Shape F: ', F_test.shape)
        print('Shape F_parsed: ', self.F_test_parsed.shape)

        F_train = np.load(self.dataset_path+F_train_filename)
        self.F_train_parsed, _ = self.get_F_parsed(F_train)
        print('Shape F: ', F_train.shape)
        print('Shape F_parsed: ', self.F_train_parsed.shape)


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
    
    def plot(self):
        plt.scatter(self.F_test_parsed[:,0], self.F_test_parsed[:,1], s=1)
        plt.scatter(self.F_train_parsed[:,0], self.F_train_parsed[:,1], s=1)
        plt.show()

    
if __name__ == "__main__":

    param_space_plotter = Plotter('', 'datasets_two_forces_dense_lowforce/', 'F_finetune_test.npy', 'F_finetune_train.npy')
    param_space_plotter.plot()