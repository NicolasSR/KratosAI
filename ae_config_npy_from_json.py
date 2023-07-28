import os
import sys

# logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import math

import contextlib

import h5py
import numpy as np

import matplotlib.pyplot as plt


if __name__ == "__main__":


    with open("ae_config.json", "r") as config_file:
        config=json.load(config_file)

    np.save('ae_config.npy', config)