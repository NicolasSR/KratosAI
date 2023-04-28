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


    with open("saved_models_conv2d/history.json", "r") as history_file:
        history=json.load(history_file)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('x_loss', color=color)
    plt.plot(np.arange(len(history['loss_x'])), history['loss_x'], label='x_train', color='red')
    try:
        plt.plot(np.arange(len(history['val_loss_x'])), history['val_loss_x'], label='x_val', color='orange')
    except:
        print('No val_loss_x data')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    try:
        color = 'tab:blue'
        ax2.set_ylabel('r_loss', color=color)  # we already handled the x-label with ax1
        plt.plot(np.arange(len(history['err_r'])), history['err_r'], label='r_train', color='blue')
        try:
            plt.plot(np.arange(len(history['val_err_r'])), history['val_err_r'], label='r_val', color='cyan')
        except:
            print('No val_err_r data')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color)
    except:
        print("No r loss to plot")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    try:
        plt.plot(np.arange(len(history['lr'])), history['lr'])
        plt.yscale('log')
        plt.show()

        plt.plot(np.arange(len(history['w'])), history['w'])
        plt.show()

        plt.plot(np.arange(len(history['r_norm_factor'])), history['r_norm_factor'])
        plt.yscale('log')
        plt.show()

    except:
        print("No lr, w or r_norm_factor to plot")