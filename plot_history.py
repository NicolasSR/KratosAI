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


    with open("saved_models_new/history.json", "r") as history_file:
        history=json.load(history_file)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('x_loss', color=color)
    plt.plot(np.arange(len(history['lr'])), history['loss_x'], label='x_train', color='red')
    plt.plot(np.arange(len(history['lr'])), history['val_loss_x'], label='x_val', color='orange')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    try:
        color = 'tab:blue'
        ax2.set_ylabel('r_loss', color=color)  # we already handled the x-label with ax1
        plt.plot(np.arange(len(history['lr'])), history['err_r'], label='r_train', color='blue')
        plt.plot(np.arange(len(history['lr'])), history['val_err_r'], label='r_val', color='cyan')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color)
    except:
        print("No r loss to plot")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    plt.plot(np.arange(len(history['lr'])), history['lr'])
    plt.yscale('log')
    plt.show()

    plt.plot(np.arange(len(history['lr'])), history['w'])
    plt.show()

    plt.plot(np.arange(len(history['r_norm_factor'])), history['r_norm_factor'])
    plt.yscale('log')
    plt.show()