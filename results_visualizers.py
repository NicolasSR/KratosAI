import streamlit as st
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pandas as pd

def draw_x_error_image(s_err_image):
    fig, (ax1) = plt.subplots(ncols=1)
    im1 = ax1.imshow(np.abs(s_err_image), interpolation='none')
    # ax1.set_aspect(1/2e8)
    cbar1 = plt.colorbar(im1)
    plt.xlabel('index')
    plt.ylabel('force')
    plt.title('Displacement Abs Error')
    st.pyplot(fig)

def draw_r_error_image(r_err_image):
    fig, (ax1) = plt.subplots(ncols=1)
    im1 = ax1.imshow(np.abs(r_err_image), interpolation='none')
    cbar1 = plt.colorbar(im1)
    plt.xlabel('index')
    plt.ylabel('force')
    plt.title('Residual Abs Error')
    st.pyplot(fig)

def load_results_files(model_path, weights_type, test_base_type):
    with open(model_path+'history.json', "r") as history_file:
        history=json.load(history_file)
    ae_config=np.load(model_path+'ae_config.npy', allow_pickle=True).item()
    s_err_image=np.load(model_path+'s_err_image'+best_name_part[weights_type]+test_base_name_part[test_base_type]+'.npy')
    x_norm_err=np.load(model_path+'x_norm_errs'+best_name_part[weights_type]+test_base_name_part[test_base_type]+'.npy')
    r_err_image=np.load(model_path+'r_err_image'+best_name_part[weights_type]+test_base_name_part[test_base_type]+'.npy')
    r_norm_err=np.load(model_path+'r_norm_errs'+best_name_part[weights_type]+test_base_name_part[test_base_type]+'.npy')
    return history, ae_config, s_err_image, x_norm_err, r_err_image, r_norm_err

def get_best_epoch_num(model_path, prefix):
    matching_files_epochs = [int(file[len(prefix):][:-len('.h5')])-1 for file in os.listdir(model_path+'best/') if file.startswith(prefix)]
    highest_epoch = np.max(matching_files_epochs)
    return highest_epoch

# Create a Streamlit app
st.title("TensorFlow History Plot")
st.header("Training History")

folder=st.sidebar.selectbox("Select a folder", sorted(os.listdir(os.getcwd()+'/saved_models_newexample/')),key=1)
model_path_1='saved_models_newexample/'+folder+'/'
folder_2=st.sidebar.selectbox("Select a folder", sorted(os.listdir(os.getcwd()+'/saved_models_newexample/')),key=2)
model_path_2='saved_models_newexample/'+folder_2+'/'

weights_type_1=st.sidebar.selectbox("Select type of weights", ['Last','Best x','Best r'],key=3)
weights_type_2=st.sidebar.selectbox("Select type of weights", ['Last','Best x','Best r'],key=4)
best_name_part={'Last':'','Best x':'_bestx_','Best r':'_bestr_'}

test_base_type_1=st.sidebar.selectbox("Select type of weights", ['Validation','Large test'],key=5)
test_base_type_2=st.sidebar.selectbox("Select type of weights", ['Validation','Large test'],key=6)
test_base_name_part={'Validation':'','Large test':'_test_large_'}

history_1, ae_config_1, s_err_image_1, x_norm_err_1, r_err_image_1, r_norm_err_1 = load_results_files(model_path_1, weights_type_1, test_base_type_1)
history_2, ae_config_2, s_err_image_2, x_norm_err_2, r_err_image_2, r_norm_err_2 = load_results_files(model_path_2, weights_type_2, test_base_type_2)

error_table_headers_x = ['X Mean Rel L2 Norm', 'X Rel Frobenius Norm']
error_table_headers_r = ['R Mean Rel L2 Norm', 'R Rel Frobenius Norm']
error_table_col_names = ['Test', 'Train']

col1, col2 = st.columns([2,2])
both_history_r = False

y_min_x = np.min([np.min(history_1['loss_x']), np.min(history_2['loss_x']), np.min(history_1['val_loss_x']), np.min(history_2['val_loss_x'])])
y_max_x = np.max([np.max(history_1['loss_x']), np.max(history_2['loss_x']), np.max(history_1['val_loss_x']), np.max(history_2['val_loss_x'])])
try:
    y_min_r = np.min([np.min(history_1['loss_r']), np.min(history_2['loss_r']), np.min(history_1['val_loss_r']), np.min(history_2['val_loss_r'])])
    y_max_r = np.max([np.max(history_1['loss_r']), np.max(history_2['loss_r']), np.max(history_1['val_loss_r']), np.max(history_2['val_loss_r'])])
    both_history_r = True
except:
    pass

best_epoch_1=None
best_epoch_2=None
if weights_type_1=='Best x':
    best_epoch_1=get_best_epoch_num(model_path_1,'weights_x_')
elif weights_type_1=='Best r':
    best_epoch_1=get_best_epoch_num(model_path_1,'weights_r_')
if weights_type_2=='Best x':
    best_epoch_2=get_best_epoch_num(model_path_2,'weights_x_')
elif weights_type_2=='Best r':
    best_epoch_2=get_best_epoch_num(model_path_2,'weights_r_')

with col1:

    # Plot loss curve
    fig, ax = plt.subplots(1,1)
    ax.plot(history_1['loss_x'], label='x', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss_x')
    ax.plot(history_1['val_loss_x'], label='x_val', color='orange')
    plt.semilogy()
    plt.legend()
    plt.ylim(y_min_x, y_max_x)
    try:
        plt.axvline(x=best_epoch_1, color='black', linestyle='--')
    except:
        pass
    st.pyplot(fig)

    # Plot loss curve
    try:
        fig, ax = plt.subplots(1,1)
        ax.plot(history_1['loss_r'], label='r', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss_r')
        ax.plot(history_1['val_loss_r'], label='r_val', color='cyan')
        plt.semilogy()
        plt.legend()
        if both_history_r:
            plt.ylim(y_min_r, y_max_r)
        try:
            plt.axvline(x=best_epoch_1, color='black', linestyle='--')
        except:
            pass
        st.pyplot(fig)
    except:
        pass

    df_x_norm_err_1=pd.DataFrame(x_norm_err_1, columns=error_table_headers_x)
    df_r_norm_err_1=pd.DataFrame(r_norm_err_1, columns=error_table_headers_r)
    df_norm_err_1=pd.concat([df_x_norm_err_1, df_r_norm_err_1], axis=1)
    df_norm_err_1.insert(0, 'Dataset', error_table_col_names)

    st.write('Norm errors:')
    st.dataframe(df_norm_err_1)

    draw_x_error_image(s_err_image_1)

    draw_r_error_image(r_err_image_1)

with col2:

    # Plot loss curve
    fig, ax = plt.subplots(1,1)
    ax.plot(history_2['loss_x'], label='x', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss_x')
    ax.plot(history_2['val_loss_x'], label='x_val', color='orange')
    plt.semilogy()
    plt.legend()
    plt.ylim(y_min_x, y_max_x)
    try:
        plt.axvline(x=best_epoch_2, color='black', linestyle='--')
    except:
        pass
    st.pyplot(fig)

    # Plot loss curve
    try:
        fig, ax = plt.subplots(1,1)
        ax.plot(history_2['loss_r'], label='r', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss_r')
        ax.plot(history_2['val_loss_r'], label='r_val', color='cyan')
        plt.semilogy()
        plt.legend()
        if both_history_r:
            plt.ylim(y_min_r, y_max_r)
        try:
            plt.axvline(x=best_epoch_2, color='black', linestyle='--')
        except:
            pass
        st.pyplot(fig)
    except:
        pass

    df_x_norm_err_2=pd.DataFrame(x_norm_err_2, columns=error_table_headers_x)
    df_r_norm_err_2=pd.DataFrame(r_norm_err_2, columns=error_table_headers_r)
    df_norm_err_2=pd.concat([df_x_norm_err_2, df_r_norm_err_2], axis=1)
    df_norm_err_2.insert(0, 'Dataset', error_table_col_names)

    st.write('Norm errors:')
    st.dataframe(df_norm_err_2)

    draw_x_error_image(s_err_image_2)

    draw_r_error_image(r_err_image_2)
