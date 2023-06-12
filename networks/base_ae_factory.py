import abc
import datetime

import tensorflow as tf
from tensorflow.keras.initializers import HeNormal

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as ROM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

from utils.custom_scheduler import CustomLearningRateScheduler

class Base_AE_Factory(abc.ABC):

    def __init__(self):
        super().__init__()

    def my_metrics_function(_,y_true,y_pred):
        return (y_true - y_pred) ** 2
    
    @abc.abstractmethod
    def keras_model_selector(self,ae_config):
        'Defined in the subclasses'
        
    @abc.abstractmethod
    def define_network(self, input_data, ae_config):
        'Defined in the subclasses'

    def predict_snapshot(self, network, snapshot):
        a = network.predict(snapshot)
        return a
    
    def encode_snapshot(self, encoder, autoencoder, snapshot):
        norm_snapshot=autoencoder.data_normalizer.process_raw_to_input_format(snapshot)
        out=encoder.predict(norm_snapshot)
        return out

    def train_network(self, model, input_data, grad_data, val_input, val_truth, ae_config):
        # Train the model
        def lr_steps_schedule(epoch, lr):
            if epoch==0:
                new_lr=ae_config["learning_rate"][1]
            elif epoch%ae_config["learning_rate"][4]==0:
                new_lr=max(lr/ae_config["learning_rate"][2], ae_config["learning_rate"][3])
            else:
                new_lr=lr
            return new_lr

        def lr_const_schedule(epoch, lr):
            new_lr=ae_config["learning_rate"][1]
            return new_lr
        
        def w_lin_schedule(epoch, w):
            slope_min=ae_config["residual_loss_ratio"][1]
            slope_width=ae_config["residual_loss_ratio"][3]
            slope_max=ae_config["residual_loss_ratio"][2]
            new_w=min((slope_max-slope_min)/slope_width*epoch+slope_min,slope_max)
            return new_w
        
        def w_bin_schedule(epoch, w):
            if epoch<ae_config["residual_loss_ratio"][3]:
                new_w=ae_config["residual_loss_ratio"][1]
            else:
                new_w=new_w=ae_config["residual_loss_ratio"][2]
            return new_w
        
        def w_const_schedule(epoch, w):
            new_w=ae_config["residual_loss_ratio"][1]
            return new_w
        
        def lam_const_schedule(epoch, lam):
            new_lam=ae_config["orthogonal_loss_ratio"][1]
            return new_lam

        if ae_config["learning_rate"][0]=='const':
            lr_schedule = lr_const_schedule
        elif ae_config["learning_rate"][0]=='steps':
            lr_schedule = lr_steps_schedule
        else: print('Unvalid lr scheduler')

        if ae_config["residual_loss_ratio"][0]=='const':
            w_schedule = w_const_schedule
        elif ae_config["residual_loss_ratio"][0]=='linear':
            w_schedule = w_lin_schedule
        elif ae_config["residual_loss_ratio"][0]=='binary':
            w_schedule = w_bin_schedule
        else: print('Unvalid w scheduler')

        if ae_config["orthogonal_loss_ratio"][0]=='const':
            lam_schedule = lam_const_schedule
        else: print('Unvalid lambda scheduler')

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss_r', patience=5)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_best_callback = tf.keras.callbacks.ModelCheckpoint(ae_config["models_path"]+ae_config["name"]+"/best/weights_{epoch:03d}.h5",save_weights_only=True,save_best_only=True,monitor="val_loss_r",mode="min")
        checkpoint_last_callback = tf.keras.callbacks.ModelCheckpoint(ae_config["models_path"]+ae_config["name"]+"/last/weights.h5",save_weights_only=True,save_freq="epoch")
        lr_w_lam_scheduler_callback = CustomLearningRateScheduler(lr_schedule, w_schedule, lam_schedule, verbose=0)

        history = model.fit(
            input_data, grad_data,
            epochs=ae_config["epochs"],
            shuffle=True,
            batch_size=ae_config["batch_size"],
            validation_data=(val_input,val_truth),
            callbacks = [
                lr_w_lam_scheduler_callback,
                # early_stop_callback,
                # tensorboard_callback,
                checkpoint_best_callback,
                checkpoint_last_callback,
            ]
        )

        return history

    def test_network(self, model, input_data, grad_data):
        result = model.evaluate(input_data, grad_data, batch_size=1)
        return result

    def calculate_gradients(self, decoder_model, autoencoder_model, input_data):
        in_tf_var = tf.Variable(input_data)
        with tf.GradientTape(persistent=True) as tape_d:
            tape_d.watch(in_tf_var)
            x_pred = decoder_model(in_tf_var)
            x_pred_denorm = autoencoder_model.data_normalizer.process_input_to_raw_format_tf(x_pred)

        jac = tape_d.jacobian(x_pred_denorm, in_tf_var, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)
        return jac[0,:,0,:].numpy()

    def compute_full_gradient():
        return None