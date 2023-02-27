import os
import sys
import math

import numpy as np
import scipy

import keras
import tensorflow as tf

import utils
import networks.network as network

from keras import layers
from keras import regularizers
import keras.backend as K

from tensorflow.keras.initializers import HeNormal

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as ROM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

# Create a custom Model:
loss_x_tracker = keras.metrics.Mean(name="loss_x")
loss_r_tracker = keras.metrics.Mean(name="err_r")

class GradModel2(keras.Model):

    def __init__(self,*args,**kwargs):
        super(GradModel2,self).__init__(*args,**kwargs)
        self.normalization_mode = None
        self.feat_means = None
        self.feat_stds = None
        self.data_min = None
        self.data_max = None
        self.w = None

    def gen_random_matrices(self, length):
        rng = np.random.default_rng(2021)
        self.a_vec = rng.random((1,length))*40-20
        self.b_mat = rng.random((length,length))*40-20
        print('a and B')
        print(self.a_vec)
        print(self.a_vec.shape)
        print(self.b_mat)
        print(self.b_mat.shape)

    def get_r(self, y_pred):
        
        id_mat=np.identity(y_pred.shape[1])
        y_pred=y_pred.numpy()
        
        b_1=y_pred.T@self.a_vec
        b_2=b_1@y_pred.T
        b_3=self.b_mat@y_pred.T
        b = b_2+b_3
        b=b.T

        A_1=y_pred.T@self.a_vec
        A_2=y_pred@self.a_vec.T
        A_3=A_2*id_mat
        A = A_1+A_3+self.b_mat
        
        return A, b

    # Mean square error of the data
    def diff_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def denormalize_data(self, input_data):
        if self.normalization_mode == "per_feature":
            if tf.is_tensor(input_data):
                output_data=input_data*np.array([self.feat_stds])
                output_data=output_data+np.array([self.feat_means])
            else:
                for i in range(input_data.shape[1]):
                    output_data=input_data
                    output_data[:,i]= (output_data[:,i]*self.feat_stds[i])+self.feat_means[i]
        elif self.normalization_mode == "global":
            output_data = input_data * (self.data_max - self.data_min) + self.data_min
        else:
            output_data = input_data
        return output_data
        
    def normalize_data(self, input_data):
        if self.normalization_mode == "per_feature":
            if tf.is_tensor(input_data):
                output_data=input_data-np.array([self.feat_means])
                output_data=output_data/(np.array([self.feat_stds])+0.00000001)
            else:
                for i in range(input_data.shape[1]):
                    output_data=input_data
                    output_data[:,i]=(output_data[:,i]-self.feat_means[i])/(self.feat_stds[i]+0.00000001)
        elif self.normalization_mode == "global":
            output_data = (input_data - self.data_min) / (self.data_max - self.data_min)
        else:
            output_data = input_data
        return output_data

    def set_normalization_data(self, normalization_mode, norm_data):
        self.normalization_mode = normalization_mode
        if self.normalization_mode == "per_feature":
            self.feat_means, self.feat_stds = norm_data
        elif self.normalization_mode == "global":
            self.data_min, self.data_max = norm_data

    def train_step(self,data):
        w = self.w
        x_true, r_true = data
        trainable_vars = self.trainable_variables

        if w == 1:

            with tf.GradientTape(persistent=True) as tape_d:
                tape_d.watch(trainable_vars)
                x_pred = self(x_true, training=True)
                loss_x = self.diff_loss(x_true, x_pred)

            # Compute gradients
            gradients_loss_x = tape_d.gradient(loss_x, trainable_vars)
        
            total_gradients = []
            for i in range(len(gradients_loss_x)):
                total_gradients.append(gradients_loss_x[i])

            # Backpropagation
            self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

            loss_x_tracker.update_state(loss_x)

            return {"loss_x": loss_x_tracker.result()}
        
        else:
            x_true_denorm = self.denormalize_data(x_true)
            A_true, b_true = self.get_r(x_true_denorm)

            with tf.GradientTape(persistent=True) as tape_d:
                tape_d.watch(trainable_vars)
                x_pred = self(x_true, training=True)
                loss_x = self.diff_loss(x_true, x_pred)
                x_pred_denorm = self.denormalize_data(x_pred)

            grad_loss_x = tape_d.gradient(loss_x,trainable_vars)
            jac_u = tape_d.jacobian(x_pred_denorm, trainable_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)

            A_pred, b_pred = self.get_r(x_pred_denorm)
            A_pred  = tf.constant(A_pred)

            # alpha=1
            # jac_r = -(0-b_pred) @ (alpha * A_pred.T)
            # jac_r  = tf.constant(jac_r)
            # print(jac_r)
            # A_pred = tf.expand_dims(A_pred, axis=0)


            ## Check how similar b_true and r_true are. Thheyshould be the same
            # print(A_true)
            # print(A_pred)
            # print(r_true)
            # print(b_true)
            # print(b_pred)

            r_pred = b_pred
            err_r = b_true-r_pred
            err_r = tf.expand_dims(tf.constant(err_r),axis=0)
            loss_r = self.diff_loss(b_true, r_pred)

            total_gradients = []

            i=0
            for layer in jac_u:

                l_shape=tf.shape(layer)
                if len(l_shape) == 4:
                    layer=tf.reshape(layer,(l_shape[0],l_shape[1],l_shape[2]*l_shape[3]))

                pre_grad=tf.linalg.matmul(A_pred,tf.squeeze(layer,axis=0),a_is_sparse=True)
                grad_loss_r=tf.matmul(err_r,pre_grad)*(-2)

                if len(l_shape) == 4:
                    grad_loss_r=tf.reshape(grad_loss_r,(l_shape[2],l_shape[3]))
                else:
                    grad_loss_r=tf.reshape(grad_loss_r,(l_shape[2]))

                total_gradients.append(w*grad_loss_x[i]+(1-w)*grad_loss_r)
                
                i+=1

            self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

            # Compute our own metrics
            loss_x_tracker.update_state(loss_x)
            # print(loss_r)
            # print(np.mean(loss_r))
            loss_r_tracker.update_state(loss_r)

            return {"loss_x": loss_x_tracker.result(), "err_r": loss_r_tracker.result()}


    def test_step(self, data):
        w = self.w
        x_true, r_true = data

        if w == 1:

            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)

            loss_x_tracker.update_state(loss_x)

            return {"loss_x": loss_x_tracker.result()}

        else:
            
            x_true_denorm = self.denormalize_data(x_true)
            A_true, b_true = self.get_r(x_true_denorm)

            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)
            x_pred_denorm = self.denormalize_data(x_pred)

            A_pred, b_pred = self.get_r(x_pred_denorm)
            A_pred  = tf.constant(A_pred)

            r_pred = b_pred
            loss_r = self.diff_loss(b_true, r_pred)

            # Compute our own metrics
            loss_x_tracker.update_state(loss_x)
            loss_r_tracker.update_state(loss_r)

            return {"loss_x": loss_x_tracker.result(), "err_r": loss_r_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        if self.w==1:
            return [loss_x_tracker]
        else:
            return [loss_x_tracker, loss_r_tracker]


class GradientShallow(network.Network):

    def __init__(self):
        '''
        Initis the network class

        model_name:     name of the directory in which the model will be saved.
        valid:          percentage of the input which is valid. (First steps are typically)
        '''
        super().__init__()

        self.model_name = "./saved_models/gradient_shallow"
        self.valid = 0.8

    def my_metrics_function(_,y_true,y_pred):
        return (y_true - y_pred) ** 2

    def define_network(self, input_data, custom_loss, ae_config):
        data = np.transpose(input_data)
		
        leaky_alpha = 0.3

        encoding_factor = ae_config["encoding_factor"]
        hidden_layers = ae_config["hidden_layers"]
        use_batch_normalisation = ae_config["use_batch_normalisation"]
        dropout_rate = ae_config["dropout_rate"]

        use_dropout = dropout_rate != 0.0
        
        decoded_size = data.shape[1]
        encoded_size=int(decoded_size*encoding_factor)

        model_input = tf.keras.Input(shape=(decoded_size,))
        decod_input = tf.keras.Input(shape=(encoded_size,))

        hid_sizes_encoder = np.linspace(encoded_size, decoded_size, hidden_layers[0]+1, endpoint=False, dtype=int)
        hid_sizes_encoder = np.flip(hid_sizes_encoder[1:])
        hid_sizes_decoder = np.linspace(encoded_size, decoded_size, hidden_layers[1]+1, endpoint=False, dtype=int)[1:]

        encoder_out = model_input
        if use_batch_normalisation:
            encoder_out = tf.keras.layers.BatchNormalization()(encoder_out)
        if use_dropout:
            encoder_out = tf.keras.layers.Dropout(dropout_rate)(encoder_out)
        for layer_size in hid_sizes_encoder:
            encoder_out = tf.keras.layers.Dense(layer_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(encoder_out)
            encoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(encoder_out)
            if use_batch_normalisation:
                encoder_out = tf.keras.layers.BatchNormalization()(encoder_out)
            if use_dropout:
                encoder_out = tf.keras.layers.Dropout(dropout_rate)(encoder_out)
        encoder_out = tf.keras.layers.Dense(encoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(encoder_out)
        encoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(encoder_out)

        decoder_out = decod_input
        if use_batch_normalisation:
            decoder_out = tf.keras.layers.BatchNormalization()(decoder_out)
        if use_dropout:
            decoder_out = tf.keras.layers.Dropout(dropout_rate)(decoder_out)
        for layer_size in hid_sizes_decoder:
            decoder_out = tf.keras.layers.Dense(layer_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decoder_out)
            decoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(decoder_out)
            if use_batch_normalisation:
                decoder_out = tf.keras.layers.BatchNormalization()(decoder_out)
            if use_dropout:
                decoder_out = tf.keras.layers.Dropout(dropout_rate)(decoder_out)
        decoder_out = tf.keras.layers.Dense(decoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decoder_out)
        
        """ encoder_out = tf.keras.layers.Dense(hid_size_2, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(model_input)
        encoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(encoder_out)
        # encoder_out = tf.keras.layers.BatchNormalization()(encoder_out)
        # encoder_out = tf.keras.layers.Dropout(dropout_rate)(encoder_out)
        encoder_out = tf.keras.layers.Dense(hid_size_3, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(encoder_out)
        encoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(encoder_out)
        # encoder_out = tf.keras.layers.BatchNormalization()(encoder_out)
        # encoder_out = tf.keras.layers.Dropout(dropout_rate)(encoder_out)
        encoder_out = tf.keras.layers.Dense(encoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(encoder_out)
        encoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(encoder_out) """

        """ decoder_out = tf.keras.layers.Dense(hid_size_3, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decod_input)
        decoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(decoder_out)
        # decoder_out = tf.keras.layers.BatchNormalization()(decoder_out)
        # decoder_out = tf.keras.layers.Dropout(dropout_rate)(decoder_out)
        decoder_out = tf.keras.layers.Dense(hid_size_2, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decod_input)
        decoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(decoder_out)
        decoder_out = tf.keras.layers.Dense(hid_size_1, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decod_input)
        decoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(decoder_out)
        # decoder_out = tf.keras.layers.BatchNormalization()(decoder_out)
        # decoder_out = tf.keras.layers.Dropout(dropout_rate)(decoder_out)
        decoder_out = tf.keras.layers.Dense(decoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decoder_out) """

        self.encoder_model = tf.keras.Model(model_input, encoder_out, name='Encoder')
        self.decoder_model = tf.keras.Model(decod_input, decoder_out, name='Decoder')
        self.autoenco = GradModel2(model_input, self.decoder_model(self.encoder_model(model_input)), name='Autoencoder')
        self.autoenco.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, amsgrad=True), run_eagerly=True, metrics=[self.my_metrics_function])

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.autoenco.summary()

        return self.autoenco

    def encode_snapshot(self, encoder, snapshot):

        input_snap = (snapshot.T - self.data_min) / (self.data_max - self.data_min)
        encoded_snap = encoder.predict(input_snap) 

        return encoded_snap.T

    def decode_snapshot(self, decoder, encoded_snapshot):

        input_snap = encoded_snapshot.T
        decoded_snap = decoder.predict(input_snap) * (self.data_max - self.data_min) + self.data_min

        return decoded_snap.T

    def predict_snapshot(self, network, snapshot):

        a = network.predict(snapshot)

        return a

    def train_network(self, model, input_data, grad_data, learning_rate, epochs=1):
        # Train the model
        def scheduler_fnc(epoch, lr):
            new_lr = learning_rate
            return new_lr

        # feed_data = input_data.T
        # print(f"{feed_data.shape}")
        # feed_data = feed_data.reshape(feed_data.shape[0], feed_data.shape[1], 1)
        # print(f"{feed_data.shape}")

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss_x', patience=5)

        history = model.fit(
            input_data.T, grad_data,
            epochs=epochs,
            shuffle=False,
            batch_size=1,
            validation_split=0.1,
            callbacks = [
                tf.keras.callbacks.LearningRateScheduler(scheduler_fnc),
                # early_stop_callback
            ]
        )

        return history

    def test_network(self, model, input_data, grad_data):
        # feed_data = input_data.T
        # print(f"{feed_data.shape}")
        # feed_data = feed_data.reshape(feed_data.shape[0], feed_data.shape[1], 1)
        # print(f"{feed_data.shape}")
        result = model.evaluate(input_data.T, grad_data, batch_size=1)
        return result

    def set_normalization_data(self, feature_means, feature_stds):
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        return

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None
