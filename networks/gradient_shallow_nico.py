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
loss_r_tracker = keras.metrics.Mean(name="loss_r")

class GradModel2(keras.Model):

    def project_prediction(self, y_pred, modelpart):
        values = y_pred[0]

        itr = 0

        for node in modelpart.Nodes:
            if not node.IsFixed(KMP.DISPLACEMENT_X):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_X, values[itr+0])
                node.X = node.X0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_X)

            if not node.IsFixed(KMP.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_Y, values[itr+1])
                node.Y = node.Y0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_Y)

            itr += 2

    def get_r(self, y_pred):
        space =     KMP.UblasSparseSpace()
        strategy  = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        buildsol  = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        scheme    = KMP.ResidualBasedIncrementalUpdateStaticScheme()
        modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()
        
        A = strategy.GetSystemMatrix()
        b = KMP.Vector(52)

        space.SetToZeroMatrix(A)
        space.SetToZeroVector(b)

        self.project_prediction(y_pred, modelpart)

        buildsol.Build(scheme, modelpart, A, b)

        b=np.array(list(b.__iter__()))

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))

        raw_A = -Ascipy.todense()
        
        return raw_A, b

    # Mean square error of the data
    def diff_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def train_step_debug(self, data):

        print("ENTERED TRAIN STEP DEBUG")

        w = self.w
        x_true, r_true = data

        # IDEN = np.identity(52)
        # IDEN[0,0] = 0
        # IDEN[1,1] = 0
        # IDEN[2,2] = 0
        # IDEN[3,3] = 0

        A_true, b_true = self.get_r(x_true)

        b_true = tf.convert_to_tensor(b_true)

        A_true = tf.convert_to_tensor(A_true)
        A_true_h = A_true[:4]/1.0e9
        A_true_l = A_true[4:]/1.0e9

        print('------')
        print('A_t: ')
        print(A_true_h)
        print('x_true')
        print(x_true)
        print('b_t: ')
        print(b_true[:4])
        print('')
        chain_rule = tf.transpose(tf.linalg.matmul(A_true_h,x_true,transpose_b=True))*1.0e9
        print('chain_rule')
        print(chain_rule)
        print('')

        print('------')
        print('A_t: ')
        print(A_true_l)
        print('x_true')
        print(x_true)
        print('b_t: ')
        print(b_true[4:])
        print('')
        chain_rule = tf.transpose(tf.linalg.matmul(A_true_l,x_true,transpose_b=True))
        print('chain_rule')
        print(chain_rule)
        print('')

        # Automatic Gradient
        with tf.GradientTape(persistent=True) as tape_d:
            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)
            A_pred, b_pred = self.get_r(x_pred)
            A_pred = tf.convert_to_tensor(A_pred)
            b_pred = tf.convert_to_tensor(b_pred)
            # print('------')
            # print('b_pred: ')
            # print(type(b_pred))
            # print(b_pred)
            # print('')
            # print('A_pred: ')
            # print(type(A_pred))
            # print(A_pred)
            chain_rule = tf.transpose(tf.linalg.matmul(A_pred,x_pred,transpose_b=True))
            # print('')
            # print('Chain rule: ')
            # print(chain_rule)
            loss_r = self.diff_loss(b_true,chain_rule)

            # r_true_trim = IDEN @ tf.transpose(r_true)
            # b_true_trim = IDEN @ b_true
            # b_pred_trim = IDEN @ b_pred

        exit()

        print("R expected:\n", r_true[0])
        print("R calc U:\n", b_true)
        print("R calc P:\n", b_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients_loss_x = tape_d.gradient(loss_x, trainable_vars)
        gradients_loss_r = tape_d.gradient(loss_r, trainable_vars)

        print(trainable_vars)

        total_gradients = []
        for i in range(len(gradients_loss_x)):
            total_gradients.append(w*gradients_loss_x[i]+(1-w)*gradients_loss_r[i])

        # Backpropagation
        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        # Compute our own metrics
        loss_x_tracker.update_state(loss_x)
        loss_r_tracker.update_state(loss_r)
            
        print({"loss_x": loss_x_tracker.result(), "loss_r": loss_r_tracker})
        exit()
        return {"loss_x": loss_x_tracker.result(), "loss_r": loss_r_tracker}

    def denormalize_data(self, input_data, data_min, data_max):
        return input_data * (data_max - data_min) + data_min

    def normalize_data(self, input_data, data_min, data_max):
        return (input_data - data_min) / (data_max - data_min)

    def train_step(self, data):

        w = self.w
        x_true, r_true = data

        if w == 1:

            with tf.GradientTape(persistent=True) as tape_d:
                x_pred = self(x_true, training=True)
                loss_x = self.diff_loss(x_true, x_pred)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients_loss_x = tape_d.gradient(loss_x, trainable_vars)
        
            total_gradients = []
            for i in range(len(gradients_loss_x)):
                total_gradients.append(gradients_loss_x[i])

            # Backpropagation
            self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

            loss_x_tracker.update_state(loss_x)

            return {"loss_x": loss_x_tracker.result()}
        
        else:
            
            x_true_denorm = self.denormalize_data(x_true, self.data_min, self.data_max)
            A_true, b_true = self.get_r(x_true_denorm)
            # b_true = tf.convert_to_tensor(b_true)
            residual_true = tf.transpose(tf.linalg.matmul(A_true,x_true_denorm,transpose_b=True))
            # min_residual = tf.math.reduce_min(residual_true)
            # max_residual = tf.math.reduce_max(residual_true)
            # residual_true = self.normalize_data(residual_true, min_residual, max_residual)
            # print(residual_true)

            residual_high = residual_true*self.residuals_mask
            residual_low = residual_true*(1-self.residuals_mask)

            print(residual_high)
            print(residual_low)

            residual_high = self.normalize_data(residual_high, self.R_high_min, self.R_high_max)
            residual_low = self.normalize_data(residual_low, self.R_low_min, self.R_low_max)

            print(residual_high)
            print(residual_low)

            exit()

            # Automatic Gradient
            with tf.GradientTape(persistent=True) as tape_d:
                x_pred = self(x_true, training=True)
                x_pred_denorm = self.denormalize_data(x_pred, self.data_min, self.data_max)
                loss_x = self.diff_loss(x_true, x_pred)
                # A_pred, b_pred = self.get_r(x_pred)
                # A_pred = tf.convert_to_tensor(A_pred)
                # b_pred = tf.convert_to_tensor(b_pred)
                # chain_rule = tf.transpose(tf.linalg.matmul(A_true,x_pred,transpose_b=True))
                chain_rule = tf.transpose(tf.linalg.matmul(A_true,x_pred_denorm,transpose_b=True))
                chain_rule = self.normalize_data(chain_rule, min_residual, max_residual)
                print(chain_rule)
                loss_r = self.diff_loss(residual_true,chain_rule)

            print(loss_r)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients_loss_x = tape_d.gradient(loss_x, trainable_vars)
            # print('GRADIENT 1')
            # print(gradients_loss_x)
            gradients_loss_r = tape_d.gradient(loss_r, trainable_vars)
            # print(' ')
            # print('GRADIENT 2')
            # print(gradients_loss_r)
            
            total_gradients = []
            for i in range(len(gradients_loss_x)):
                total_gradients.append(w*gradients_loss_x[i]+(1-w)*gradients_loss_r[i])

            # Backpropagation
            self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

            # Compute our own metrics
            loss_x_tracker.update_state(loss_x)
            loss_r_tracker.update_state(loss_r)

            return {"loss_x": loss_x_tracker.result(), "loss_r": loss_r_tracker.result()}

    def test_step(self, data):
        w = self.w
        x_true, r_true = data

        if w == 1:

            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)

            loss_x_tracker.update_state(loss_x)

            return {"loss_x": loss_x_tracker.result()}

        else:
            print('CASE WITH W=/=1 NOT IMPLEMENTED')
            exit()

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

    def define_network(self, input_data, custom_loss, encoded_size):
        data = np.transpose(input_data)
		
        leaky_alpha = 0.3
        dropout_rate = 0.1
        
        decoded_size = data.shape[1]

        print(f'{encoded_size=} and {decoded_size=}')

        model_input = tf.keras.Input(shape=(decoded_size,))
        decod_input = tf.keras.Input(shape=(encoded_size,))

        hid_size_1 = int((decoded_size-encoded_size)*2/3+encoded_size)
        hid_size_2 = int((decoded_size-encoded_size)/3+encoded_size)
        
        encoder_out = tf.keras.layers.Dense(hid_size_1, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(model_input)
        encoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(encoder_out)
        # encoder_out = tf.keras.layers.Dropout(dropout_rate)(encoder_out)
        encoder_out = tf.keras.layers.Dense(hid_size_2, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(encoder_out)
        encoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(encoder_out)
        # encoder_out = tf.keras.layers.Dropout(dropout_rate)(encoder_out)
        encoder_out = tf.keras.layers.Dense(encoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(encoder_out)
        encoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(encoder_out)

        decoder_out = tf.keras.layers.Dense(hid_size_2, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decod_input)
        decoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(decoder_out)
        # decoder_out = tf.keras.layers.Dropout(dropout_rate)(decoder_out)
        decoder_out = tf.keras.layers.Dense(hid_size_1, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decoder_out)
        decoder_out = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)(decoder_out)
        # decoder_out = tf.keras.layers.Dropout(dropout_rate)(decoder_out)
        decoder_out = tf.keras.layers.Dense(decoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decoder_out)

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

    def train_network(self, model, input_data, grad_data, epochs=1):
        # Train the model
        def scheduler_fnc(epoch, lr):
            new_lr = 0.005
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

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None
