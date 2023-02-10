import os
import sys

import numpy as np
import scipy

import keras
import tensorflow as tf

import utils
import networks.network as network

from keras import layers
from keras import regularizers

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as ROM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

# Create a custom Model:
loss_tracker = keras.metrics.Mean(name="loss")
loss_d_tracker = keras.metrics.Mean(name="loss_d")
loss_r_tracker = keras.metrics.Mean(name="loss_r")
mse_metric = keras.metrics.MeanSquaredError(name="mse")

class GradModel2(keras.Model):

    # Mean square error of the data
    def diff_loss(self, y_true, y_pred,):
        return (y_true - y_pred) ** 2

    # Train Step
    def train_step(self, data):
        x_true, (x_next, r_true, f_true) = data

        # Automatic Gradient
        with tf.GradientTape(persistent=True) as tape_d:
            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_next, x_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables

        gradients_x = tape_d.gradient(loss_x, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients_x, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss_x)

        # Update metrics
        mse_metric.update_state(x_next, x_pred)
            
        return {
            # "loss": loss_tracker.result(), 
            "loss_x": loss_x, 
            "mse": mse_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, mse_metric]


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

    def define_network(self, input_data, custom_loss, encoded_size):
        data = np.transpose(input_data)
        
        decoded_size = data.shape[1]
        encoded_size = 2

        ds  = decoded_size
        eAs = encoded_size * encoded_size
        ebs = encoded_size

        print(f'{encoded_size=} and {decoded_size=}')

        tfcns = tf.keras.constraints.NonNeg()

        self.model_input = tf.keras.Input(shape=(decoded_size,), dtype=tf.float64)
        self.decod_input = tf.keras.Input(shape=(encoded_size,), dtype=tf.float64)

        la = tf.keras.activations.linear
        ki = tf.keras.initializers.RandomNormal(stddev=0.01)

        self.A_encoder = tf.keras.layers.Dense(34 , activation=la, use_bias=True, kernel_initializer=ki)(self.model_input)
        self.A_encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                           (self.A_encoder)
        self.A_encoder = tf.keras.layers.Dense(20 , activation=la, use_bias=True, kernel_initializer=ki)(self.A_encoder)
        self.A_encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                           (self.A_encoder)
        self.A_encoder = tf.keras.layers.Dense(eAs, activation=la, use_bias=True, kernel_initializer=ki)(self.A_encoder)
        self.A_encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                           (self.A_encoder)
        self.A_encoder = tf.keras.layers.Reshape((ebs,ebs,), input_shape=self.A_encoder.shape)          (self.A_encoder)

        self.b_encoder = tf.keras.layers.Dense(26 , activation=la, use_bias=True, kernel_initializer=ki)(self.model_input)
        self.b_encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                           (self.b_encoder)
        self.b_encoder = tf.keras.layers.Dense(12 , activation=la, use_bias=True, kernel_initializer=ki)(self.b_encoder)
        self.b_encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                           (self.b_encoder)
        self.b_encoder = tf.keras.layers.Dense(ebs, activation=la, use_bias=True, kernel_initializer=ki)(self.b_encoder)
        self.b_encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                           (self.b_encoder)
        self.b_encoder = tf.keras.layers.Reshape((ebs,1,), input_shape=self.A_encoder.shape)            (self.b_encoder)

        self.solverAbx = tf.keras.layers.Lambda(lambda x:x)                                             (tf.linalg.matmul(tf.linalg.inv(self.A_encoder), self.b_encoder, transpose_a=True))

        self.Dx_decode = tf.keras.layers.Reshape((ebs,), input_shape=(ebs,1))                           (self.solverAbx)
        self.Dx_decode = tf.keras.layers.Dense(12 , activation=la, use_bias=True, kernel_initializer=ki)(self.Dx_decode)
        self.Dx_decode = tf.keras.layers.LeakyReLU(alpha=0.3)                                           (self.Dx_decode)
        self.Dx_decode = tf.keras.layers.Dense(26 , activation=la, use_bias=True, kernel_initializer=ki)(self.Dx_decode)
        self.Dx_decode = tf.keras.layers.LeakyReLU(alpha=0.3)                                           (self.Dx_decode)
        self.Dx_decode = tf.keras.layers.Dense(ds , activation=la, use_bias=True, kernel_initializer=ki)(self.Dx_decode)
        
        self.reconstru = tf.keras.layers.Add()([self.model_input, self.Dx_decode])

        self.autoenco = GradModel2(self.model_input, (self.reconstru, self.A_encoder, self.solverAbx, self.b_encoder))
        # self.autoenco = GradModel2(self.model_input, self.reconstru)

        self.autoenco.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True), run_eagerly=False)
        self.autoenco.summary()

        return self.autoenco, self.autoenco

    def encode_snapshot(self, encoder, snapshot):

        input_snap = (snapshot.T - self.data_min) / (self.data_max - self.data_min)
        encoded_snap = encoder.predict(input_snap) 

        return encoded_snap.T

    def decode_snapshot(self, decoder, encoded_snapshot):

        input_snap = encoded_snapshot.T
        decoded_snap = decoder.predict(input_snap) * (self.data_max - self.data_min) + self.data_min

        return decoded_snap.T

    def predict_snapshot(self, network, snapshot):

        a = network.predict(snapshot.T)

        return a

    def train_network(self, model, input_data, grad_data, num_files, epochs=1):
        # Train the model
        def scheduler_fnc(epoch, lr):
            new_lr = 6e-2
            # if epoch < 1500:
            #     new_lr = 0.01*0.2
            if epoch < 5000:
                new_lr = 5e-4
            if epoch < 1000:
                new_lr = 2.5e-3
            return new_lr

        model.fit(
            input_data.T, grad_data,
            epochs=epochs,
            shuffle=True,
            batch_size=20,
            callbacks = [
                tf.keras.callbacks.LearningRateScheduler(scheduler_fnc),
            ]
        )

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None
