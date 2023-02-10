import os

import numpy as np

import keras
import tensorflow as tf

import utils
import networks.network as network

from keras import layers
from keras import regularizers

# Create a custom Model:
loss_tracker = keras.metrics.Mean(name="loss")
mse_metric = keras.metrics.MeanSquaredError(name="mse")

class PinnModel(keras.Model):

    def __init__(self):
        ''' Inits the model 
            ===============
            w int: weight of the data (w*data+(1-w)*PDE)
        '''

        super().__init__()
        self.w = 1

    def set_m_grad(self, m_grad):
        pass

    def set_g_weight(self, w):
        pass

    def residual(self):
        pass

    def train_step(self, data):
        x, y = data

        # Automatic Gradient
        with tf.GradientTape(persistent=True) as tape:
            pred = self(x, training=True)
            loss = self.compiled_loss(y, pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Compute our own metrics
        loss_tracker.update_state(loss)

        # Update metrics
        mse_metric.update_state(y, pred)
            
        return {"loss": loss_tracker.result(), "mse": mse_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, mse_metric]

class Pinn(network.Network):

    def __init__(self):
        '''
        Initis the network class

        model_name:     name of the directory in which the model will be saved.
        valid:          percentage of the input which is valid. (First steps are typically)
        '''
        super().__init__()

        self.model_name = "./saved_models/gradient_shallow"
        self.valid = 0.8

    def define_network(self, inp_data, out_data, custom_loss, encoded_size):

        decoded_size = inp_data.shape[0]
        outputn_size = out_data.shape[0]

        print(f'{decoded_size=} and {encoded_size=} and {outputn_size=}')

        tfcns = tf.keras.constraints.NonNeg()

        self.model_input = tf.keras.Input(shape=(decoded_size,))
        self.decod_input = tf.keras.Input(shape=(2,))

        self.encoder = tf.keras.layers.Dense(128, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1), activity_regularizer=tf.keras.regularizers.l1(10e-5))(self.model_input)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                               (self.encoder)
        self.encoder = tf.keras.layers.Dense(64 , activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),  activity_regularizer=tf.keras.regularizers.l1(10e-4))(self.encoder)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                               (self.encoder)
        self.encoder = tf.keras.layers.Dense(32 , activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),  activity_regularizer=tf.keras.regularizers.l1(10e-3))(self.encoder)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                               (self.encoder)
        self.encoder = tf.keras.layers.Dense(16 , activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),  activity_regularizer=tf.keras.regularizers.l1(10e-2))(self.encoder)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                               (self.encoder)
        self.cmprssd = tf.keras.layers.Dense(encoded_size, activation=tf.keras.activations.linear, use_bias=True, )(self.encoder)  
        
        self.decoder = tf.keras.layers.Dense(outputn_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))(self.decod_input)                                                                                                                        (self.decoder)

        self.encoder_model = PinnModel(self.model_input, self.cmprssd)
        self.decoder_model = PinnModel(self.decod_input, self.decoder)
        
        self.autoenco = PinnModel(self.encoder_model.input, self.decoder_model(self.encoder_model.output))
        self.autoenco = PinnModel(self.model_input, self.cmprssd)

        self.autoenco.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, amsgrad=True), run_eagerly=False)

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

        return a.T

    def train_network(self, model, data, sres, num_files, epochs=1):
        model.fit(
            data.T, sres.T,
            epochs=epochs,
            shuffle=True,
            batch_size=125,
        )

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None