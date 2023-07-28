import abc
import datetime

import tensorflow as tf
from tensorflow.keras.initializers import HeNormal

tf.keras.backend.set_floatx('float64')

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as ROM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

from utils.custom_scheduler import CustomLearningRateScheduler

class Identity_AE_Factory():

    def define_network(self, input_size):

        model_input = tf.keras.Input(shape=(input_size))
        decod_input = tf.keras.Input(shape=(input_size,))

        IdentitiyInit = tf.keras.initializers.Identity

        encoder_out = model_input
        encoder_out = tf.keras.layers.Dense(input_size, activation=tf.keras.activations.linear, kernel_initializer=IdentitiyInit(), use_bias=False)(encoder_out)
        encoder_out = tf.keras.layers.Dense(input_size, activation=tf.keras.activations.linear, kernel_initializer=IdentitiyInit(), use_bias=False)(encoder_out)
        
        decoder_out = decod_input
        decoder_out = tf.keras.layers.Dense(input_size, activation=tf.keras.activations.linear, kernel_initializer=IdentitiyInit(), use_bias=False)(decoder_out)
        decoder_out = tf.keras.layers.Dense(input_size, activation=tf.keras.activations.linear, kernel_initializer=IdentitiyInit(), use_bias=False)(decoder_out)
        
        self.encoder_model = tf.keras.Model(model_input, encoder_out, name='Encoder')
        self.decoder_model = tf.keras.Model(decod_input, decoder_out, name='Decoder')
        self.autoenco = tf.keras.Model(model_input, self.decoder_model(self.encoder_model(model_input)), name='Autoencoder')
        
        self.autoenco.compile(optimizer=tf.keras.optimizers.experimental.AdamW(), run_eagerly=False, metrics='mse')

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.autoenco.summary()

        return self.autoenco, self.encoder_model, self.decoder_model
    

if __name__ == "__main__":

    ae_factory=Identity_AE_Factory()

    _, encoder, decoder = ae_factory.define_network(20)

    encoder.save('identity_encoder_model')
    decoder.save('identity_decoder_model')