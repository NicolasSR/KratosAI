import tensorflow as tf

from tensorflow.keras.initializers import HeNormal

from networks.base_ae_factory import Base_AE_Factory
from networks.smain_ae import SnaphotMainAEModel
from networks.rmain_ae import  ResidualMainAEModel
from networks.sonly_ae import  SnapshotOnlyAEModel

from utils.normalizers import Conv2D_AE_Normalizer_ChannelRange, Conv2D_AE_Normalizer_FeatureStand, Conv2D_AE_Normalizer_ChannelScale

class Conv2D_AE_Factory(Base_AE_Factory):

    def __init__(self):
        super().__init__()

    def keras_model_selector(self,ae_config):
        if '_smain' in ae_config["nn_type"]:
            print('Using SnaphotMainAEModel model with Conv2D architecture')
            return SnaphotMainAEModel
        elif '_rmain' in ae_config["nn_type"]:
            print('Using Conv2DResidualMainAEModel model')
            return ResidualMainAEModel
        elif '_sonly' in ae_config["nn_type"]:
            print('Using Conv2DSnaphotOnlyAEModel model')
            return SnapshotOnlyAEModel
        else:
            print('No valid ae model was selected')
            return None
        
    def normalizer_selector(self, working_path, ae_config):
        if ae_config["normalization_strategy"] == 'channel_range':
            return Conv2D_AE_Normalizer_ChannelRange()
        elif ae_config["normalization_strategy"] == 'channel_scale':
            return Conv2D_AE_Normalizer_ChannelScale()
        elif ae_config["normalization_strategy"] == 'feature_stand':
            return Conv2D_AE_Normalizer_FeatureStand()
        else:
            print('Normalization strategy is not valid')
            return None
        
    def define_network(self, input_data, ae_config):

        keras_submodel=self.keras_model_selector(ae_config)

        if "use_bias" in ae_config:
            use_bias = ae_config["use_bias"]
            print('USE BIAS: ', use_bias)
        else:
            use_bias = True

        decoded_size = input_data.shape[1:]
        encoded_size = ae_config["encoding_size"]
        lay_size_x = decoded_size[0]
        lay_size_y = decoded_size[1]
        num_layers = len(ae_config["hidden_layers"])
        input_channels = input_data.shape[-1]

        model_input = tf.keras.Input(shape=(decoded_size))
        decod_input = tf.keras.Input(shape=(encoded_size,))

        encoder_out = model_input
        for layer_info in ae_config["hidden_layers"]:
            encoder_out = tf.keras.layers.Conv2D(layer_info[0], kernel_size=layer_info[1], strides=layer_info[2], activation='elu', padding='same', use_bias=use_bias)(encoder_out)
            lay_size_x=lay_size_x//layer_info[2][0]
            lay_size_y=lay_size_y//layer_info[2][1]
        encoder_out = tf.keras.layers.Flatten()(encoder_out)
        encoder_out = tf.keras.layers.Dense(encoded_size, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=use_bias)(encoder_out)
        
        decoder_out = decod_input
        flat_size=lay_size_x*lay_size_y*ae_config["hidden_layers"][-1][0]
        decoder_out = tf.keras.layers.Dense(flat_size, activation='elu', kernel_initializer=HeNormal(), use_bias=use_bias)(decoder_out)
        decoder_out = tf.keras.layers.Reshape((lay_size_x,lay_size_y,ae_config["hidden_layers"][-1][0]))(decoder_out)
        for i in range(num_layers-1):
            layer_channels=ae_config["hidden_layers"][num_layers-i-2][0]
            layer_info=ae_config["hidden_layers"][num_layers-i-1]
            decoder_out = tf.keras.layers.Conv2DTranspose(layer_channels, kernel_size=layer_info[1], strides=layer_info[2], activation='elu', padding='same', use_bias=use_bias)(decoder_out)
        layer_info = ae_config["hidden_layers"][0]
        decoder_out = tf.keras.layers.Conv2DTranspose(input_channels, kernel_size=layer_info[1], strides=layer_info[2], activation='linear', padding='same', use_bias=use_bias)(decoder_out)
        
        self.encoder_model = tf.keras.Model(model_input, encoder_out, name='Encoder')
        self.decoder_model = tf.keras.Model(decod_input, decoder_out, name='Decoder')
        self.autoenco = keras_submodel(model_input, self.decoder_model(self.encoder_model(model_input)), name='Autoencoder')
        
        self.autoenco.compile(optimizer=tf.keras.optimizers.experimental.AdamW(), run_eagerly=self.autoenco.run_eagerly, metrics=[self.my_metrics_function])

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.autoenco.summary()

        return self.autoenco, self.encoder_model, self.decoder_model