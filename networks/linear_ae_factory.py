import tensorflow as tf

from tensorflow.keras.initializers import HeNormal

from networks.base_ae_factory import Base_AE_Factory
from networks.smain_ae import SnaphotMainAEModel
from networks.sonly_ae import SnapshotOnlyAEModel
from networks.rmain_ae import ResidualMainAEModel

from utils.normalizers import AE_Normalizer_SVD_Whitening, AE_Normalizer_SVD_Whitening_NoStand, AE_Normalizer_SVD_Prenorm, AE_Normalizer_SVD_PrenormChan, AE_Normalizer_SVD, AE_Normalizer_SVD_Uniform, AE_Normalizer_ChannelScale

class Linear_AE_Factory(Base_AE_Factory):

    def __init__(self):
        super().__init__()
    
    def keras_model_selector(self,ae_config, keras_default):
        if not keras_default:
            if '_smain' in ae_config["nn_type"]:
                print('Using SnaphotMainAEModel model with Dense architecture')
                return SnaphotMainAEModel
            elif '_sonly' in ae_config["nn_type"]:
                print('Using SnaphotOnlyAEModel model with Dense architecture')
                return SnapshotOnlyAEModel
            elif '_rmain' in ae_config["nn_type"]:
                print('Using ResidualMainAEModel model with Dense architecture')
                return ResidualMainAEModel
            else:
                print('No valid ae model was selected')
                return None
        else:
            return tf.keras.Model
        
    def normalizer_selector(self, working_path, ae_config):
        if ae_config["normalization_strategy"] == 'svd':
            return AE_Normalizer_SVD(working_path, ae_config["dataset_path"])
        elif ae_config["normalization_strategy"] == 'svd_unif':
            return AE_Normalizer_SVD_Uniform(working_path, ae_config["dataset_path"])
        elif ae_config["normalization_strategy"] == 'svd_prenorm':
            return AE_Normalizer_SVD_Prenorm(working_path, ae_config["dataset_path"])
        elif ae_config["normalization_strategy"] == 'svd_prenorm_chan':
            return AE_Normalizer_SVD_PrenormChan(working_path, ae_config["dataset_path"])
        elif ae_config["normalization_strategy"] == 'svd_white':
            return AE_Normalizer_SVD_Whitening(working_path, ae_config["dataset_path"])
        elif ae_config["normalization_strategy"] == 'svd_white_nostand':
            return AE_Normalizer_SVD_Whitening_NoStand(working_path, ae_config["dataset_path"])
        elif ae_config["normalization_strategy"] == 'channel_scale':
            return AE_Normalizer_ChannelScale()
        else:
            print('Normalization strategy is not valid')
            return None

    def define_network(self, input_data, ae_config, keras_default=False):

        keras_submodel=self.keras_model_selector(ae_config, keras_default)

        if "use_bias" in ae_config:
            use_bias = ae_config["use_bias"]
            print('USE BIAS: ', use_bias)
        else:
            use_bias = True

        decoded_size = input_data.shape[1]
        encoded_size = ae_config["encoding_size"]
        
        num_layers = len(ae_config["hidden_layers"])

        model_input = tf.keras.Input(shape=(decoded_size))
        decod_input = tf.keras.Input(shape=(encoded_size,))

        encoder_out = model_input
        for layer_size in ae_config["hidden_layers"]:
            encoder_out = tf.keras.layers.Dense(layer_size, activation='linear', kernel_initializer=HeNormal(), use_bias=use_bias)(encoder_out)
        encoder_out = tf.keras.layers.Dense(encoded_size, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=use_bias)(encoder_out)
        
        decoder_out = decod_input
        for i in range(num_layers):
            layer_size=ae_config["hidden_layers"][num_layers-1-i]
            decoder_out = tf.keras.layers.Dense(layer_size, activation='linear', kernel_initializer=HeNormal(), use_bias=use_bias)(decoder_out)
        decoder_out = tf.keras.layers.Dense(decoded_size, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=use_bias)(decoder_out)
        
        self.encoder_model = tf.keras.Model(model_input, encoder_out, name='Encoder')
        self.decoder_model = tf.keras.Model(decod_input, decoder_out, name='Decoder')
        self.autoenco = keras_submodel(model_input, self.decoder_model(self.encoder_model(model_input)), name='Autoencoder')
        
        self.autoenco.compile(optimizer=tf.keras.optimizers.experimental.AdamW(), run_eagerly=self.autoenco.run_eagerly, metrics=[self.my_metrics_function])

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.autoenco.summary()

        return self.autoenco, self.encoder_model, self.decoder_model