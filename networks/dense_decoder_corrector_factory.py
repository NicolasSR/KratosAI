import tensorflow as tf

from tensorflow.keras.initializers import HeNormal

from networks.base_ae_factory import Base_AE_Factory
from networks.scorrect_decoder import SnapshotCorrectDecoderModel

from utils.normalizers import DecCorr_Normalizer_FeatureScale, DecCorr_Normalizer_ChannelScale, Decoder_Normalizer_SVD_Whitening_NoStand, Decoder_Normalizer_SVD_Range

class Dense_Decoder_Corrector_Factory(Base_AE_Factory):

    def __init__(self):
        super().__init__()
    
    def keras_model_selector(self,ae_config, keras_default):
        if not keras_default:
            if '_correctsonly' in ae_config["nn_type"]:
                print('Using SnapshotCorrectDecoderModel model with Dense architecture')
                return SnapshotCorrectDecoderModel
            else:
                print('No valid ae model was selected')
                return None
        else:
            return tf.keras.Model
        
    def normalizer_selector(self, working_path, ae_config):
        # if ae_config["normalization_strategy"] == 'channel_scale':
        #     return AE_Normalizer_ChannelScale()
        if ae_config["normalization_strategy"] == 'feat_scale':
            return DecCorr_Normalizer_FeatureScale(working_path, ae_config["dataset_path"])
        elif ae_config["normalization_strategy"] == 'channel_scale':
            return DecCorr_Normalizer_ChannelScale(working_path, ae_config["dataset_path"])
        elif ae_config["normalization_strategy"] == 'svd_white_nostand':
            return Decoder_Normalizer_SVD_Whitening_NoStand(working_path, ae_config["dataset_path"])
        elif ae_config["normalization_strategy"] == 'svd_range':
            return Decoder_Normalizer_SVD_Range(working_path, ae_config["dataset_path"])
        else:
            print('Normalization strategy is not valid')
            return None

    def define_network(self, sample_output_data, ae_config, keras_default=False):

        keras_submodel=self.keras_model_selector(ae_config, keras_default)

        if "use_bias" in ae_config:
            use_bias = ae_config["use_bias"]
            print('USE BIAS: ', use_bias)
        else:
            use_bias = True

        decoded_size = sample_output_data.shape[1]
        encoded_size = ae_config["encoding_size"]

        decod_input = tf.keras.Input(shape=(encoded_size,))

        decoder_out = decod_input
        for layer_size in ae_config["hidden_layers"]:
            decoder_out = tf.keras.layers.Dense(layer_size, activation='elu', kernel_initializer=HeNormal(), use_bias=use_bias)(decoder_out)
        decoder_out = tf.keras.layers.Dense(decoded_size, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=use_bias)(decoder_out)

        self.decoder_model = keras_submodel(decod_input, decoder_out, name='Decoder')
        
        self.decoder_model.compile(optimizer=tf.keras.optimizers.experimental.AdamW(), run_eagerly=self.decoder_model.run_eagerly, metrics=[self.my_metrics_function])

        self.decoder_model.summary()

        return self.decoder_model, None, None