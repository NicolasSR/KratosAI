import tensorflow as tf

from tensorflow.keras.initializers import HeNormal

from networks.base_ae_factory import Base_AE_Factory
from networks.smain_ae import SnaphotMainAEModel
from networks.sonly_ae import SnapshotOnlyAEModel
from networks.rmain_ae import ResidualMainAEModel
from networks.rnormmain_ae import ResidualNormMainAEModel

from utils.normalizers import AE_Normalizer_SVD_Whitening, AE_Normalizer_SVD_Whitening_NoStand, AE_Normalizer_SVD_Prenorm, AE_Normalizer_SVD_PrenormChan, AE_Normalizer_SVD, AE_Normalizer_SVD_Uniform, AE_Normalizer_ChannelScale

class SparseLayerDense(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 density,
                 use_bias=True,
                 activation=None,
                 kernel_initializer=None,
                 full="output",
                 multiple=1):
        super(SparseLayerDense, self).__init__()
        self.units = units
        self.density = density
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.full = full
        self.multiple = multiple


    def build(self, input_shape):
        self.in_features = int(input_shape[-1])

        n_parameters = self.in_features * self.units
        
        
        if self.full == "input":
            if n_parameters * self.density < self.in_features:
                self.density = self.in_features / n_parameters
                print(f"Density set to : {self.density}")
        elif self.full == "output":
            if n_parameters * self.density < self.units:
                self.density = self.units / n_parameters
                print(f"Density set to : {self.density}")
        else:
            raise NameError('full argument must be "input" or "output"')

        if self.multiple * self.density > 1.0:
            self.multiple = 1 / self.multiple
            print(f"Multiple set to : {self.multiple}")
        
        n_sparse_parameters = int(self.multiple * self.density * n_parameters)
      
        
        
        if self.full == "input":
            Total_Indexs = []
            for_each_row = n_sparse_parameters // self.in_features
            remain = n_sparse_parameters % self.in_features

            remain_index = np.random.choice(self.in_features, remain, replace=False)
            row_indexs = np.random.choice(self.in_features, self.in_features, replace=False)
            for counter, row_index in enumerate(row_indexs):
                if row_index in remain_index:
                    column_indexs = np.random.choice(self.units, for_each_row + 1, replace=False)
                else:
                    column_indexs = np.random.choice(self.units, for_each_row, replace=False)
                Total_Indexs.append(np.stack([row_index * np.ones_like(column_indexs), column_indexs], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        elif self.full == "output":
            Total_Indexs = []
            for_each_column = n_sparse_parameters // self.units
            remain = n_sparse_parameters % self.units

            remain_index = np.random.choice(self.units, remain, replace=False)
            column_indexs = np.random.choice(self.units, self.units, replace=False)
            for counter, column_index in enumerate(column_indexs):
                if column_index in remain_index:
                    row_indexs = np.random.choice(self.in_features, for_each_column + 1, replace=False)
                else:
                    row_indexs = np.random.choice(self.in_features, for_each_column, replace=False)
                Total_Indexs.append(np.stack([row_indexs, column_index * np.ones_like(row_indexs)], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        else:
            raise NameError('full argument must be "input" or "output"')
            
            
        
        if self.kernel_initializer is None:
            self.kernel = tf.Variable(tf.initializers.glorot_uniform()((n_sparse_parameters,)), trainable=True)
        else:
            self.kernel = tf.Variable(self.kernel_initializer((n_sparse_parameters,)), trainable=True)

            
            
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros((self.units,)), trainable=True)

        super(SparseLayerDense, self).build(input_shape)
    

    @tf.function
    def sparse_matmul(self,input, kernel):
        return tf.sparse.sparse_dense_matmul(input, kernel)


    def call(self, inputs):        
        new_kernel = tf.SparseTensor(indices=self.Total_Indexs,
                                     values=self.kernel,
                                     dense_shape=(self.in_features, self.units))
      
        out = self.sparse_matmul(inputs, new_kernel)
        if self.use_bias:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out) 
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class Dense_AE_Factory(Base_AE_Factory):

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
            elif '_rnormmain' in ae_config["nn_type"]:
                print('Using ResidualNNormMainAEModel model with Dense architecture')
                return ResidualNormMainAEModel
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
            encoder_out = tf.keras.layers.Dense(layer_size, activation='elu', kernel_initializer=HeNormal(), use_bias=use_bias)(encoder_out)
        encoder_out = tf.keras.layers.Dense(encoded_size, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=use_bias)(encoder_out)
        
        decoder_out = decod_input
        for i in range(num_layers):
            layer_size=ae_config["hidden_layers"][num_layers-1-i]
            decoder_out = tf.keras.layers.Dense(layer_size, activation='elu', kernel_initializer=HeNormal(), use_bias=use_bias)(decoder_out)
        decoder_out = tf.keras.layers.Dense(decoded_size, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=use_bias)(decoder_out)
        
        self.encoder_model = tf.keras.Model(model_input, encoder_out, name='Encoder')
        self.decoder_model = tf.keras.Model(decod_input, decoder_out, name='Decoder')
        self.autoenco = keras_submodel(model_input, self.decoder_model(self.encoder_model(model_input)), name='Autoencoder')
        
        self.autoenco.compile(optimizer=tf.keras.optimizers.experimental.AdamW(), run_eagerly=self.autoenco.run_eagerly, metrics=[self.my_metrics_function])
        self.encoder_model.summary()
        self.decoder_model.summary()
        self.autoenco.summary()

        # # Initialize an empty list to store non-trainable parameter names
        # non_trainable_params = []

        # # Loop through each layer in the model
        # for layer in self.autoenco.layers:
        #     # Retrieve the layer's variables/weights and add their names to the list
        #     non_trainable_params.extend([var.name for var in layer.weights])

        # # Print the names of non-trainable parameters
        # print("Non-trainable parameters:")
        # for param_name in non_trainable_params:
        #     print(param_name)

        # exit()

        # for tensor in self.encoder_model.trainable_variables:
        #     print(tensor.name)
        #     print(tensor.shape)
        #     print('')

        # # for tensor in self.decoder_model.trainable_variables:
        # #     print(tensor.name)
        # #     print(tensor.shape)
        # #     print('')

        for tensor in self.autoenco.variables:
            print(tensor.name)
            print(tensor.shape)
            print('')
        exit()

        return self.autoenco, self.encoder_model, self.decoder_model