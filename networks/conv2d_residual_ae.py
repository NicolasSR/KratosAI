import os
import sys
import math

import datetime

import numpy as np
import scipy

import keras
import tensorflow as tf

from tensorflow.keras.initializers import HeNormal

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as ROM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

from utils.custom_scheduler import CustomLearningRateScheduler


# Create a custom Model:
loss_x_tracker = keras.metrics.Mean(name="loss_x")
loss_r_tracker = keras.metrics.Mean(name="err_r")

class Conv2DResidualAEModel(keras.Model):

    def __init__(self,*args,**kwargs):
        super(Conv2DResidualAEModel,self).__init__(*args,**kwargs)
        self.normalization_mode = None
        self.feat_means = None
        self.feat_stds = None
        self.data_min = None
        self.data_max = None
        self.w=0
        self.residual_weights_vec=None
        self.r_norm_factor=0
        self.max_grad_diff=1

    def set_config_values(self, ae_config, R, data_normalizer):
        self.set_residual_weighting_vec(R)
        self.data_normalizer=data_normalizer
        # self.loss_combination_method = ae_config["loss_combination_method"]

    def project_prediction(self, y_pred, f_true, modelpart):
        values = y_pred[0]
        f_value=f_true[0]

        itr = 0
        for node in modelpart.Nodes:
            if not node.IsFixed(KMP.DISPLACEMENT_X):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_X, values[itr+0])
                node.X = node.X0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_X)

            if not node.IsFixed(KMP.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_Y, values[itr+1])
                node.Y = node.Y0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_Y)

            if not node.IsFixed(KMP.DISPLACEMENT_X):
                itr += 2
            
        for condition in modelpart.Conditions:
            condition.SetValue(SMA.POINT_LOAD, f_value)

    def get_r(self, y_pred, f_true):
        space =     KMP.UblasSparseSpace()
        strategy  = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        buildsol  = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        scheme    = KMP.ResidualBasedIncrementalUpdateStaticScheme()
        modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()

        A = strategy.GetSystemMatrix()
        b = strategy.GetSystemVector()

        space.SetToZeroMatrix(A)
        space.SetToZeroVector(b)

        self.project_prediction(y_pred, f_true, modelpart)

        buildsol.Build(scheme, modelpart, A, b)

        # buildsol.ApplyDirichletConditions(scheme, modelpart, A, b, b)

        b=np.array(b)

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))

        raw_A = -Ascipy.todense()
        raw_A = raw_A[:,4:]
        
        return raw_A/1e9, b/1e9

    # Mean square error of the data
    def diff_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def set_residual_weighting_vec(self, R_orig):
        self.residual_weights_vec=np.ones(R_orig.shape[1], dtype=np.float64)
        print('residual_weights_vec', self.residual_weights_vec, self.residual_weights_vec.shape)
        # init_vec = np.mean(np.abs(R_orig), axis=0)
        # print('residual_weights_vec', init_vec)
        # self.residual_weights_vec=[]
        # for weight in init_vec:
        #     if weight > 1e0:
        #         self.residual_weights_vec.append(1/weight)
        #     else:
        #         self.residual_weights_vec.append(1)
        # self.residual_weights_vec=np.array(self.residual_weights_vec)
        # print('residual_weights_vec', self.residual_weights_vec)

    @tf.function
    def get_jacobians(self, trainable_vars, x_true):
        with tf.GradientTape(persistent=True) as tape_d:
            tape_d.watch(trainable_vars)
            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)
            x_pred_flat = self.data_normalizer.reorganize_into_original_tf(x_pred)
            x_pred_denorm = self.data_normalizer.denormalize_data_tf(x_pred_flat)

        grad_loss_x = tape_d.gradient(loss_x,trainable_vars)
        jac_u = tape_d.jacobian(x_pred_denorm, trainable_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)
        return grad_loss_x, jac_u, loss_x, x_pred_denorm

    def train_step(self,data):
        w = self.w
        r_norm_factor = self.r_norm_factor
        x_true, (x_orig,r_orig,f_true) = data
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

            b_true=r_orig/1e9

            grad_loss_x, jac_u, loss_x, x_pred_denorm = self.get_jacobians(trainable_vars, x_true)

            A_pred, b_pred = self.get_r(x_pred_denorm,f_true)
            A_pred  = tf.constant(A_pred)

            err_r = b_true-b_pred
            err_r = tf.expand_dims(tf.constant(err_r),axis=0)
            loss_r = self.diff_loss(b_true, b_pred)

            total_gradients = []

            i=0
            for layer in jac_u:

                l_shape=layer.shape

                last_dim_size=1
                for dim in l_shape[2:]:
                    last_dim_size=last_dim_size*dim
                layer=tf.reshape(layer,(l_shape[0],l_shape[1],last_dim_size))
                
                pre_grad=tf.linalg.matmul(A_pred,tf.squeeze(layer,axis=0),a_is_sparse=True)

                aux_err=tf.multiply(err_r,self.residual_weights_vec)
                grad_loss_r=tf.matmul(aux_err,pre_grad)*(-2)
                
                grad_loss_r=tf.reshape(grad_loss_r, l_shape[2:])

                # self.max_grad_diff=max(np.max((np.abs(grad_loss_r)-np.abs(grad_loss_x[i]))/np.abs(grad_loss_x[i])), self.max_grad_diff)
                # print(np.mean((np.abs(grad_loss_r)-np.abs(grad_loss_x[i]))/np.abs(grad_loss_x[i])))
                # print(np.min((np.abs(grad_loss_r)-np.abs(grad_loss_x[i]))/np.abs(grad_loss_x[i])))
                # print(np.max((np.abs(grad_loss_r)-np.abs(grad_loss_x[i]))/np.abs(grad_loss_x[i])))

                total_gradients.append(w*grad_loss_x[i]+(1-w)*grad_loss_r/r_norm_factor)
                # total_gradients.append(grad_loss_x[i]+w*grad_loss_r/r_norm_factor)
                
                i+=1

            # print(total_gradients)

            self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

            # Compute our own metrics
            loss_x_tracker.update_state(loss_x)
            loss_r_tracker.update_state(loss_r)

            return {"loss_x": loss_x_tracker.result(), "err_r": loss_r_tracker.result()}


    def test_step(self, data):
        w = self.w
        x_true, (r_orig,f_true) = data

        if w == 1:

            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)

            loss_x_tracker.update_state(loss_x)

            return {"loss_x": loss_x_tracker.result()}

        else:
            b_true=r_orig/1e9

            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)
            x_pred_flat = self.data_normalizer.reorganize_into_original_tf(x_pred)
            x_pred_denorm = self.data_normalizer.denormalize_data_tf(x_pred_flat)

            _, b_pred = self.get_r(x_pred_denorm,f_true)
        
            loss_r = self.diff_loss(b_true, b_pred)

            # Compute our own metrics
            loss_x_tracker.update_state(loss_x)
            loss_r_tracker.update_state(loss_r)

            return {"loss_x": loss_x_tracker.result(), "err_r": loss_r_tracker.result()}
            
    def get_r_array(self, samples, F_true):
        samples_flat = self.data_normalizer.reorganize_into_original_tf(samples)
        samples_denorm = self.data_normalizer.denormalize_data_tf(samples_flat)
        b_list=[]
        for i, sample in enumerate(samples_denorm):
            _, b = self.get_r(tf.expand_dims(sample, axis=0), F_true[i])
            b_list.append(b)
        b_array=np.array(b_list)
        return b_array

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


class Conv2D_Residual_AE():

    def __init__(self):

        super().__init__()

    def my_metrics_function(_,y_true,y_pred):
        return (y_true - y_pred) ** 2

    def define_network(self, input_data, ae_config):

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
            encoder_out = tf.keras.layers.Conv2D(layer_info[0], kernel_size=layer_info[1], strides=layer_info[2], activation='elu', padding='same')(encoder_out)
            # encoder_out = tf.keras.layers.Conv2D(layer_info[0], kernel_size=layer_info[1], strides=layer_info[2], activation='linear', padding='same')(encoder_out)
            # encoder_out = tf.keras.layers.LeakyReLU(alpha=0.3)(encoder_out)
            lay_size_x=lay_size_x//layer_info[2][0]
            lay_size_y=lay_size_y//layer_info[2][1]
        encoder_out = tf.keras.layers.Flatten()(encoder_out)
        encoder_out = tf.keras.layers.Dense(encoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(encoder_out)
        
        decoder_out = decod_input
        flat_size=lay_size_x*lay_size_y*ae_config["hidden_layers"][-1][0]
        decoder_out = tf.keras.layers.Dense(flat_size, activation='elu', use_bias=True, kernel_initializer=HeNormal())(decoder_out)
        # decoder_out = tf.keras.layers.Dense(flat_size, activation='linear', use_bias=True, kernel_initializer=HeNormal())(decoder_out)
        # decoder_out = tf.keras.layers.LeakyReLU(alpha=0.3)(decoder_out)
        decoder_out = tf.keras.layers.Reshape((lay_size_x,lay_size_y,ae_config["hidden_layers"][-1][0]))(decoder_out)
        for i in range(num_layers-1):
            layer_channels=ae_config["hidden_layers"][num_layers-i-2][0]
            layer_info=ae_config["hidden_layers"][num_layers-i-1]
            decoder_out = tf.keras.layers.Conv2DTranspose(layer_channels, kernel_size=layer_info[1], strides=layer_info[2], activation='elu', padding='same')(decoder_out)
            # decoder_out = tf.keras.layers.Conv2DTranspose(layer_channels, kernel_size=layer_info[1], strides=layer_info[2], activation='linear', padding='same')(decoder_out)
            # decoder_out = tf.keras.layers.LeakyReLU(alpha=0.3)(decoder_out)
        layer_info = ae_config["hidden_layers"][0]
        decoder_out = tf.keras.layers.Conv2DTranspose(input_channels, kernel_size=layer_info[1], strides=layer_info[2], activation='linear', padding='same')(decoder_out)
        
        self.encoder_model = tf.keras.Model(model_input, encoder_out, name='Encoder')
        self.decoder_model = tf.keras.Model(decod_input, decoder_out, name='Decoder')
        self.autoenco = Conv2DResidualAEModel(model_input, self.decoder_model(self.encoder_model(model_input)), name='Autoencoder')
        
        
        self.autoenco.compile(optimizer=tf.keras.optimizers.experimental.AdamW(), run_eagerly=True, metrics=[self.my_metrics_function])

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.autoenco.summary()

        return self.autoenco, self.encoder_model, self.decoder_model

    def predict_snapshot(self, network, snapshot):

        a = network.predict(snapshot)

        return a

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
            new_w=ae_config["residual_loss_ratio"][1]
            slope_width=ae_config["residual_loss_ratio"][3]
            slope_min=ae_config["residual_loss_ratio"][2]
            new_w=max((slope_width-epoch)/slope_width*new_w,slope_min)
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
        
        def r_norm_const_schedule(epoch, max_grad_diff):
            new_r_norm_factor=ae_config["residual_norm_factor"][1]
            return new_r_norm_factor

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

        if ae_config["residual_norm_factor"][0]=='const':
            r_norm_schedule=r_norm_const_schedule
        else: print('Unvalid r_norm scheduler')

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss_x', patience=5)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_best_callback = keras.callbacks.ModelCheckpoint(ae_config["models_path"]+ae_config["name"]+"/best/weights_{epoch:03d}.h5",save_weights_only=True,save_best_only=True,monitor="val_loss_x",mode="min")
        checkpoint_last_callback = keras.callbacks.ModelCheckpoint(ae_config["models_path"]+ae_config["name"]+"/last/weights.h5",save_weights_only=True,save_freq="epoch")
        lr_w_scheduler_callback = CustomLearningRateScheduler(lr_schedule, w_schedule, r_norm_schedule ,0)

        history = model.fit(
            input_data, grad_data,
            epochs=ae_config["epochs"],
            shuffle=True,
            batch_size=ae_config["batch_size"],
            validation_data=(val_input,val_truth),
            callbacks = [
                lr_w_scheduler_callback,
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
            x_pred_denorm = autoencoder_model.denormalize_data(x_pred)

        jac = tape_d.jacobian(x_pred_denorm, in_tf_var, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)
        # print(in_tf_var)
        # print(jac)
        # print(jac[0,:,0,:])
        # print('Jacobian shape:', jac[0,:,0,:].shape)
        return jac[0,:,0,:].numpy()

    def compute_full_gradient():
        return None