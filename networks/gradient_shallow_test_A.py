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

from sklearn.metrics import mean_absolute_error


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
        b = strategy.GetSystemVector()

        space.SetToZeroMatrix(A)
        space.SetToZeroVector(b)

        self.project_prediction(y_pred, modelpart)

        buildsol.Build(scheme, modelpart, A, b)

        b=np.array(b,copy=False)#(list(b.__iter__()))

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

    def train_step(self,data):
        w = self.w
        x_true, (r_true,s2) = data
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

            x_true_denorm = self.denormalize_data(x_true, self.data_min, self.data_max)
            A_true, b_true = self.get_r(x_true_denorm)

            v = np.array([[0.09135639, 0.06004095, 0.14518705, 0.03457825, 0.08531023, 0.1834206, 0.21735392, 0.06511215, 0.08564291, 0.16202452, 0.16087894, 0.02394319,
  0.22034533, 0.012509, 0.14015546, 0.07401453, 0.20327647, 0.06798614, 0.2118741, 0.11581635, 0.23292818, 0.05809551, 0.09521588, 0.18812852, 0.07283481,
  0.10482378, 0.02306694, 0.0024622, 0.22503593, 0.02164383, 0.23233189, 0.14397687, 0.07073004, 0.18842502, 0.06161399, 0.20712368, 0.14168534, 0.14339753,
  0.24152375, 0.09428663, 0.0150584, 0.19269708, 0.0446976, 0.01068509, 0.10909038, 0.08244887, 0.17574347, 0.14329468, 0.08244017, 0.20258278, 0.0112641,
  0.19467362]])
            print(v)
            print(np.linalg.norm(v))
            ev=v*0.001
            x_app_denorm=x_true_denorm+ev
            _, b_app = self.get_r(x_app_denorm)

            L=b_app-b_true-A_true*ev.T
            print(L)

            print(np.abs(L))
            print(np.mean(np.abs(L)))

            exit()


            xx = s2-x_true_denorm
            print(x_true_denorm)
            print(s2)
            print(xx.numpy())
            
            b_calc = tf.matmul(A_true,xx,transpose_b=True)
            b_calc=tf.transpose(b_calc)
            print('')
            print(b_calc)
            print(b_true)


            with tf.GradientTape(persistent=True) as tape_d:
                tape_d.watch(trainable_vars)
                x_pred = self(x_true, training=True)
                loss_x = self.diff_loss(x_true, x_pred)

            grad_loss_x = tape_d.gradient(loss_x,trainable_vars)
            jac_u = tape_d.jacobian(x_pred, trainable_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)

            x_pred_denorm = self.denormalize_data(x_pred, self.data_min, self.data_max)
            A_pred, b_pred = self.get_r(x_pred_denorm)
            A_pred = tf.constant(A_pred)
            # A_pred = tf.expand_dims(A_pred, axis=0)


            ## Check how similar b_true and r_true are. Thheyshould be the same
            print(r_true)
            print(b_true)
            print(b_pred)

            print(x_true)
            print(x_pred)
            print('')

            print(r_true)
            print(b_true)
            print(b_pred2_)

            print(x_true)
            print(x_pred_2)
            print('')

            r_pred = b_pred/3e9
            loss_r = r_true-r_pred
            loss_r = tf.constant(loss_r)

            total_gradients = []

            i=0
            for layer in jac_u:

                l_shape=tf.shape(layer)
                if len(l_shape) == 4:
                    layer=tf.reshape(layer,(l_shape[0],l_shape[1],l_shape[2]*l_shape[3]))
                
                # print(-A_pred)
                # print(layer)
                pre_grad=tf.linalg.matmul(-A_pred,tf.squeeze(layer,axis=0),a_is_sparse=True)
                print(pre_grad)
                print(loss_r)
                grad_loss_r=tf.matmul(loss_r,pre_grad)*(-2)
                # print(grad_loss_r)

                if len(l_shape) == 4:
                    grad_loss_r=tf.reshape(grad_loss_r,(l_shape[2],l_shape[3]))
                else:
                    grad_loss_r=tf.reshape(grad_loss_r,(l_shape[2]))

                print(grad_loss_r)
                print(trainable_vars[i])

                exit()

                total_gradients.append(w*grad_loss_x[i]+(1-w)*grad_loss_r)
                
                i+=1

            print('COMPARE STRUCTURES')
            print(trainable_vars[0])
            print(jac_u[0])
            print(total_gradients[0])
            print('')
            print(trainable_vars[9])
            print(jac_u[9])
            print(total_gradients[9])

            print(A_pred)

            exit()

            self.optimizer.apply_gradients(zip(total_gradients, train_vars))


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
