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

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as ROM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

# Create a custom Model:
loss_tracker = keras.metrics.Mean(name="loss")
loss_d_tracker = keras.metrics.Mean(name="loss_d")
loss_r_tracker = keras.metrics.Mean(name="loss_r")
mse_metric = keras.metrics.MeanSquaredError(name="mse")

class GradModel2(keras.Model):

    # def assemble(self):
    #     KratosMultiphysics.Timer.Start("Kratos Assemble")

    #     #construct graph of system of equations 
    #     rhs = KratosMultiphysics.Vector()
    #     lhs = KratosMultiphysics.Matrix()

    #     Agraph = KratosMultiphysics.SparseContiguousRowGraph(len(self.model_part.Nodes))
    #     for elem in self.model_part.Elements:
    #         elem.Initialize(self.process_info)
    #         equation_ids = elem.EquationIdVector(self.process_info)
    #         Agraph.AddEntries(equation_ids)
    #     Agraph.Finalize()

    #     #Build FEM matrix
    #     rhs = KratosMultiphysics.Vector()
    #     lhs = KratosMultiphysics.Matrix()

    #     self.A = KratosMultiphysics.CsrMatrix(Agraph)
    #     self.b = KratosMultiphysics.SystemVector(Agraph)
    #     self.A.BeginAssemble()
    #     self.b.BeginAssemble()
    #     for elem in self.model_part.Elements:
    #         equation_ids = elem.EquationIdVector(self.process_info)
    #         elem.CalculateLocalSystem(lhs,rhs,self.process_info)
    #         self.A.Assemble(lhs,equation_ids)
    #         self.b.Assemble(rhs,equation_ids)    
    #     self.A.FinalizeAssemble()
    #     self.b.FinalizeAssemble()

    def project_prediction(self, y_pred, f_true, modelpart):
        # values = self.U @ tf.transpose(y_pred)
        values = y_pred[0]

        itr = 0
        for node in modelpart.Nodes:
            if not node.IsFixed(KMP.DISPLACEMENT_X):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_X, values[itr+0])
                # node.SetSolutionStepValue(KMP.DISPLACEMENT_X, 1, values[itr+0])
                node.X = node.X0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_X)

            if not node.IsFixed(KMP.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KMP.DISPLACEMENT_Y, values[itr+1])
                # node.SetSolutionStepValue(KMP.DISPLACEMENT_Y, 1, values[itr+1])
                node.Y = node.Y0 + node.GetSolutionStepValue(KMP.DISPLACEMENT_Y)

            itr += 2

        # for condition in modelpart.Conditions:
        #     condition.SetValue(SMA.POINT_LOAD, f_true)

    def get_r(self, y_pred, f_true):
        space =     KMP.UblasSparseSpace()
        strategy  = self.fake_simulation._GetSolver()._GetSolutionStrategy()
        buildsol  = self.fake_simulation._GetSolver()._GetBuilderAndSolver()
        scheme    = KMP.ResidualBasedIncrementalUpdateStaticScheme()
        modelpart = self.fake_simulation._GetSolver().GetComputingModelPart()
        
        A = strategy.GetSystemMatrix()
        b = KMP.Vector(52)

        space.SetToZeroMatrix(A)
        space.SetToZeroVector(b)

        self.project_prediction(y_pred, f_true, modelpart)

        buildsol.Build(scheme, modelpart, A, b)

        Ascipy = scipy.sparse.csr_matrix((A.value_data(), A.index2_data(), A.index1_data()), shape=(A.Size1(), A.Size2()))

        raw_A = -Ascipy.todense()

        return raw_A, b

    # Mean square error of the data
    def diff_loss(self, y_true, y_pred,):
        return (y_true - y_pred) ** 2

    def train_step_kk(self, data):
        x, (r, f) = data

        inp = [x]
        out = []

        # Chain Gradient
        with tf.GradientTape(persistent=True) as tape_c:
            for l in self.layers:
                print(l)
                p_it = l(inp[-1])

                inp.append(p_it)
                out.append(p_it)

            loss_c = self.diff_loss(inp[0], out[-1])
        inp = inp[:-1]
        
        gc = []
        for i in range(len(inp)-1,0,-1):
            gc.append(tape_c.gradient(loss_c, self.layers[i].trainable_variables))
        gc = gc[::-1]

        # Normal Gradient
        with tf.GradientTape(persistent=True) as tape_d:
            pred_d = self(x, training=True)

            loss_d = self.diff_loss(x, pred_d)

        trainable_vars = self.trainable_variables
        gd = tape_d.gradient(loss_d, self.trainable_variables)

        print(gc)
        print("==============")
        print(gd)

        exit(0)

    def train_step_debug(self, data):
        w = self.w
        x_true, (r_true, f_true) = data

        IDEN = np.identity(52)
        IDEN[0,0] = 0
        IDEN[1,1] = 0
        IDEN[2,2] = 0
        IDEN[3,3] = 0

        # Automatic Gradient
        with tf.GradientTape(persistent=True) as tape_d:
            x_pred, r_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)

            A_true, b_true = self.get_r(x_true, f_true[0][0])
            A_pred, b_pred = self.get_r(x_pred, f_true[0][0])
            # E_pred, e_pred = self.get_r(erro_x + x_pred, f_true[0][0])

            r_true_trim = IDEN @ tf.transpose(r_true)
            b_true_trim = IDEN @ b_true
            b_pred_trim = IDEN @ b_pred
            # e_pred_trim = IDEN @ e_pred

            # print("U", x_true)
            # print("P", x_pred)
            # print("E", erro_x)
            print("R expected:\n", r_true[0])
            print("R calc U:\n", b_true_trim)
            print("R calc P:\n", b_pred_trim)
            # print("R calc E", e_pred_trim)
            # exit()

            loss_r = self.diff_loss(0, r_pred)

            # v = -(0-b_pred_trim) @ (alpha * A_pred.T)
            # aux = x_pred @ tf.transpose(v)
            exit()

        # loss = w * loss_x + (1-w) * loss_r

        # Compute gradients
        trainable_vars = self.trainable_variables


        gradients_x = tape_d.gradient(loss_x, trainable_vars)
        gradients_r = tape_d.gradient(loss_r, trainable_vars)

        gradients = []
        for i in range(len(gradients_x)):
            if   gradients_x[i] is None:
                gradients.append(gradients_r[i] * (1-w))
            elif gradients_r[i] is None:
                gradients.append(gradients_x[i] * (w))
            else:
                gradients.append(gradients_x[i] * w + gradients_r[i] * (1-w))
        # gradients = [gx * w + (1-w) * (gr)  for gx,gr in zip(gradients_x,gradients_r)]

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        # loss_tracker.update_state(w * loss_x + (1-w) * loss_r)
        loss_tracker.update_state(loss_x)

        # Update metrics
        mse_metric.update_state(x_true, x_pred)
            
        return {
            # "loss": loss_tracker.result(), 
            "loss_x": loss_x, 
            "loss_r": loss_r,
            "mse": mse_metric.result()}

    def train_step(self, data):
        w = self.w
        print(data)
        x_true, (r_true, f_true) = data

        # Automatic Gradient
        with tf.GradientTape(persistent=True) as tape_d:
            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)

        # Compute gradients
        gradients = tape_d.gradient(loss_x, self.trainable_variables)

        # Back Propagation
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        loss_tracker.update_state(loss_x)

        # Update metrics
        mse_metric.update_state(x_true, x_pred)
            
        return {"mse": mse_metric.result()}

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

        print(f'{encoded_size=} and {decoded_size=}')

        tfcns = tf.keras.constraints.NonNeg()

        self.model_input = tf.keras.Input(shape=(decoded_size,1,))
        self.decod_input = tf.keras.Input(shape=(encoded_size,))

        self.encoder = tf.keras.layers.LSTM(2)(self.model_input)

        self.encoder = tf.keras.layers.Dense(decoded_size / 2, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(self.encoder)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)(self.encoder)
        self.encoder = tf.keras.layers.Dense(decoded_size / 2.5, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(self.encoder)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)(self.encoder)
        self.encoder = tf.keras.layers.Dense(decoded_size / 3, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(self.encoder)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)(self.encoder)
        self.cmprssd = tf.keras.layers.Dense(encoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(self.encoder)
        
        self.decoder = tf.keras.layers.Dense(decoded_size * 0.5, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(self.decod_input)
        self.decoder = tf.keras.layers.LeakyReLU(alpha=0.3)(self.decoder)
        self.decoder = tf.keras.layers.Dense(decoded_size * 1, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(self.decoder)

        self.encoder_model = GradModel2(self.model_input, self.cmprssd)
        self.decoder_model = GradModel2(self.decod_input, self.decoder)
        self.autoenco = GradModel2(self.encoder_model.input, self.decoder_model(self.encoder_model.output))
        self.autoenco.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, amsgrad=True), run_eagerly=False)

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

    def train_network(self, model, input_data, grad_data, num_files, epochs=1, mult_factor=1):
        # Train the model
        def scheduler_fnc(epoch, lr):
            new_lr = 0.0075
            return new_lr

        feed_data = input_data.T
        print(f"{feed_data.shape}")
        feed_data = feed_data.reshape(feed_data.shape[0], feed_data.shape[1], 1)


        model.fit(
            input_data.T, grad_data,
            epochs=epochs,
            shuffle=False,
            batch_size=1,
            callbacks = [
                tf.keras.callbacks.LearningRateScheduler(scheduler_fnc),
            ]
        )

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None
