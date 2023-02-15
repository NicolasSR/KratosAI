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

# Create a custom Model:
loss_x_tracker = keras.metrics.Mean(name="loss_x")
loss_r_tracker = keras.metrics.Mean(name="loss_r")

class GradModel2(keras.Model):

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

        return raw_A, b/3e9

    # Mean square error of the data
    def diff_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def train_step_debug(self, data):

        print("ENTERED TRAIN STEP DEBUG")

        A = np.array([[1.0, 1.0],[2.0, 2.0]]) ## Same definition as in app.py
        A_tens = tf.convert_to_tensor(A)
        x_true, r_true = data
        print(x_true.numpy())
        print(r_true.numpy())

        # Automatic Gradient
        with tf.GradientTape(persistent=True) as tape_d:
            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)
            chain_rule = tf.transpose(tf.linalg.matmul(A_tens,x_pred,transpose_b=True))
            loss_r = self.diff_loss(r_true,chain_rule)

        print(chain_rule)

        print()
            
        # loss = w * loss_x + (1-w) * loss_r

        # Compute gradients
        trainable_vars = self.trainable_variables

        gradients_x = tape_d.gradient(x_pred, trainable_vars)
        gradients_loss_x = tape_d.gradient(loss_x, trainable_vars)
        gradients_chain_rule = tape_d.gradient(chain_rule, trainable_vars)
        gradients_loss_r = tape_d.gradient(loss_r, trainable_vars)

        print('')
        print('S_pred:')
        print(x_pred.numpy())
        print('')
        print('Chain_rule (pred. residual):')
        print(chain_rule.numpy())
        print('')
        print('Trainable_vars (wights):')
        print(trainable_vars)
        print('')
        print('Gradients of S_pred:')
        print(gradients_x)
        print('')
        print('Gradients of S loss:')
        print(gradients_loss_x)
        print('')
        print('Gradients of ChainRule:')
        print(gradients_chain_rule)
        print('')
        print('Gradients of R loss:')
        print(gradients_loss_r)

        x_tr=x_true.numpy()[0]
        weights_hid=trainable_vars[0].numpy()
        bias_hid=trainable_vars[1].numpy()
        weights_out=trainable_vars[2].numpy()
        bias_out=trainable_vars[3].numpy()

        h1_bin=1.0
        h2_bin=1.0
        h1=max(x_tr[0]*weights_hid[0,0]+x_tr[1]*weights_hid[1,0]+bias_hid[0],0)
        if h1==0: h1_bin=0.0
        h2=max(x_tr[0]*weights_hid[0,1]+x_tr[1]*weights_hid[1,1]+bias_hid[1],0)
        if h2==0: h2_bin=0.0
        s1=h1*weights_out[0,0]+h2*weights_out[1,0]+bias_out[0]
        s2=h1*weights_out[0,1]+h2*weights_out[1,1]+bias_out[1]

        g_h1_o1=h1
        g_h1_o2=h1
        g_h2_o1=h2
        g_h2_o2=h2
        G_ho=[[g_h1_o1, g_h1_o2],[g_h2_o1, g_h2_o2]]

        g_i1_h1=(weights_out[0,0]+weights_out[0,1])*h1_bin*x_tr[0]
        g_i1_h2=(weights_out[1,0]+weights_out[1,1])*h2_bin*x_tr[0]
        g_i2_h1=(weights_out[0,0]+weights_out[0,1])*h1_bin*x_tr[1]
        g_i2_h2=(weights_out[1,0]+weights_out[1,1])*h2_bin*x_tr[1]
        G_ih=[[g_i1_h1,g_i1_h2],[g_i2_h1,g_i2_h2]]

        e1=-(x_tr[0]-x_pred.numpy()[0,0])
        e2=-(x_tr[1]-x_pred.numpy()[0,1])
        GL_ho=G_ho*np.array([[e1, e2],[e1, e2]])*2
        gl_i1_h1=(weights_out[0,0]*e1+weights_out[0,1]*e2)*h1_bin*x_tr[0]
        gl_i1_h2=(weights_out[1,0]*e1+weights_out[1,1]*e2)*h2_bin*x_tr[0]
        gl_i2_h1=(weights_out[0,0]*e1+weights_out[0,1]*e2)*h1_bin*x_tr[1]
        gl_i2_h2=(weights_out[1,0]*e1+weights_out[1,1]*e2)*h2_bin*x_tr[1]
        GL_ih=np.array([[gl_i1_h1,gl_i1_h2],[gl_i2_h1,gl_i2_h2]])*2

        g1_h1_o1=h1
        g2_h1_o1=0
        g1_h1_o2=0
        g2_h1_o2=h1
        g1_h2_o1=h2
        g2_h2_o1=0
        g1_h2_o2=0
        g2_h2_o2=h2

        g_vec_h1_o1=np.array([g1_h1_o1, g2_h1_o1]).T
        g_vec_h1_o2=np.array([g1_h1_o2, g2_h1_o2]).T
        g_vec_h2_o1=np.array([g1_h2_o1, g2_h2_o1]).T
        g_vec_h2_o2=np.array([g1_h2_o2, g2_h2_o2]).T
        g_vec_i1_h1=np.array([weights_out[0,0]*h1_bin*x_tr[0], weights_out[0,1]*h1_bin*x_tr[0]]).T
        g_vec_i1_h2=np.array([weights_out[1,0]*h2_bin*x_tr[0], weights_out[1,1]*h2_bin*x_tr[0]]).T
        g_vec_i2_h1=np.array([weights_out[0,0]*h1_bin*x_tr[1], weights_out[0,1]*h1_bin*x_tr[1]]).T
        g_vec_i2_h2=np.array([weights_out[1,0]*h2_bin*x_tr[1], weights_out[1,1]*h2_bin*x_tr[1]]).T
        
        g_ch_h1_o1= A[0] @ g_vec_h1_o1 + A[1] @ g_vec_h1_o1
        g_ch_h1_o2= A[0] @ g_vec_h1_o2 + A[1] @ g_vec_h1_o2
        g_ch_h2_o1= A[0] @ g_vec_h2_o1 + A[1] @ g_vec_h2_o1
        g_ch_h2_o2= A[0] @ g_vec_h2_o2 + A[1] @ g_vec_h2_o2
        g_ch_i1_h1= A[0] @ g_vec_i1_h1 + A[1] @ g_vec_i1_h1
        g_ch_i1_h2= A[0] @ g_vec_i1_h2 + A[1] @ g_vec_i1_h2
        g_ch_i2_h1= A[0] @ g_vec_i2_h1 + A[1] @ g_vec_i2_h1
        g_ch_i2_h2= A[0] @ g_vec_i2_h2 + A[1] @ g_vec_i2_h2
        
        GCH_ho=np.array([[g_ch_h1_o1,g_ch_h1_o2],[g_ch_h2_o1,g_ch_h2_o2]])
        GCH_ih=np.array([[g_ch_i1_h1,g_ch_i1_h2],[g_ch_i2_h1,g_ch_i2_h2]])

        er1=-(r_true.numpy()[0,0]-chain_rule.numpy()[0,0])
        er2=-(r_true.numpy()[0,1]-chain_rule.numpy()[0,1])

        glr_h1_o1= er1 * A[0] @ g_vec_h1_o1 + er2 * A[1] @ g_vec_h1_o1
        glr_h1_o2= er1 * A[0] @ g_vec_h1_o2 + er2 * A[1] @ g_vec_h1_o2
        glr_h2_o1= er1 * A[0] @ g_vec_h2_o1 + er2 * A[1] @ g_vec_h2_o1
        glr_h2_o2= er1 * A[0] @ g_vec_h2_o2 + er2 * A[1] @ g_vec_h2_o2
        glr_i1_h1= er1 * A[0] @ g_vec_i1_h1 + er2 * A[1] @ g_vec_i1_h1
        glr_i1_h2= er1 * A[0] @ g_vec_i1_h2 + er2 * A[1] @ g_vec_i1_h2
        glr_i2_h1= er1 * A[0] @ g_vec_i2_h1 + er2 * A[1] @ g_vec_i2_h1
        glr_i2_h2= er1 * A[0] @ g_vec_i2_h2 + er2 * A[1] @ g_vec_i2_h2

        GLR_ho=np.array([[glr_h1_o1,glr_h1_o2],[glr_h2_o1,glr_h2_o2]])*2
        GLR_ih=np.array([[glr_i1_h1,glr_i1_h2],[glr_i2_h1,glr_i2_h2]])*2

        print('')
        print('Cascaded result: ', [s1, s2])
        print('Hidden layers: ', [h1, h2])
        print('G_ho: ', G_ho)
        print('G_ih: ', G_ih)
        print('Gl_ho: ', GL_ho)
        print('GL_ih: ', GL_ih)
        print('GCH_ho: ', GCH_ho)
        print('GCH_ih: ', GCH_ih)
        print('GLR_ho: ', GLR_ho)
        print('GLR_ih: ', GLR_ih)

        print('Check gradients tensors have same length: ', len(gradients_loss_x)==len(gradients_loss_r))
        
        w = self.w
        total_gradients = []
        for i in range(len(gradients_loss_x)):
            total_gradients.append(w*gradients_loss_x[i]+(1-w)*gradients_loss_r[i])

        print(total_gradients)

        self.optimizer.apply_gradients(zip(total_gradients, trainable_vars))

        exit()

        # # Compute our own metrics
        # # loss_tracker.update_state(w * loss_x + (1-w) * loss_r)
        # loss_tracker.update_state(loss_x)

        # # Update metrics
        # mse_metric.update_state(x_true, x_pred)

        # return {
        #     "loss": loss_tracker.result(),
        #     # "loss_r": loss_r,
        #     "mse": mse_metric.result()}

    def train_step(self, data):

        print("ENTERED TRAIN STEP DEBUG")

        A = np.array([[1.0, 1.0],[2.0, 2.0]]) ## Same definition as in app.py
        A_tens = tf.convert_to_tensor(A)
        x_true, r_true = data
        print(x_true.numpy())
        print(r_true.numpy())

        train_vars = self.trainable_variables

        # Automatic Gradient
        with tf.GradientTape(persistent=True) as tape_d:
            tape_d.watch(train_vars)
            x_pred = self(x_true, training=True)

        auto_grad = tape_d.gradient(x_pred, train_vars)
        auto_diff = tape_d.jacobian(x_pred, train_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)

        print(auto_grad)
        print(auto_diff)

        x_tr=x_true.numpy()[0]
        weights_hid=train_vars[0].numpy()
        bias_hid=train_vars[1].numpy()
        weights_out=train_vars[2].numpy()
        bias_out=train_vars[3].numpy()

        h1_bin=1.0
        h2_bin=1.0
        h1=max(x_tr[0]*weights_hid[0,0]+x_tr[1]*weights_hid[1,0]+bias_hid[0],0)
        if h1==0: h1_bin=0.0
        h2=max(x_tr[0]*weights_hid[0,1]+x_tr[1]*weights_hid[1,1]+bias_hid[1],0)
        if h2==0: h2_bin=0.0
        s1=h1*weights_out[0,0]+h2*weights_out[1,0]+bias_out[0]
        s2=h1*weights_out[0,1]+h2*weights_out[1,1]+bias_out[1]

        d_o1_d_h1_o1=h1
        d_o1_d_h1_o2=0
        d_o1_d_h2_o1=h2
        d_o1_d_h2_o2=0

        d_o2_d_h1_o1=0
        d_o2_d_h1_o2=h1
        d_o2_d_h2_o1=0
        d_o2_d_h2_o2=h2

        D_o1_ho=[[d_o1_d_h1_o1, d_o1_d_h1_o2],[d_o1_d_h2_o1, d_o1_d_h2_o2]]
        D_o2_ho=[[d_o2_d_h1_o1, d_o2_d_h1_o2],[d_o2_d_h2_o1, d_o2_d_h2_o2]]
        

        d_o1_d_i1_h1=weights_out[0,0]*h1_bin*x_tr[0]
        d_o1_d_i1_h2=weights_out[1,0]*h2_bin*x_tr[0]
        d_o1_d_i2_h1=weights_out[0,0]*h1_bin*x_tr[1]
        d_o1_d_i2_h2=weights_out[1,0]*h2_bin*x_tr[1]

        d_o2_d_i1_h1=weights_out[0,1]*h1_bin*x_tr[0]
        d_o2_d_i1_h2=weights_out[1,1]*h2_bin*x_tr[0]
        d_o2_d_i2_h1=weights_out[0,1]*h1_bin*x_tr[1]
        d_o2_d_i2_h2=weights_out[1,1]*h2_bin*x_tr[1]

        D_o1_ih=[[d_o1_d_i1_h1,d_o1_d_i1_h2],[d_o1_d_i2_h1,d_o1_d_i2_h2]]
        D_o2_ih=[[d_o2_d_i1_h1,d_o2_d_i1_h2],[d_o2_d_i2_h1,d_o2_d_i2_h2]]

        print(D_o1_ho)
        print(D_o2_ho)
        print(D_o1_ih)
        print(D_o2_ih)

        # Now, let's apply the system's jacobian -A:

        cnt = tf.constant([[[-1.0]],[[2.0]]],dtype=tf.float64)
        print(cnt)

        for layer in auto_diff:
            print(layer)
            # if len(tf.shape(layer))==4:
            #     print(tf.slice(layer,begin=[0,0,0,0],size=[1,2,1,1]))
            # else:
            #     print(tf.slice(layer,begin=[0,0,0,],size=[1,2,1]))

            multip_tensor=cnt*layer
            print(multip_tensor)
            print(tf.math.reduce_sum(multip_tensor,axis=1))

            lay_shape=tf.shape(layer)
            if len(lay_shape) == 4:
                print(tf.reshape(layer,(lay_shape[0],lay_shape[1],lay_shape[2]*lay_shape[3])))

        exit()

        return

    def train_step_orig(self, data):

        print("ENTERED_TRAIN_STEP_ORIG")

        A = np.array([[1.0, 1.0],[2.0, 2.0]]) ## Same definition as in app.py
        A_tens = tf.convert_to_tensor(A)
        x_true, r_true = data

        # Automatic Gradient
        with tf.GradientTape(persistent=True) as tape_d:
            x_pred = self(x_true, training=True)
            loss_x = self.diff_loss(x_true, x_pred)
            chain_rule = tf.transpose(tf.linalg.matmul(A_tens,x_pred,transpose_b=True))
            loss_r = self.diff_loss(r_true,chain_rule)

        print(loss_x)
        print(loss_r)

        trainable_vars = self.trainable_variables
        gradients_loss_x = tape_d.gradient(loss_x, trainable_vars)
        gradients_loss_r = tape_d.gradient(loss_r, trainable_vars)

        w=self.w
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

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
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


    def define_network(self, input_data, custom_loss):
        data = np.transpose(input_data)
		
        leaky_alpha = 0.3
        
        decoded_size = data.shape[1]

        model_input = tf.keras.Input(shape=(decoded_size,))
        
        decoder_out = tf.keras.layers.Dense(decoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(model_input)
        decoder_out = tf.keras.activations.relu(decoder_out)
        decoder_out = tf.keras.layers.Dense(decoded_size, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=HeNormal())(decoder_out)
        
        self.autoenco = GradModel2(model_input, decoder_out, name='Autoencoder')
        self.autoenco.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, amsgrad=True), run_eagerly=True)

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

        a = network.predict(snapshot.T)

        return a

    def train_network(self, model, input_data, grad_data, epochs=1):
        # Train the model
        def scheduler_fnc(epoch, lr):
            new_lr = 0.5
            return new_lr

        feed_data = input_data.T
        print(f"{feed_data.shape}")
        feed_data = feed_data.reshape(feed_data.shape[0], feed_data.shape[1], 1)

        model.fit(
            input_data.T, grad_data,
            epochs=epochs,
            shuffle=False,
            batch_size=1,
            validation_split=0.1,
            callbacks = [
                tf.keras.callbacks.LearningRateScheduler(scheduler_fnc),
            ]
        )

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None
