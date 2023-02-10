import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np 

import tensorflow as tf 
import tensorflow.keras as keras
import tensorflow.nn

a = np.array([[1,2],[3,4]], dtype="float32")
x = tf.constant([[1,1],[2,3],[3,4],[5,6]], dtype="float32")  

exact_value = x@a

class Ours(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Ours, self).__init__()
        w_init = tf.random_normal_initializer(seed=1)

        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.random_normal_initializer(seed=1)
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    # @tf.function
    def grad(self, output_value, relu_grad):
        values = output_value
        t = 0
        tf.print("\nCalculating manual gradient\n\tShape of output_value:", values.shape[1])
        for i in range(values.shape[1]):
            tf.print("\t #======================#")
            tf.print("\t #======> i:", i, values[0,i] > 0, "<======#")
            tf.print("\t #======================#")
            tf.print("\t B: ReLU_Grad: \n", relu_grad, "\n")
            if values[0,i] > 0:
                relu_grad[i,i] = 1.0
            else:
                relu_grad[i,i] = 0.0
            tf.print("\t A: ReLU_Grad: \n", relu_grad, "\n")
        
        tf.print("\t Value of the claculated gradient:\n", relu_grad, "\n")
        grad = relu_grad@self.w

        return grad

    # @tf.function  
    def call(self, inputs):
        output_value = tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
        return output_value

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

def my_loss(y,y_pred):
    my_loss = (y - y_pred)**2
    return my_loss

class CustomModel(keras.Sequential):
    
    # @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            tape.watch(x)
            with tf.GradientTape() as gradient_tape:
                gradient_tape.watch(self.trainable_variables)
                gradient_tape.watch(x)
                y_pred = self(x, training=True)  # Forward pass
                y_pred_t = tf.transpose(y_pred)

            g = gradient_tape.jacobian(y_pred_t,x,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            g = tf.reshape(g, [a.shape[0],a.shape[1]])

            loss = my_loss(y, y_pred)

            relu_grad = np.zeros( (self.layers[0].w.shape[1],self.layers[0].w.shape[1]), dtype="float32" )
            gmanual = self.layers[0].grad(y_pred, relu_grad)
            
            grad_loss = (g-a.T)**2
            old_loss = loss
            loss = loss + tf.reduce_sum(grad_loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        del tape
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {}
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, mae_metric]

my_layer = Ours(2,2)

# Define Sequential model with 3 layers
model = CustomModel(
    [
        Ours(2, 2)
    ]
)

# Call model on a test input
y = model(x)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=False)

model.compile(
    optimizer=adam_optimizer, 
    loss=my_loss, 
    run_eagerly=True
)

model.fit(
    x, exact_value,
    epochs=1,
    batch_size=1
)

print(model.layers[0].w)