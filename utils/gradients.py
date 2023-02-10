import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf

a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])

print("ab:", a@b)

tensor1 = tf.Variable(a, dtype="float64")
tensor2 = tf.Variable(b, dtype="float64")

input = tf.Variable(np.array([[1, 1, 1]]).T, dtype="float64")

with tf.GradientTape(persistent=True) as tape: 
    tape.watch(input)
    output =tf.relu(tensor1 @ tensor2 @ input)

    gradients0 = tape.gradient(output[0],  input)
    gradients1 = tape.gradient(output[1],  input)
    gradients2 = tape.gradient(output[2],  input)



print("output:", output)

print("gradient:", gradients0)
print("gradient:", gradients1)
print("gradient:", gradients2)