
import os
import math
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

''' Function to be evaluated '''
def f(q):
    return np.array([q[0]**3,3*q[1]**2])

def ef(q):
    return tf.math.pow(q,[3,2]) * [1,3]

''' First derivative of f(q) '''
def df(q):
    return np.array([3*q[0]**2,6*q[1]])

''' First derivative of f(q) '''
def auto_df(q):
    in_tf_var = tf.Variable([q])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(in_tf_var)
        auto_output = ef(in_tf_var)
    # output = df(q)

    auto_grad = tape.batch_jacobian(auto_output, in_tf_var, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False) 

    # Compute gradients
    return auto_grad[0].numpy()

''' How to compute x axis values '''
def xAxis(q,dq,ε):
    return np.array([np.log(ε[i]) for i in range(10)])

''' How to compute y axis values '''
def yAxis(q,dq,ε):
    
    res = []
    for e in ε:
        dq_effective = e*dq
        q_perturbed = q + dq_effective
        f_q = f(q)
        f_q_perturbed = f(q_perturbed)
        r = np.linalg.norm( f_q_perturbed- f_q - dq_effective @ auto_df(q))
        print("for eps = ,",e," r=",r, " f(q)=",np.linalg.norm(f_q-f_q_perturbed) )
        res.append(math.log(r))
        
    return res

''' Calculate Slope '''
def slope(x,y):
    return np.polyfit(x, y, 1)[0]

''' Check that gradient slope is correct '''
def main():
    # Get dq something small
    dq  = np.random.normal(0, 1, size=(2,))
    dq /= np.linalg.norm(dq)

    # Get a range of ε values to evaluate the gradient at
    ε = [1e-1/(2**(1+e)) for e in range(10)]

    # Evaluate at different points
    q = np.array([4.0,7.0])
    
    # Get the x,y axis values for the plot
    x  = xAxis(q,dq,ε)
    y  = yAxis(q,dq,ε)

    # Check the slope
    s = slope(x,y)

    # Plot
    plt.plot(x, y, label=f'at q={q}, slope={s:0.2f}')

    # plt.rcParams.update({
    #     "text.usetex": True
    # })

    plt.title('$f(q)=||f(q+\\epsilon{}dq)-f(q)-\\nabla{}f|_{q}\\epsilon{}dq||$')
    plt.legend(loc="upper left")
    plt.xlabel('$log(\\epsilon{})$')
    plt.ylabel('$log(f(q))$')
    plt.show()

if __name__ == "__main__":
    main()