""" Optimize a keras model using scipy.optimize
"""
import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K  # pylint: disable=import-error

from tensorflow.python.keras.engine import data_adapter


class ScipyOptimizerKratos():
    """ Implements a training function that uses scipy optimize in order
        to determine the weights for the model.

        The minimize function expects to be able to attempt multiple solutions
        over the model. It calls a function which collects all gradients for
        all steps and then returns the gradient information to the optimizer.
    """

    def __init__(self, model, method='L-BFGS-B', verbose=1, maxiter=1):
        self.model = model
        self.method = method
        self.verbose = verbose
        self.maxiter = maxiter
        if model.run_eagerly:
            self.func = model.__call__
        else:
            print("Can't run with Kratos without eager mode")
            exit()

    def _update_weights(self, x):
        x_offset = 0
        for var in self.model.trainable_variables:
            shape = var.get_shape()
            w_size = np.prod(shape)
            value = np.array(x[x_offset:x_offset+w_size]).reshape(shape)
            K.set_value(var, value)
            x_offset += w_size
        assert x_offset == len(x)

    def _fun_generator(self, x, data):
        """ Function optimized by scipy minimize.

            Returns function cost and gradients for all trainable variables.
        """
        model = self.model
        self._update_weights(x)

        w=model.w
        r_norm_factor = model.r_norm_factor

        x_true_set, (x_orig_set,r_orig_set,f_true_set) = data
        n_steps=x_true_set.shape[0]

        progbar = keras.utils.Progbar(n_steps, verbose=self.verbose)

        total_gradients=[]
        total_loss_x=0
        total_loss_r=0

        for step in range(n_steps):
            x_true=np.expand_dims(x_true_set[step], axis=0)
            x_orig=np.expand_dims(x_orig_set[step], axis=0)
            r_orig=np.expand_dims(r_orig_set[step], axis=0)
            f_true=np.expand_dims(f_true_set[step], axis=0)

            b_true=r_orig/1e9

            grad_loss_x, jac_u, loss_x, x_pred_denorm = model.get_jacobians(model.trainable_variables, x_true)
            total_loss_x+=np.sum(loss_x)

            A_pred, b_pred = model.get_r(x_pred_denorm,f_true)
            A_pred  = tf.constant(A_pred)

            err_r = b_true-b_pred
            err_r = tf.expand_dims(tf.constant(err_r),axis=0)
            loss_r = model.diff_loss(b_true, b_pred)
            total_loss_r+=np.sum(loss_r)

            i=0
            for layer in jac_u:

                l_shape=layer.shape

                last_dim_size=1
                for dim in l_shape[2:]:
                    last_dim_size=last_dim_size*dim
                layer=tf.reshape(layer,(l_shape[0],l_shape[1],last_dim_size))
                
                pre_grad=tf.linalg.matmul(A_pred,tf.squeeze(layer,axis=0),a_is_sparse=True)

                grad_loss_r=tf.matmul(err_r,pre_grad)*(-2)
                
                grad_loss_r=tf.reshape(grad_loss_r, l_shape[2:])

                if step == 0:
                    total_gradients.append(grad_loss_x[i]+w*grad_loss_r/r_norm_factor)
                else:
                    total_gradients[i]+=grad_loss_x[i]+w*grad_loss_r/r_norm_factor
                
                i+=1
            
            progbar.update(step, [('loss', loss_x), ('err_r', loss_r)])

        cost = (total_loss_x + w*total_loss_r/r_norm_factor)/n_steps
        for i in range(len(total_gradients)):
            total_gradients[i]=total_gradients[i]/n_steps

        if all(isinstance(x, tf.Tensor) for x in total_gradients):
            xgrads = np.concatenate([x.numpy().reshape(-1) for x in total_gradients])
            return cost, xgrads

        if all(isinstance(x, tf.IndexedSlices) for x in total_gradients):
            xgrad_list = []
            for var, grad in zip(model.trainable_variables, total_gradients):
                value = tf.Variable(np.zeros(var.shape), dtype=var.dtype)
                value.assign_add(grad)
                xgrad_list.append(value.numpy())
            xgrads = np.concatenate([x.reshape(-1) for x in xgrad_list])
            return cost, xgrads

        raise NotImplementedError()

    def train_function(self, data):
        """ Called by model fit.
        """
        min_options = {
            'maxiter': self.maxiter,
            'disp': bool(self.verbose),
        }

        var_list = self.model.trainable_variables
        x0 = np.concatenate([x.numpy().reshape(-1) for x in var_list])

        result = minimize(
            self._fun_generator, x0, method=self.method, jac=True,
            options=min_options, args=(data,))

        self._update_weights(result['x'])
        return {'loss': result['fun']}


def make_train_function(model, **kwargs):
    """ Returns a function that will be called to train the model.

        model._steps_per_execution must be set in order for train function to
        be called once per epoch.
    """
    model._assert_compile_was_called()  # pylint:disable=protected-access
    model._configure_steps_per_execution(tf.int64.max)  # pylint:disable=protected-access
    opt = ScipyOptimizerKratos(model, **kwargs)
    return opt.train_function
