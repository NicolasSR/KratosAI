import numpy as np

def mean_relative_l2_error(true_data, pred_data):
    # Returns relative l2-norm error as defined in paper https://arxiv.org/pdf/2203.00360.pdf,
    # Non-linear manifold ROM with Convolutional Autoencoders and Reduced Over-Collocation method
    N=true_data.shape[0]
    err_numer=np.linalg.norm(true_data-pred_data, ord=2, axis=1)

    print('NUMERATOR ERROR L2', np.sum(err_numer)/N)
    err_denom=np.linalg.norm(true_data, ord=2, axis=1)
    print('DENOM ERROR L2', np.sum(err_denom)/N)
    return np.sum(err_numer/err_denom)/N

def relative_forbenius_error(true_data, pred_data):
    err_numer=np.linalg.norm(true_data-pred_data)
    print('NUMERATOR ERROR FROB', err_numer)
    err_denom=np.linalg.norm(true_data)
    print('DENOM ERROR FROB', err_denom)
    return err_numer/err_denom

def relative_l2_error_list(true_data, pred_data):
    err_numer=np.linalg.norm(true_data-pred_data, ord=2, axis=1)
    err_denom=np.linalg.norm(true_data, ord=2, axis=1)
    return err_numer/err_denom

def mean_l2_error(true_data, pred_data):
    # Returns relative l2-norm error as defined in paper https://arxiv.org/pdf/2203.00360.pdf,
    # Non-linear manifold ROM with Convolutional Autoencoders and Reduced Over-Collocation method
    N=true_data.shape[0]
    err_numer=np.linalg.norm(true_data-pred_data, ord=2, axis=1)
    return np.sum(err_numer)/N

def forbenius_error(true_data, pred_data):
    err_numer=np.linalg.norm(true_data-pred_data)
    return err_numer