from scipy.interpolate import interp1d
import numpy as np

def fill_by_interpolation(X, Y, invalid, kind=None):
    if not isinstance(X, np.ndarray): raise Exception(f'All the arrays must be numpy arrays with a shape of only 1 column.\n\t {X} is not of type {np.ndarray}')
    if not isinstance(Y, np.ndarray): raise Exception(f'All the arrays must be numpy arrays with a shape of only 1 column.\n\t {Y} is not of type {np.ndarray}')
    if X.shape[1] != 1:raise Exception(f'All the arrays must have a shape with only 1 column.\n\t {X.shape[1]} is not equal to {1}')
    if Y.shape[1] != 1:raise Exception(f'All the arrays must have a shape with only 1 column.\n\t {Y.shape[1]} is not equal to {1}')

    X_except_invalid_values = X[Y != invalid]
    Y_except_invalid_values = Y[Y != invalid]
    f = interp1d(X_except_invalid_values, Y_except_invalid_values) # <X_except_invalid_values> and <Y_except_invalid_values> are already simple Python list even if are still 'numpy ndarray'
    new_Y = f(X.ravel())
    return new_Y