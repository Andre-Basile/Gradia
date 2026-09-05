import numpy as np

def check_shapes(real, prediction, x_values = None):
    if x_values is None:
        if not(real.shape[0] == prediction.shape[0]):raise Exception(f'\tTarget values, predictions and x_values_concerned dont hae the same numbers of rows in the <weight_derivative> function. \n\t reals : {real.shape} - predictions : {prediction.shape}')
        if not(real.shape[1] == prediction.shape[1]):raise Exception(f'\tTarget values, predictions and x_values_concerned dont hae the same numbers of columns in the <weight_derivative> function.  \n\treals : matrix{real.shape} - predictions : matrix{prediction.shape}')
    else:
        if not(real.shape[0] == prediction.shape[0] == x_values.shape[0]):raise Exception(f'Target values, predictions and x_values_concerned dont hae the same numbers of rows in the <weight_derivative> function. \n\t reals : {real.shape} - predictions : {prediction.shape} -  x_-values : {x_values.shape}')
        if not(real.shape[1] == prediction.shape[1] == x_values.shape[1]):raise Exception(f'Target values, predictions and x_values_concerned dont hae the same numbers of columns in the <weight_derivative> function.  \n\t reals : {real.shape} - predictions : {prediction.shape} -  x_-values : {x_values.shape}')
    
def quadratic_bias_derivative(prediction, real):
    N = len(list(zip(prediction, real)))
    check_shapes(real=real, prediction=prediction) 
    # formula : derv = -(1/N) * sum((real_i - prediction_i) * 2)
    s =  - 1 * (real - prediction) * 2
    return (s.sum() / N)


def quadratic_weight_derivative(x_values, prediction, real, column):
    x_values_concerned = x_values[:, column:column + 1] # the x-values concerned by the current weight
    x_values_concerned = x_values_concerned.reshape(x_values_concerned.size , 1) 
    N = len(list(zip(prediction, real)))

    check_shapes(real=real, prediction=prediction, x_values=x_values_concerned) 
    # formula : derv = -(1/N) * sum((real_i - prediction_i) * 2x_values_for_weight_i)
    s =  - 1 * (real - prediction) * (2 * x_values_concerned)
    return (s.sum() / N)

def logistic_bias_derivative(prediction, real):
    N = len(list(zip(prediction, real)))
    return (1 / N) * ((prediction - real).sum())

def logistic_weight_derivative(x_values, prediction, real, column):
    x_values_concerned = x_values[:, column:column + 1] # the x-values concerned by the current weight
    x_values_concerned = x_values_concerned.reshape(x_values_concerned.size , 1) 
    N = len(list(zip(prediction, real)))

    s =  (x_values_concerned * (prediction - real)).sum()
    return (s.sum() / N)