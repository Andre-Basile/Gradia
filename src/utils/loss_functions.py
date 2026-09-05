import numpy as np

# loss-function : Mean Square Erros(MSE)
def quadratic_mse(prediction, real):
    sum_ = ((real - prediction) ** 2).sum()
    N = len(list(zip(prediction, real)))
    return sum_ / N

def logistic(prediction, real):
    elements = (prediction ** real) * ((1 - prediction) ** (1 - real))
    N = len(list(zip(prediction, real)))
    return - np.log(np.prod(elements)) / N
