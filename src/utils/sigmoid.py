import numpy as np

def sigmoid(x): # 1 / (1+exp(-x))
    return  1 / (1 + np.exp(-x))