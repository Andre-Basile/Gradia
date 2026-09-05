import numpy as np

predict = np.linspace(0, 0.5, 20)
real = np.linspace(0, 1, 20)

predict = predict.reshape(1, predict.shape[0])
real = real.reshape(1, real.shape[0])

def logistic(prediction, real):
    elements = (prediction ** real) / ((1 - prediction) ** (1 - real))
    N = len(list(zip(prediction, real)))
    return - np.log(np.prod(elements)) / N

print(logistic(predict, real))
print(np.unique(np.array([1, 23])) == [1, 2])