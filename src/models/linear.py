import numpy as np
from src.models.model0 import Model0
from src.exceptions.exceptions import InvalidTrainDataException
from src.utils.derivatives import quadratic_bias_derivative, quadratic_weight_derivative
from src.configuration import LEARNING_RATE, TOLERANCE
from src.utils.loss_functions import quadratic_mse


# LINEAR REGRESSION MODEL
class Gradia(Model0):
    def __init__(self, max_iter=1000, tolerance=TOLERANCE, learning_rate=LEARNING_RATE):
        super().__init__()
        self.max_iteration = max_iter
        self.learning_rate = learning_rate
        self.tolerance = tolerance

    def train(self, X_train, Y_train):
        f = self.f
        try:
            reals = Y_train
            self.check_if_valid_train_datas(X_train, Y_train) # Throw error if a problem happen here
            self.weigths = np.zeros((1, X_train.shape[1]))
            number_of_weights = len(np.ravel(self.weigths))
            self.losses = []
            self.bias = 0
            X = np.array([x.reshape(1, x.shape[0]) for x in X_train])
            iteration = 1
            
            while(True):
                predictions = np.array([f(x) for x in X]).reshape(Y_train.shape)
                derivative_bias = quadratic_bias_derivative(prediction=predictions, real=reals)
                derivative_weights_matrix = np.array([quadratic_weight_derivative(x_values=X_train, prediction=predictions, real=reals, column=i) for i in range(number_of_weights)])  # a matrix because we can have a lot of parameters
                # NEWEST WEIGHTS AND BIAS
                self.weigths = self.weigths - (derivative_weights_matrix * self.learning_rate)
                self.bias = self.bias - (derivative_bias * self.learning_rate)
                
                # MANAGING LOSS
                current_loss = quadratic_mse(prediction=predictions, real=reals)
                previous_loss = self.losses[-1] if len(self.losses) > 0 else 0
                
                self.losses.append(float(current_loss))
                iteration += 1
                loss_change = abs(previous_loss - current_loss)

               # print( f"Iteration {iteration} | " f"Loss = {current_loss} | "    f"ΔLoss = {abs(previous_loss - current_loss)}")
                #print("Iteration", iteration)
                if (loss_change < self.tolerance) or (iteration >= self.max_iteration):break
            self.model_trained = True
            self.iterations = iteration
            
        except Exception as e:
            err = "Error"
            print(f"\n\t{'-' * len(err)}\n\t{err}\n\t{'-' * len(err)}")
            print(e)


    def f(self, X):  # y = bias + sum(weight_i * parameter_i)
        if X.shape[0] != 1:raise InvalidTrainDataException(f"\tError : X must have only one row.")
        estimated_y = (X * self.weigths).sum() + self.bias
        return estimated_y
    # f(X) will be used by the logistic regression model

    def predict(self, X):
        if not self.model_trained: raise Exception('Model not trained first.Call the \'train\' function.')
        return np.array([self.f(row.reshape(1, row.shape[0])) for row in X])