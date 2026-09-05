import numpy as np
from configuration import LEARNING_RATE, TOLERANCE
from utils.loss_functions import quadratic_mse
from utils.derivatives import quadratic_bias_derivative, quadratic_weight_derivative
from utils.graphics import draw

# EXCEPTIONS
class InvalidTrainDataException(Exception):pass

class Gradia:
    def __init__(self):
        self.model_trained = False
        self.weigths = None
        self.bias = None
        self.losses = [] # for visualizing losses for MSE

    def train(self, X_train, Y_train):
        f = self.f
        try:
            reals = Y_train
            self.check_if_valid_train_datas(X_train, Y_train) # Throw error if a problem happen here
            self.weigths = np.zeros((1, X_train.shape[1]))
            number_of_weights = len(np.ravel(self.weigths))
            self.bias = 0
            X = np.array([x.reshape(1, x.shape[0]) for x in X_train])
            iteration = 1
            while(True):
                predictions = np.array([f(x) for x in X]).reshape(Y_train.shape)
                derivative_bias = quadratic_bias_derivative(prediction=predictions, real=reals)
                derivative_weights_matrix = np.array([quadratic_weight_derivative(x_values=X_train, prediction=predictions, real=reals, column=i) for i in range(number_of_weights)])  # a matrix because we can have a lot of parameters

                # NEWEST WEIGHTS AND BIAS
                self.weigths = self.weigths - (derivative_weights_matrix * LEARNING_RATE)
                self.bias = self.bias - (derivative_bias * LEARNING_RATE)

                # MANAGING LOSS
                current_loss = quadratic_mse(prediction=predictions, real=reals)
                previous_loss = self.losses[-1] if len(self.losses) > 0 else 0
               
                self.losses.append(float(current_loss))
                iteration += 1

                loss_change = abs(previous_loss - current_loss)

                if loss_change < TOLERANCE:
                    #print(f"Convergence atteinte au bout de {iteration} itérations : loss = {loss_change}")
                    break

                if iteration >= 1000:
                    #print("Nombre maximal d'itérations atteint.")
                    break

            self.model_trained = True
            
        except Exception as e:
            print(e)

    def check_if_valid_train_datas(self, X_train, Y_train):
        if not (X_train.shape[0] == Y_train.shape[0]): raise InvalidTrainDataException(f"\t Error : Train datas have'nt the same numbers of rows.\n\t Invalid shapes {X_train.shape} - {Y_train.shape} because {X_train.shape[0]} != {Y_train.shape[0]}")
        if not (Y_train.shape[1] == 1): raise(f"\t Error : Target train data must have only 1 column.\n\t Invalid shapes {Y_train.shape} for 'target' because {Y_train.shape[1]} != {1}")
        return True

    def f(self, X):  # y = bias + sum(weight_i * parameter_i)
        if X.shape[0] != 1:raise InvalidTrainDataException(f"\tError : X must have only one row.")
        estimated_y = (X * self.weigths).sum() + self.bias
        return estimated_y
    # f(X) will be used by the logistic regression model

    def predict(self, X):
        if not self.model_trained: raise Exception('Model not trained first.Call the \'train\' function.')
        return self.f(X)
    
    def new_bias(self, derivative):
        return self.bias - (derivative * LEARNING_RATE)

    def new_weigth(self, weight_column, derivative):
        return self.weigths[0][weight_column] - (derivative * LEARNING_RATE)
""" 
def newest_param(old_param_value, param_derivative, learning_rate):
    return old_param_value - (param_derivative * learning_rate) """


if __name__ == '__main__':







    model = Gradia()
    X = np.arange(20).reshape(4, 5)
    Y = np.arange(4).reshape(4, 1)

    # entrainer le modèle 
    model.train(X, Y)
    x_test = np.array([[20, 21, 22, 23, 24]])

    print('\nPREDICTIONS SUR X = ', x_test)
    p = model.predict(x_test)
    print('y =',p)
    # draw(m.losses,xlabel='itérations', ylabel='loss', title='Losses function') 