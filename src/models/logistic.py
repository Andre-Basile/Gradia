import numpy as np
from src.models.model0 import Model0
from src.exceptions.exceptions import InvalidTrainDataException
from src.utils.derivatives import logistic_bias_derivative, logistic_weight_derivative
from src.configuration import LEARNING_RATE, TOLERANCE
from src.utils.loss_functions import logistic
from src.utils.sigmoid import sigmoid

# LOGISTIC REGRESSION MODEL
class Logistica(Model0):
    def __init__(self, max_iter=1000, tolerance=TOLERANCE, learning_rate=LEARNING_RATE):
        super().__init__()
        self.max_iteration = max_iter
        self.learning_rate = learning_rate
        self.tolerance = tolerance

    def train(self, X_train, Y_train):
        f = self.f
        try:
            reals = Y_train
            self.check_y_train_validity(Y_train=Y_train)
            self.check_if_valid_train_datas(X_train, Y_train)
            self.weigths = np.zeros((1, X_train.shape[1]))
            number_of_weights = len(np.ravel(self.weigths))
            self.losses = []
            self.bias = 0
            X = np.array([x.reshape(1, x.shape[0]) for x in X_train])
            iteration = 1
            
            while(True):
                predictions = np.array([f(x) for x in X]).reshape(Y_train.shape)
                derivative_bias = logistic_bias_derivative(prediction=predictions, real=reals)
                derivative_weights_matrix = np.array([logistic_weight_derivative(x_values=X_train, prediction=predictions, real=reals, column=i) for i in range(number_of_weights)])  # a matrix because we can have a lot of parameters
                
                # NEWEST WEIGHTS AND BIAS AND LOSSES
                self.weigths = self.weigths - (derivative_weights_matrix * self.learning_rate)
                self.bias = self.bias - (derivative_bias * self.learning_rate)
               
                # MANAGE LOSS
                previous_loss = self.losses[-1] if len(self.losses) > 0 else 0
                current_loss = logistic(prediction=predictions, real=reals)
                
                self.losses.append(float(current_loss))
                iteration += 1                
                loss_change = abs(previous_loss - current_loss)
                
                if (loss_change < self.tolerance) or (iteration >= self.max_iteration):break

            self.model_trained = True
                
        except Exception as e:
            raise 

    def check_y_train_validity(self, Y_train):
        uniques = np.unique(Y_train)
        if len(uniques) > 2:raise InvalidTrainDataException(f"Y train datas must have exactly two values but {len(uniques)} founded.")

    def f(self, X):  #  1 / ( 1 + exp(-y))  where --> y = bias + sum(weight_i * parameter_i) .We wanted at our ease toduplicate this function in all models
        if X.shape[0] != 1:raise InvalidTrainDataException(f"\tError : X must have only one row.")
        estimated_y = (X * self.weigths).sum() + self.bias
        return sigmoid(estimated_y)

    def predict(self, X):
        if not self.model_trained: raise Exception('Model not trained first.Call the \'train\' function.')
        return np.array([self.f(row) for row in X])

if __name__ == '__main__':
    model = Logistica()
    X = np.arange(20).reshape(4, 5)
    Y = np.arange(4).reshape(4, 1)
    print(Y[True])

    # entrainer le modèle 
    model.train(X, Y)
    x_test = np.array([[20, 21, 22, 23, 24]])