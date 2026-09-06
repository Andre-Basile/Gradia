import numpy as np
from src.utils.scaler import StandardScaler

def normalize(X):
    return ((X - X.mean(axis=0)) / X.std(axis=0), X.mean(axis=0), X.std(axis=0))

def mse(prediction, real):
            prediction = np.asarray(prediction).ravel()
            real = np.asarray(real).ravel()
            if prediction.shape != real.shape:
                raise ValueError(f"Prediction and target shapes differ: {prediction.shape} != {real.shape}")
            return np.mean((real - prediction) ** 2)


def test_lot_tolerances(tolerances:list, model, X_train, X_test, Y_train, Y_test):
    scaler = StandardScaler()
    scaler.fit(X_train=X_train)
    X_train_normalized = scaler.normalize(data=X_train)
    X_test_normalized = scaler.normalize(data=X_test)


    for tolerance in tolerances:
        model.set_tolerance(tolerance)
        model.train(X_train=X_train_normalized, Y_train=Y_train)

        y_train_predictions = model.predict(X=X_train_normalized)
        train_mse = mse(y_train_predictions, Y_train)

        y_predicted = model.predict(X=X_test_normalized)
        test_mse =  mse(y_predicted, Y_test)

        print(f"\t Tolerance = {tolerance}, MSE(train) = {train_mse}, MSE(test) = {test_mse}, RMSE(train) = {np.sqrt(train_mse)}, RMSE(test) = {np.sqrt(test_mse)}, Iterations = {model.iterations}")