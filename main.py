import os
import pandas as pd
import numpy as np
from src.models.linear import Gradia
from src.models.logistic import Logistica
from src.utils.strings import search_substr_in_list
from src.utils.csv_read import CSV_Reader
from src.utils.scaler import StandardScaler
from src.utils.normalisation import test_lot_tolerances

regression_models = [
    ("Gradia", "linear"),
    ("Logistica", "logistic")
]
welcome_message = "Simple Machine Learning Models"
def get_model(choice, max_iter=None, tolerance=None):
    if max_iter is None:
        if choice == 0: return Gradia()
        if choice == 1: return Logistica()
    else:
        if tolerance is None:
            if choice == 0: return Gradia(max_iter=max_iter)
            if choice == 1: return Logistica(max_iter=max_iter)
        else:
            if choice == 0: return Gradia(max_iter=max_iter, tolerance=tolerance)
            if choice == 1: return Logistica(max_iter=max_iter, tolerance=tolerance)
            
    raise Exception("Invalid choice")


def wish_welcome(welcome_message):
    print(f"\t{"-" * (len(welcome_message) + 4)}\n\t| {welcome_message} |\n\t{"-" * (len(welcome_message) + 4)}")


if __name__ == '__main__':
    try:
        datas_folder_files = os.listdir('./datas')
        if(len(datas_folder_files) == 0): raise Exception("No dataset available.Add a csv file to the datas folder and try again !")
        wish_welcome(welcome_message=welcome_message)
        print('\t> Datasets availables : ')
        for index, dataset in enumerate(datas_folder_files):print(f"\t\t{index+1}.{dataset}")
        print(f"\tWhich dataset do you want to select(enter an index in [{1}, {len(datas_folder_files)}]) ?")
        choice = None

        complete_datasets = [f"{index+1}.{value}".lower() for index, value in enumerate(datas_folder_files)]
      
        while True:
            entry = input(f"\t>")
            valid = search_substr_in_list(string_array=complete_datasets, substr=entry.lower())
            if valid[0]:
                choice = datas_folder_files[valid[1]]
                break

        print(f'\nLoading dataset \'{choice}\'...')
        complete_path = f"./datas/{choice}"
        csv_reader = CSV_Reader(complete_path)
        X_train, X_test, Y_train, Y_test = csv_reader.dataset_parser()

        print('\t> Models availables : ')
        for index, model in enumerate(regression_models):print(f"\t\t{index+1}.{model[0]}({model[1]} regression)")
        print(f"\tWhich regression model do you want to select(enter an index too) ?")

        choice = None        
        complete_regressions_models = [f"{index+1}.{model[0]}({model[1]} regression)".lower() for index, model in enumerate(regression_models)]
        while True:
            entry = input(f"\t>")
            valid = search_substr_in_list(string_array=complete_regressions_models, substr=entry.lower())
            if valid[0]:
                choice = regression_models[valid[1]]
                break
        print(f'Loading {choice[1]} regression model \'{choice[0]}\'...')


        # CREATING MODEL AND OPERATIONS MAKING
        model = get_model(regression_models.index(choice), max_iter=20500, tolerance=1070)
        print(f"NOMBRES DES ITERATIONS AU DEPART : {model.iterations}")

        scaler = StandardScaler()
        scaler.fit(X_train=X_train)
        X_train_normalized = scaler.normalize(data=X_train)
        X_test_normalized = scaler.normalize(data=X_test)

        print("Training model...")
        model.train(X_train=X_train_normalized, Y_train=Y_train)
        print("Training successfully finished !")

        y_predicted = model.predict(X_test_normalized)
        print("Predictions :")
        for p, r in zip(y_predicted, Y_test): print(f"\t Prédit : {str(p):<20} -- Réel : {str(r):<15} >> écart : {abs(r - p)}")

        def mse(prediction, real):
            prediction = np.asarray(prediction).ravel()
            real = np.asarray(real).ravel()
            if prediction.shape != real.shape:
                raise ValueError(f"Prediction and target shapes differ: {prediction.shape} != {real.shape}")
            return np.mean((real - prediction) ** 2)

        # affichages

        y_train_predictions = model.predict(X=X_train_normalized)
        train_mse = mse(y_train_predictions, Y_train)
        test_mse =  mse(y_predicted, Y_test)

        print("Train MSE :", train_mse)
        print("Test MSE  :", test_mse)

        print("Train RMSE :", np.sqrt(train_mse))
        print("Test RMSE  :", np.sqrt(test_mse))

        # 
        print(f"NOMBRES DES ITERATIONS ATTEINTES : {model.iterations}")
        #tolerances = [1070, 1075, 1080]
        
       # test_lot_tolerances(tolerances=tolerances, model=model, X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
        
        

    except Exception as e:
        print(e)