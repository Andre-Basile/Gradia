# EXCEPTIONS
class InvalidTrainDataException(Exception):pass

class Model0:
    def __init__(self):
        self.model_trained = False
        self.weigths = None
        self.bias = None
        self.losses = [] # for visualizing losses

    def check_if_valid_train_datas(self, X_train, Y_train):
        if not (X_train.shape[0] == Y_train.shape[0]): raise InvalidTrainDataException(f"\t Error : Train datas have'nt the same numbers of rows.\n\t Invalid shapes {X_train.shape} - {Y_train.shape} because {X_train.shape[0]} != {Y_train.shape[0]}")
        if not (Y_train.shape[1] == 1): raise(f"\t Error : Target train data must have only 1 column.\n\t Invalid shapes {Y_train.shape} for 'target' because {Y_train.shape[1]} != {1}")
        return True
