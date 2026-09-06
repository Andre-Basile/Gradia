import pandas as pd

class CSV_Reader:
    def __init__(self, path): 
        self.path = path
        self.content = pd.read_csv(self.path).to_numpy()

    def read_csv(self):
        return  self.content
    def target(self): return self.content[:, -1:]
    def features(self): return self.content[:, :-1]

    def dataset_parser(self, test_size=0.2):
        if not 0 <= test_size <= 1:raise Exception("Test size must be in 0 and 1.Try another value for it.")
        train_size = 1 - test_size
        cut_index = int(len(self.content) * train_size)
        X_train = self.features()[:cut_index, :]
        X_test =  self.features()[cut_index:, :]
        Y_train = self.target()[:cut_index, :]
        Y_test =  self.target()[cut_index:, :]
        return (X_train, X_test, Y_train, Y_test)