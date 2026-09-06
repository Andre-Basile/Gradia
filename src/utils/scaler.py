class StandardScaler:
    
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X_train): # wil find mean and std for the X_train data
        self.mean =  X_train.mean(axis=0)
        self.std = X_train.std(axis=0)

    def normalize(self, data):
        if self.mean is None or self.std is None: raise Exception("Scaler must be fitted first.")
        return (data - self.mean) / self.std