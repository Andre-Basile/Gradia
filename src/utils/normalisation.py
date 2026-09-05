import numpy as np

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


if __name__ == '__main__': # permet d'éviter de lancer ce script quand on importe
    X = np.arange(20).reshape(4, 5)
    r = normalize(X)
    print(r)