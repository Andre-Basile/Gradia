import src.models.logistic as l
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = l.Logistica()
X = np.arange(20).reshape(4, 5)
Y = np.array((1, 0, 1, 0)).reshape(4, 1)

# Entraîner le modèle personnel.
model.train(X, Y)
print('Weights : ', model.weigths)
print('biais :', model.bias)

# Entraîner le modèle sklearn sur les mêmes données.
sklearn_model = LogisticRegression(max_iter=10000)
sklearn_model.fit(X, Y.ravel())

X_test = np.array([
    [20, 21, 22, 23, 24],
    [0, 1, 2, 3, 4],
])
custom_probabilities = np.array([model.predict(row.reshape(1, -1)) for row in X_test])
custom_predictions = (custom_probabilities >= 0.5).astype(int)
sklearn_probabilities = sklearn_model.predict_proba(X_test)[:, 1]
sklearn_predictions = sklearn_model.predict(X_test)

print('\nCOMPARAISON DES PREDICTIONS')
for index, row in enumerate(X_test):
    print(f'X = {row}')
    print(f'  Modèle personnel : probabilité={custom_probabilities[index]:.6f}, classe={custom_predictions[index]}')
    print(f'  Scikit-learn     : probabilité={sklearn_probabilities[index]:.6f}, classe={sklearn_predictions[index]}')

print('\nACCURACY SUR LES DONNÉES D\'ENTRAÎNEMENT')
custom_train_predictions = np.array([
    int(model.predict(row.reshape(1, -1)) >= 0.5) for row in X
])
print('Modèle personnel :', accuracy_score(Y.ravel(), custom_train_predictions))
print('Scikit-learn     :', accuracy_score(Y.ravel(), sklearn_model.predict(X)))