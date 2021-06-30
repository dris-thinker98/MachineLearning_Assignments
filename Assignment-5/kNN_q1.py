import numpy as np
from collections import Counter

class kNN:
    def __init__(self, k=3):
        self.k = k

    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def fit(self, X, y):            #will be used later in _compute function
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        #X = X.transpose()
        y_pred = [self._compute(x) for x in X]
        return np.array(y_pred)

    def _compute(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]