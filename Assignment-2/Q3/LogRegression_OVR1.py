import numpy as np

class LogRegression(object):
    
    def __init__(self, n_iter=1000, learn_rate = 0.03):
        self.lr = learn_rate
        self.n_iters = n_iter
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def OVR(y,i):
        for k in range(len(y)):
            if y[k] == i:
                y[k] = 1
            else:
                y[k] = 0
        return y
    
    def fit(self, X, y):   
        X = np.array(X)
        y = np.array(y).reshape(-1)
        samples, features = X.shape

        self.weights = np.zeros(features)
        self.bias = 0

        # for gradient descent
        for r in range(self.n_iters):
            # using linear model based on linear regression and then applying sigmoid to it
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / samples) * np.dot(X.T, (y_predicted - y))    #parial derivative of weights
            db = (1 / samples) * np.sum(y_predicted - y)   #parial derivative of bias
            
            # update parameters uisng regularization constant or weight decay "lambda" as lr 
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_predicted = [1 if i > 0.5 else 0 for i in y_pred]
        y_predicted = np.array(y_predicted)
        return y_predicted
    
