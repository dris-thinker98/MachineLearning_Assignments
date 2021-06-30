from sklearn.linear_model import LinearRegression 
import numpy as np

class Regression(object):    

    y_predicted = 0

    def __init__(self):
        super(Regression, self).__init__()
        self.coeff_ = None
        self.intercept_ = None

    def fit(self, xtrain, ytrain):
        X_train = xtrain
        y_train = ytrain

        LinReg = LinearRegression()
        fitted = LinReg.fit(X_train, y_train)
        self.coeff_ = fitted.coef_
        self.intercept_ = fitted.intercept_

    def predict(self, X_test):
        #if len(X_test.shape) == 1:
           # X_test = X_test.reshape(-1,1)
        test = X_test.to_numpy()
        test = np.transpose(test)
        self.predicted_ = np.dot(self.coeff_ , test) + self.intercept_
        y_predicted = self.predicted_
        y_predicted = np.transpose(y_predicted)
        return y_predicted

    def calculate_MSE(self, y_test, y_pred):
        test = np.array(y_test)
        pred = y_pred
        total = 0.000000
        for i in range(len(y_pred)):
            sq_diff = (test[i] - pred[i])*(test[i] - pred[i])
            total += sq_diff
        mse = total/len(y_pred)
        return mse