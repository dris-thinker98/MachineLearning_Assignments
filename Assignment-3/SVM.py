from sklearn.svm import SVC

class SVM_RBF(object):
    def __init__(self,c,g):
        #super(SVM_RBF, self).__init__()
        self.clf = SVC(kernel='rbf',C=c,gamma=g)
        
    def fit(self, xtrain, ytrain):
        X_train = xtrain
        y_train = ytrain
 
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        #y_pred = []
        val = self.clf.decision_function(X_test)
# =============================================================================
#         for i in range(len(val)):
#             if val[i]>0:
#                 y_pred.append(1)
#             else:
#                 y_pred.append(0)
# =============================================================================
        return val