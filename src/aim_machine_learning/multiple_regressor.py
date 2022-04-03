from aim_machine_learning.base_regressor import Regressor
import numpy as np

class MultipleRegressor(Regressor):

    def __init__(self, a, b):
        self.a=a
        self.b=b 
    
    def __add__ (self, model2):
        return MultipleRegressor([[self.a], [model2.a]],self.b + model2.b )

    def fit(self, X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
    
    def predict(self,X_test):
        a=np.array(self.a).reshape(-1)
        return (np.dot(X_test,a)+self.b).reshape(-1) #La stima è data dalla retta che si costruisce con a e b

    

