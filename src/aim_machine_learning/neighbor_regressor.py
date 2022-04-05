
import numpy as np
from aim_machine_learning.base_regressor import Regressor
from aim_machine_learning.distance import Euclidean_Distance
from sklearn.neighbors import KNeighborsRegressor

class NeighborRegressor(Regressor):

    def __init__(self,k=1,**kwargs):
        self.k=int(k)
        self.status_fit=0
        super().__init__(**kwargs)
        
    def fit(self,x,y):
        # Si salvano samples di input ed output di training. Sono entrambi np.array e non serve lavorare con wi
        # essendo il metodo non parametrico
        self.status_fit=1  # Questo rende lecito chiamare predict. Se non facessi prima fit, non avrei con che lavorare
        self.xtrain = x  
        self.ytrain = y 
    
    def predict(self,x):

        if x.shape[1]!= self.xtrain.shape[1]:
            raise NameError('Il dataset di training e il dataset di test hanno #features diverso')

        if not self.status_fit:
            raise NameError('Non si è trainato il modello, non è possibile chiamare predict')

        e_d=Euclidean_Distance() #Si inizializza un oggetto con cui calcolare la distanza euclidea
        n_test=x.shape[0]
        n_train=self.xtrain.shape[0]
        ypredict=np.zeros(n_test) #Si inizializza un np.array di zeri che verrà riempito con le y_hat con cui si farà previsione

        #Attraverso il try-except si rende il metodo consistente per xtest aventi un qualsiasi n di features

        try:
            for row in range(n_test):
                distances=np.zeros(n_train)
                for i in range(n_train):
                    distances[i]=e_d(x[row,:],self.xtrain[i,:])
                index= np.argpartition(distances,self.k)[0:self.k]
                ypredict[row]=np.mean(self.ytrain[index])

        except IndexError:
            for row in range(n_test):
                distances=np.zeros(n_train)
                for i in range(n_train):
                    distances[i]=e_d(x[row],self.xtrain[i])   
                index= np.argpartition(distances,self.k)[0:self.k]
                ypredict[row]=np.mean(self.ytrain[index])

        return ypredict

class MySklearnNeighborRegressor (KNeighborsRegressor, Regressor):

    '''Eredita i metodi del regressore di scikit learn ed evaluate da Regressor'''

    pass 
    
    
            


                

                





