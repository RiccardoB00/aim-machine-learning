import numpy as np

class Euclidean_Distance():

    ''' Dati due vettori in Rn in ingresso restituisce
    la distanza euclidea tra i due'''

    def __call__(self,a,b):

        return np.sum((a-b)**2)
     
    #Non è necessario calcolare anche la radice vista la monotonia crescente della funzione sqrt