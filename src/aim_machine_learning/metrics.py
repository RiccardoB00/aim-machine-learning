import numpy as np

class Evaluator():

    def __init__(self,supported_metrics):
        self.supported_metrics=supported_metrics
        
    def set_metric(self,new_metric):
        if new_metric not in self.supported_metrics:
            raise NameError ('{} non e\' una metrica supportata'.format(new_metric))
        self.metric=new_metric
        return self
    
    def __repr__(self):
        return 'Current metric is {}'.format(self.metric)
    
    def __call__ (self,y_true,y_pred):

        if y_true.shape!=y_pred.shape:
            raise ValueError ('Le dimensioni dei dati devono essere le stesse')
        
        if self.metric== 'mse':
            mean= round(((y_true - y_pred)**2).mean(),2)
            std= round(np.std((y_true - y_pred)**2),2)
            return {'mean':mean, 'std':std}

        if self.metric== 'mae':
            mean= round((abs(y_true - y_pred)).mean(),2)
            std= round(np.std(abs(y_true - y_pred)),2)
            return {'mean':mean, 'std':std}
             
        if self.metric== 'corr':
            coef=round(np.corrcoef(y_true,y_pred)[0,1],2)
            return {'corr':coef}
    
        

    
        

