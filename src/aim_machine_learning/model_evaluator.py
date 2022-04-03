
import numpy as np

class ModelEvaluator():
    def __init__(self,model_class,params,X,y):  #Ogni qualvolta costruisco, lo faccio per una particolare coppia X,y

        self.model=model_class(**params)  # si crea l'oggetto modello con cui poi si lavorerà
        self.input=X 
        self.y=y 

    def sum_score(self,dic_result,dic_score):
        for chiave,valore in dic_score.items():
            dic_result[chiave]=dic_result.get(chiave,0)+valore
        return dic_result

    def mean_score(self,dic_result,n):
        for chiave,valore in dic_result.items():
            dic_result[chiave]=round(valore/n,2)
        return dic_result
        

    def train_test_split_eval(self,eval_obj,test_proportion): # sono argomenti non di default ma che devono essere passati obbligatoriamente

        if test_proportion <=0 or test_proportion>1:
            raise ValueError('La proporzione passata non è accettabile')
        
        # Si vuole estrarre 80% dei dati in X ed y per allenare. Si calcolano dunque le righe a cui si deve arrivare

        n_input_test=int(self.input.shape[0]*test_proportion) # Mi interesso solo al numero delle righe, non il numero dei features
        n_y_test=int(self.y.shape[0]*test_proportion) # mi interesso solo alla cardinalità, non al numero delle features

        x_test,x_train=self.input[0:n_input_test,:],self.input[n_input_test:,:] # si prende il primo 20% dei dati di input da usare come x_test
        y_test,y_train=self.y[0:n_y_test],self.y[n_y_test:]

        self.model.fit(x_train,y_train)  #Si addestra il modello 
        #Si impiega il modello addestrato per fare previsione e si valuta quanto ottenuto
        predictions= self.model.predict(x_test)

        return eval_obj(y_true=y_test,y_pred=predictions)
    
    def kfold_cv_eval(self,eval_obj,K):

        # SI TRALASCIA TEMPORANEAMENTE IL CASO DI K=1, che non è di utilità pratica
        #Devo dividere il dataset in K parti uguali, di cui le prime k-1 destinate al training
        if isinstance(K,float):
            K=int(K)

        delta=self.input.shape[0]//K
        result={}
        # Sia dato un generico K,intero,>=1
        
        # Si procede con la prima iterazione
        self.model.fit(self.input[delta:],self.y[delta:]) #primo training avviene su tutti tranne che sul primo blocco
        pred=self.model.predict(self.input[0:delta])
        score=eval_obj(self.y[0:delta],pred) #Primo dizionario di valutazione consegnato, a cui si sommeranno gli altri
        result=self.sum_score(result,score)

        #Si procede con le interazioni intermedie
        for i in range(1,K-1):
            self.model.fit(np.concatenate((self.input[0:delta*i],self.input[delta*(i+1):])),np.concatenate((self.y[0:delta*i],self.y[delta*(i+1):])))
            pred=self.model.predict(self.input[delta*i:delta*(i+1)]) #delta*i -- delta*(i+1)
            score=eval_obj(self.y[delta*i:delta*(i+1)],pred)
            result=self.sum_score(result,score)

        # Si procede con la ultima interazione --> x_test è dato dal primo blocco
        self.model.fit(self.input[0:(K-1)*delta],self.y[0:delta*(K-1)])
        pred=self.model.predict(self.input[(K-1)*delta:])
        score=eval_obj(self.y[delta*(K-1):],pred)
        result=self.sum_score(result,score)
        return self.mean_score(result,K)





    


