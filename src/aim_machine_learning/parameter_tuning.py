import numpy as np
from aim_machine_learning.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt

class ParametersTuner():

    def __init__(self, model_class, X, y, supported_eval_types, **output_path):

        # Si vede che ouput_path viene dato in input solo successivamrente, quindi non è settato di default ed è da considerarsi kwargs aggiuntivo

        self.model_class=model_class
        self.X=X
        self.y=y
        self.supported_eval_types=supported_eval_types
        self.output_path=output_path
    
    def tune_parameters(self, k_dict, eval_type, eval_obj,fig_name=None,**params):

        params_keys=list(k_dict.keys())
        neighbors=False

        if len(params_keys)==1:
            my_k=params_keys[0]
            neighbors=True 

        self.eval_type=eval_type
    
        if eval_type not in self.supported_eval_types:
            raise NameError('{} eval type not supported'.format(eval_type))
        
        min_value=float('inf') # Si assegna quantità più grande rappresentabile -> oppure min=-1 e si rafforzava il for appesantendo

        results=[]

        if neighbors:
        
            best_params={params_keys[0]:-1}
            
            for k in k_dict[my_k]: # Si ciclano i values relativi alla prima chiave (siamo nel caso KN, vale in generale?)

                mod_eval=ModelEvaluator(self.model_class, {my_k:k}, self.X, self.y) # Inizializzo oggetto di tipo ModelEvaluator

                if eval_type=='ttsplit':

                    MSE=mod_eval.train_test_split_eval(eval_obj, params['test_proportion'])
                
                else:

                    MSE=mod_eval.kfold_cv_eval(eval_obj, params['K'])
                
                # Entrambi i for ritornano un dizionario di chiavi 'mean' e 'std' , che occorre sommare ai fini della valutazione
                
                result=MSE['mean']+MSE['std']
                
                results.append(result)

                # SI vuole tenere traccia del migliore risultato e del corrispondente parametro

                if result<min_value:
                    min_value=result
                    best_params[params_keys[0]]=k
        
        else:

            best_params={params_keys[0]:-1,params_keys[1]:-1}

            for b in k_dict[params_keys[1]]:

                for a in k_dict[params_keys[0]]:

                    mod_eval=ModelEvaluator(self.model_class,{'a':a,'b':b}, self.X, self.y)

                    if eval_type=='ttsplit':

                        MSE=mod_eval.train_test_split_eval(eval_obj, params['test_proportion'])
                
                    else:

                        MSE=mod_eval.kfold_cv_eval(eval_obj, params['K'])
                
                    result=MSE['mean']+MSE['std']
                
                    results.append(result)  #E' come una matrice di dimensioni #a * #b che viene sviluppato

                    if result<min_value:
                        min_value=result
                        best_params[params_keys[0]]=a 
                        best_params[params_keys[1]]=b 

        if len(self.output_path)>0 and fig_name is not None:

            plt.figure()

            if neighbors:
                plt.plot(np.array(k_dict[my_k]), np.array(results))
                plt.xlabel(my_k)
            
            else:
                results=np.array(results).reshape(len(list(k_dict[params_keys[1]])),len(list(k_dict[params_keys[0]])))
                for i in range(len(list(k_dict[params_keys[1]]))): #si cicla sul secondo parametro(lo si tiene fissato nel plot) così da rendere corretto l'output presente nel main
                    plt.plot(k_dict[params_keys[0]],results[i,:],label='{} = {}'.format(params_keys[1],k_dict[params_keys[1]][i]))
                    plt.xlabel(params_keys[0])
                plt.legend()

            plt.plot(best_params[params_keys[0]],min_value,marker='o',markerfacecolor='navy')
            plt.title(fig_name)
            plt.ylabel('Upper bound MSE')
            plt.savefig('{} {}'.format((self.output_path)['output_path'],fig_name))
                      
        return best_params

   

