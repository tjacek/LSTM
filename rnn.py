import theano
import theano.tensor as T
import numpy as np
import pickle
import lstm,tools,optim,layer

class RNN(object):
    def __init__(self,hyper_params,
    	              params,nn_vars,
    	              pred,loss,updates):
        in_var=nn_vars['in_var']
        target_var=nn_vars['target_var']
        #print(dir(params[0]))
        self.params=params
        self.pred=theano.function([in_var], pred,allow_input_downcast=True,on_unused_input='warn')        
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)

    def get_params_values(self):
        param_values=[ (param_i.name,param_i.get_value()) 
                 for param_i in self.params]
        return dict(param_values)

    def set_params(self,values):
        for param_i in self.params:
            value_i=values[param_i.name]
            param_i.set_value(value_i)
  
    def save(self,filename):
        params=self.get_params_values()
        pickle.dump(params,open(filename, "wb" ))

    def read(self,filename):
        with open(filename, 'rb') as handle:
             values = pickle.load(handle)
             self.set_params(values)

class MaskRNN(object):
    def __init__(self,hyper_params,
                      nn_vars,
                      pred,loss,updates):
        in_var=nn_vars['in_var']
        target_var=nn_vars['target_var']
        mask_var=nn_vars['mask_var']
        self.pred=theano.function([in_var,mask_var], pred,allow_input_downcast=True,on_unused_input='warn')        
        self.loss=theano.function([in_var,target_var,mask_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var,mask_var], loss, 
                               updates=updates,allow_input_downcast=True)

def build_rnn(hyper_params):
    nn_vars=init_variables(False)
    builder=lstm.LSTMBuilder(hyper_params,True)
    hidden=builder.get_output(nn_vars)
    params=builder.get_params()
    #pred_y,loss=regresion(hidden,nn_vars['target_var'])
    pred_y,loss=prob(hidden,nn_vars['target_var'],hyper_params,params)
    
    updates=optim.momentum_sgd(loss,hyper_params,params)
    return RNN(hyper_params,params,nn_vars,pred_y,loss,updates)

def regresion(hidden,target_var):
    pred_y= T.mean(T.mean(hidden,axis=0),axis=1)
    loss=T.mean((pred_y - target_var)**2)           
    return pred_y,loss

def prob(hidden,y,hyper_params,params):
    loss=T.mean(T.nnet.categorical_crossentropy(hidden,y))
    return hidden,loss

def init_variables(mask_var=False):
    nn_vars={}
    nn_vars['in_var'] = T.ltensor3('in_var')
    nn_vars['target_var'] = T.ltensor3('target_var')
    if(mask_var):
        nn_vars['mask_var']= T.lmatrix('mask_var')
    return nn_vars	

def init_hidden_value(hyper_params,var_name='hidden_dim'):
    return T.zeros((hyper_params[var_name],),dtype=float)

def default_params():
    return {'n_cats':3,'seq_dim':2,'hidden_dim':3,
            'learning_rate':0.1,'momentum':0.9}