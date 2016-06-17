import theano
import theano.tensor as T
import numpy as np 
import tools,optim,layer
import lstm

class RNN(object):
    def __init__(self,hyper_params,
    	              nn_vars,
    	              pred,loss,updates):
        in_var=nn_vars['in_var']
        target_var=nn_vars['target_var']
        self.pred=theano.function([in_var], pred,allow_input_downcast=True,on_unused_input='warn')        
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)

class MaskRNN(object):
    def __init__(self,hyper_params,
                      in_var,target_var,mask_var,
                      pred,loss,updates):
        self.pred=theano.function([in_var,mask_var], pred,allow_input_downcast=True,on_unused_input='warn')        
        self.loss=theano.function([in_var,target_var,mask_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var,mask_var], loss, 
                               updates=updates,allow_input_downcast=True)

def build_rnn(hyper_params):
    nn_vars=init_variables(True)
    #s,params=lstm.mask_lstm(hyper_params)
    builder=lstm.LSTMBuilder(hyper_params)
    s=builder.get_output(nn_vars)
    params=builder.get_params()
    pred_y= T.mean(T.mean(s,axis=0),axis=1)
    target_var=nn_vars['target_var']
    loss=T.mean((pred_y - target_var)**2)#)
    updates= optim.momentum_sgd(loss,hyper_params,params)
    return RNN(hyper_params,nn_vars,pred_y,loss,updates)

def init_variables(mask_var=False):
    nn_vars={}
    nn_vars['in_var'] = T.ltensor3('in_var')
    nn_vars['target_var'] = T.lvector('target_var')
    if(mask_var):
        nn_vars['mask_var']= T.ltensor3('mask_var')
    return nn_vars	

def init_hidden_value(hyper_params,var_name='hidden_dim'):
    return T.zeros((hyper_params[var_name],),dtype=float)

def default_params():
    return {'n_cats':3,'seq_dim':2,'hidden_dim':3,
            'learning_rate':0.1,'momentum':0.9}