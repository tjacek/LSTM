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
                      nn_vars,
                      pred,loss,updates):
        in_var=nn_vars['in_var']
        target_var=nn_vars['target_var']
        mask_var=nn_vars['mask_var']
        self.pred=theano.function([in_var,mask_var], pred,allow_input_downcast=True,on_unused_input='warn')        
        self.loss=theano.function([in_var,target_var,mask_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var,mask_var], loss, 
                               updates=updates,allow_input_downcast=True)

class TestRNN(object):
    def __init__(self, hyper_params,nn_vars,pred,loss,updates):
        in_var=nn_vars['in_var']
        target_var=nn_vars['target_var']

        mask_var=nn_vars['mask_var']
        self.pred=theano.function([in_var,mask_var,target_var], pred,allow_input_downcast=True,on_unused_input='warn')     

def build_rnn(hyper_params):
    nn_vars=init_variables(True)
    builder=lstm.MaskLSTMBuilder(hyper_params,True)
    hidden=builder.get_output(nn_vars)
    params=builder.get_params()
    #pred_y,loss=regresion(hidden,nn_vars['target_var'])
    pred_y,loss=prob(hidden,nn_vars['target_var'],hyper_params,params)
    updates=optim.momentum_sgd(loss,hyper_params,params)
    return MaskRNN(hyper_params,nn_vars,pred_y,loss,updates)

def regresion(hidden,target_var):
    pred_y= T.mean(T.mean(hidden,axis=0),axis=1)
    loss=T.mean((pred_y - target_var)**2)#)            masked_hid = T.switch(mask_t, h_t, h_t_next)
            #masked_out = T.switch(mask_t, h_t, h_t_next)

    return pred_y,loss

def prob(hidden,y,hyper_params,params):
    p_x=hidden
    y_hot=T.extra_ops.to_one_hot(y,p_x.shape[2])
    y_full=T.tile(y_hot, (p_x.shape[0],1,1) )
    loss=T.sum(T.nnet.categorical_crossentropy( p_x,y_full))#,loss,updates
    return p_x,loss

def init_variables(mask_var=False):
    nn_vars={}
    nn_vars['in_var'] = T.ltensor3('in_var')
    nn_vars['target_var'] = T.ivector('target_var')
    if(mask_var):
        nn_vars['mask_var']= T.lmatrix('mask_var')
    return nn_vars	

def init_hidden_value(hyper_params,var_name='hidden_dim'):
    return T.zeros((hyper_params[var_name],),dtype=float)

def default_params():
    return {'n_cats':3,'seq_dim':2,'hidden_dim':3,
            'learning_rate':0.1,'momentum':0.9}