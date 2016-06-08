import theano
import theano.tensor as T
import numpy as np 
import tools,layer

class RNN(object):
    def __init__(self,hyper_params,
    	              in_var,target_var,
    	              pred,loss,updates):
        self.pred=theano.function([in_var], pred,allow_input_downcast=True,on_unused_input='warn')        
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)

def build_rnn(hyper_params):
    U, V, W = init_params(hyper_params)
    in_var,target_var=init_variables()
    
    def recurrence(x_t,h_state,U,V,W):
        h_t = T.nnet.sigmoid(T.dot(x_t, U) + T.dot(h_state, V) )
        s_t = T.nnet.softmax(T.dot(h_t, W))
        return [h_t[0],s_t]#[s_t,h_t[0]]

    [h,s], updates = theano.scan(
            recurrence,
            sequences=in_var,
            outputs_info=[init_hidden_value(hyper_params),None],
            non_sequences=[U, V, W],
            n_steps=in_var.shape[0])
    
    p_x= T.sum(s,axis=0)
    prediction =  T.argmax(p_x,axis=1)
    #loss = None
    loss=T.mean(T.nnet.categorical_crossentropy(p_x, target_var))#)
    updates=simple_sgd(loss,hyper_params,[U,V,W])
    return RNN(hyper_params,in_var,target_var,prediction,loss,updates)

def init_params(hyper_params):
    seq_dim=hyper_params['seq_dim']
    hidden_dim=hyper_params['hidden_dim']
    n_cats=hyper_params['n_cats']
    U=tools.create_shared(out_size=hidden_dim, in_size=seq_dim, name='U')
    V=tools.create_shared(out_size=hidden_dim, in_size=hidden_dim, name='V')
    W=tools.create_shared(out_size=n_cats, in_size=hidden_dim, name='W')
    return U,V,W

def init_variables():
    in_var = T.ltensor3('in_var')
    #in_var=T.lvector('in_var')
    target_var = T.lvector('target_var')
    return in_var,target_var	

def init_hidden_value(hyper_params,var_name='hidden_dim'):
    return T.zeros((hyper_params[var_name],),dtype=float)

def simple_sgd(loss,hyper_params,params):
    learning_rate=hyper_params['learning_rate']
    diff=[ T.grad(loss, param_i) for param_i in params]
    updates=[(param_i, param_i - learning_rate * diff_i)
                for param_i,diff_i in zip(params,diff)]
    return updates

def default_params():
    return {'n_cats':3,'seq_dim':3,'hidden_dim':3,'learning_rate':0.001}