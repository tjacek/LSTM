import theano
import theano.tensor as T
import numpy as np 
import tools,optim,layer
import lstm

class RNN(object):
    def __init__(self,hyper_params,
    	              in_var,target_var,
    	              pred,loss,updates):
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
    in_var,target_var,mask_var=init_variables(True)
    s,params=lstm.mask_lstm(hyper_params,in_var,mask_var)
    pred_y= T.mean(T.mean(s,axis=0),axis=1)
    loss=T.mean((pred_y - target_var)**2)#)
    updates= optim.momentum_sgd(loss,hyper_params,params)
    return RNN(hyper_params,in_var,target_var,pred_y,loss,updates)

#def build_rnn(hyper_params):
#    in_var,target_var,mask_var=init_variables(True)
#    s,params=lstm.mask_lstm(hyper_params,in_var,mask_var)
#    p_x= T.mean(s,axis=0)
#    prediction = T.argmax(p_x,axis=1)
#    loss=T.mean(T.nnet.categorical_crossentropy(p_x, target_var))#)
#    updates=optim.ada_grad(loss,hyper_params,params)
#    return MaskRNN(hyper_params,in_var,target_var,mask_var,prediction,loss,updates)

def simple_rnn(hyper_params,in_var):
    U, V, W = init_params(hyper_params)
    def recurrence(x_t,h_state):
        h_t = T.nnet.sigmoid( U.linear(x_t) + V.linear(h_state) )
        s_t = T.nnet.softmax(W.linear(h_t))
        return [h_t[0],s_t]

    [h,s], updates = theano.scan(
            recurrence,
            sequences=in_var,
            outputs_info=[init_hidden_value(hyper_params),None],
            n_steps=in_var.shape[0])
    params=layer.get_params([U,V,W])
    return s,params

def mask_rnn(hyper_params,in_var,mask_var):
    U, V, W = init_params(hyper_params)
    def recurrence(x_t,h_state):
        h_t = T.nnet.sigmoid( U.linear(x_t) + V.linear(h_state) )
        s_t = T.nnet.softmax(W.linear(h_t))
        return [h_t[0],s_t]

    [h,s], updates = theano.scan(
            recurrence,
            sequences=in_var,
            outputs_info=[init_hidden_value(hyper_params),None],
            n_steps=in_var.shape[0])
    final= mask_var*s#T.dot(mask_var,s)
    params=layer.get_params([U,V,W])
    return final,params

def init_params(hyper_params):
    seq_dim=hyper_params['seq_dim']
    hidden_dim=hyper_params['hidden_dim']
    n_cats=hyper_params['n_cats']
    U=layer.create_layer(out_size=hidden_dim, in_size=seq_dim, name='U')
    V=layer.create_layer(out_size=hidden_dim, in_size=hidden_dim, name='V')
    W=layer.create_layer(out_size=n_cats, in_size=hidden_dim, name='W')
    return U,V,W

def init_variables(mask_var=False):
    in_var = T.ltensor3('in_var')
    target_var = T.lvector('target_var')
    if(mask_var):
        mask_var= T.ltensor3('mask_var')#T.lmatrix('mask_var')
        return in_var,target_var,mask_var
    return in_var,target_var	

def init_hidden_value(hyper_params,var_name='hidden_dim'):
    return T.zeros((hyper_params[var_name],),dtype=float)

def default_params():
    return {'n_cats':3,'seq_dim':2,'hidden_dim':3,
            'learning_rate':0.1,'momentum':0.9}