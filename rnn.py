import theano
import theano.tensor as T
import numpy as np 
import tools,layer

class RNN(object):
    def __init__(self,hyper_params,
    	              in_var,target_var,
    	              pred,loss,updates):
        self.pred=theano.function([in_var], pred,allow_input_downcast=False,on_unused_input='warn')        
        #self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        #self.updates=theano.function([in_var, target_var], loss, 
        #                       updates=updates,allow_input_downcast=True)

def build_rnn(hyper_params):
    U, V, W = init_params(hyper_params)
    in_var,target_var=init_variables()
    
    def recurrence(in_var,h_state,U,V,W):
        x_t=in_var
        #h_t = T.nnet.sigmoid(T.dot(x_t, U) + T.dot(h_tm1, V) )
        #s_t = T.nnet.softmax(T.dot(h_t, W))
        return [x_t,h_state]#[x_t,h_tm1]#[h_t, s_t]

    [o,s], updates = theano.scan(
            recurrence,
            sequences=in_var,
            outputs_info=[None,init_hidden_value(hyper_params)],
            #outputs_info=[None, dict(initial=init_hidden_value(hyper_params))],
            non_sequences=[U, V, W],
            #truncate_gradient=self.bptt_truncate,
            n_steps=in_var.shape[0])

    prediction = T.argmax(o)#, axis=1)
    loss = None#T.sum(T.nnet.categorical_crossentropy(o, target_var))
    updates=None#simple_sgd(loss,hyper_params,[U,V,W])
    return RNN(hyper_params,in_var,target_var,prediction,loss,updates)

def init_params(hyper_params):
    seq_dim=hyper_params['seq_dim']
    hidden_dim=hyper_params['hidden_dim']
    n_cats=hyper_params['n_cats']
    U=tools.create_shared(out_size=hidden_dim, in_size=seq_dim, name='U')
    #layer.create_layer(hidden_dim,seq_dim,layer_name="U")
    V=tools.create_shared(out_size=hidden_dim, in_size=hidden_dim, name='V')
    #layer.create_layer(seq_dim,seq_dim,layer_name="V")
    W=tools.create_shared(out_size=n_cats, in_size=hidden_dim, name='W')
    #layer.create_layer(hidden_dim,hidden_dim,layer_name="W")
    return U,V,W

def init_variables():
    in_var = T.lvector('in_var')
    target_var = T.lvector('target_var')
    return in_var,target_var	

def init_hidden_value(hyper_params,var_name='hidden_dim'):
    return T.zeros((hyper_params[var_name],1),dtype=float)

#def recurrence(x_t, h_tm1)(x_t, s_t_prev, U, V, W):
    #s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
    #o_t = T.nnet.softmax(V.dot(s_t))
#    return [x_t,s_t_prev]#[o_t[0], s_t]

def simple_sgd(loss,hyper_params,params):
    learning_rate=hyper_params['learning_rate']
    diff=[ T.grad(loss, param_i) for param_i in params]
    updates=[(param_i, param_i - learning_rate * diff_i)
                for param_i,diff_i in zip(params,diff)]
    return updates

def default_params():
    return {'n_cats':2,'seq_dim':3,'hidden_dim':3,'learning_rate':0.01}