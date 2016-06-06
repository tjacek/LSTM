import theano
import theano.tensor as T
import numpy as np 
import tools

class RNN(object):
    def __init__(self,hyper_params,
    	              in_var,target_var,
    	              pred,loss,updates):
        self.pred=theano.function([in_var], pred,allow_input_downcast=True)        
        self.loss=theano.function([in_var,target_var], loss,allow_input_downcast=True)
        self.updates=theano.function([in_var, target_var], loss, 
                               updates=updates,allow_input_downcast=True)

def build_rnn(hyper_params):
    U, V, W = init_params(hyper_params)
    in_var,target_var=init_variables()
    [o,s], updates = theano.scan(
            forward_step,
            sequences=in_var,
            outputs_info=[None, dict(initial=init_hidden_value(hyper_params))],
            non_sequences=[U, V, W],
            #truncate_gradient=self.bptt_truncate,
            strict=True)

    prediction = T.argmax(o, axis=1)
    loss = T.sum(T.nnet.categorical_crossentropy(o, target_var))
    updates=simple_sgd(loss,hyper_params,[U,V,W])
    return RNN(hyper_params,in_var,target_var,prediction,loss,updates)

def init_params(hyper_params):
    seq_dim=hyper_params['seq_dim']
    hidden_dim=hyper_params['hidden_dim']
    U=tools.create_shared(out_size=hidden_dim, in_size=seq_dim, name='U')
    V=tools.create_shared(out_size=seq_dim, in_size=hidden_dim, name='V')
    W=tools.create_shared(out_size=hidden_dim, in_size=hidden_dim, name='W')
    return U,V,W

def init_variables():
    in_var = T.ivector('in_var')
    target_var = T.ivector('target_var')
    return in_var,target_var	

def init_hidden_value(hyper_params):
    return T.zeros(hyper_params['hidden_dim'])

def forward_step(x_t, s_t_prev, U, V, W):
    s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
    o_t = T.nnet.softmax(V.dot(s_t))
    return [o_t[0], s_t]

def simple_sgd(loss,hyper_params,params):
    learning_rate=hyper_params['learning_rate']
    diff=[ T.grad(loss, param_i) for param_i in params]
    updates=[(param_i, param_i - learning_rate * diff_i)
                for param_i,diff_i in zip(params,diff)]
    return updates

def default_params():
    return {'seq_dim':3,'hidden_dim':3,'learning_rate':0.01}

build_rnn(default_params())