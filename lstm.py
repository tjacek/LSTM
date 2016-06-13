import theano, theano.tensor as T
import numpy as np
import tools

def mask_lstm(hyper_params,in_var,mask_var):
    #U, V, W = init_params(hyper_params)
    def recurrence(x_t,h_state,cell_state):
        #h_t = T.nnet.sigmoid( U.linear(x_t) + V.linear(h_state) )
        #s_t = T.nnet.softmax(W.linear(h_t))
        return [h_t[0],s_t]

    [h,s], updates = theano.scan(
            recurrence,
            sequences=in_var,
            outputs_info=[init_hidden_value(hyper_params),,None],
            n_steps=in_var.shape[0])
    final= mask_var*s#T.dot(mask_var,s)
    params=layer.get_params([U,V,W])
    return final,params