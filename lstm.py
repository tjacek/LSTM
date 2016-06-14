import theano, theano.tensor as T
import numpy as np
import tools

def mask_lstm(hyper_params,in_var,mask_var):
    #U, V, W = init_params(hyper_params)
    forget_gate,in_gate,cell_gate,out_gate=lstm_params(hyper_params)
    def recurrence(x_t1,h_t,c_t):
        xh_t=T.concatenate([x_t1,h_t])
        f_t=T.nnet.sigmoid(forget_gate.linear(xh_t))
        i_t=T.nnet.sigmoid(in_gate.linear(xh_t))
        c_t_prop=T.nnet.tanh(cell_gate.linear(xh_t))
        c_t_next=f_t*c_t+i_t*c_t_prop
        o_t=T.nnet.sigmoid(out_gate.linear(xh_t))
        h_t_next=o_t*T.nnet.tanh(c_t_next)
        return [h_t_next,c_t_next]

    [h,s], updates = theano.scan(
            recurrence,
            sequences=in_var,
            outputs_info=[init_hidden_value(hyper_params),,None],
            n_steps=in_var.shape[0])
    final= mask_var*s#T.dot(mask_var,s)
    params=layer.get_params([U,V,W])
    return final,params