import theano, theano.tensor as T
import numpy as np
import tools

class LSTM(object):
    def __init__(self,hyper_params,in_gate,in_gate2,forget_gate,out_gate):
        self.hyper_params=hyper_params
        self.hidden_size=self.hidden_size['hidden_size']
        self.in_gate=in_gate
        self.in_gate2=in_gate
        self.forget_gate=forget_gate
        self.out_gate=out_gate

    def activate(self, x, h):
        if h.ndim > 1:
            prev_c = h[:, :self.hidden_size]
            prev_h = h[:, self.hidden_size:]
        else:
            prev_c = h[:self.hidden_size]
            prev_h = h[self.hidden_size:]

        if h.ndim > 1:
            obs = T.concatenate([x, prev_h], axis=1)
        else:
            obs = T.concatenate([x, prev_h])

        in_gate = self.in_gate.activate(obs)
        forget_gate = self.forget_gate.activate(obs)
        in_gate2 = self.in_gate2.activate(obs)
        next_c = forget_gate * prev_c + in_gate2 * in_gate
        out_gate = self.out_gate.activate(obs)
        next_h = out_gate * T.tanh(next_c)
        
        if h.ndim > 1:
            return T.concatenate([next_c, next_h], axis=1)
        else:
            return T.concatenate([next_c, next_h])

def make_lstm(input_size,hidden_size):
    in_gate     = tools.Layer(input_size + hidden_size, hidden_size, T.nnet.sigmoid )
    forget_gate = tools.Layer(input_size + hidden_size, hidden_size, T.nnet.sigmoid )
    in_gate2    = tools.Layer(input_size + hidden_size, hidden_size, T.nnet.sigmoid)
    out_gate    = tools.Layer(input_size + hidden_size, hidden_size, T.nnet.sigmoid)
    hyper_params={'input_size':input_size,'hidden_size':hidden_size}
    return LSTM(hyper_params,in_gate,in_gate2,forget_gate,out_gate)