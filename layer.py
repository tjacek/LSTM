import theano, theano.tensor as T
import numpy as np
import tools

class Layer(object):

    def __init__(self,W,b, activation=T.nnet.sigmoid):
        self.w=W
        self.b=b
        self.activation  = activation

    def linear(self,x):
        return T.dot(x,self.w.T)+self.b

    def params(self):
        return [self.w,self.b]

    def non_linear(self,x):
        return self.activation(self.linear(x))

class Gate(object):
    def __init__(self,w,u,b):
        self.w = w
        self.u = u
        self.b =b

    def linear(self,x,h):
        return T.dot(x,self.w.T) + T.dot(h,self.u.T) +self.b

    def params(self):
        return [self.w,self.u,self.b]

def create_layer(out_size,in_size,name="layer"):
    linear_matrix = tools.create_shared(out_size, in_size, name="W_"+name)
    bias_matrix   = tools.create_shared(out_size, name="b_"+name)
    hyper_params={'hidden_size':out_size,'input_size':in_size}
    return Layer(hyper_params,linear_matrix,bias_matrix)

def create_gate(out_size,x_size,h_size,name="layer"):
    x_matrix = tools.create_shared(out_size, x_size, name="W_"+name)
    h_matrix = tools.create_shared(out_size, h_size, name="U_"+name)
    bias_matrix   = tools.create_shared(out_size, name="b_"+name)
    #hyper_params={'hidden_size':out_size,'input_size':in_size}
    return Gate(x_matrix,h_matrix,bias_matrix)

def create_softmax(hyper_params):
    hidden_size=hyper_params['hidden_dim']
    n_cats=hyper_params['n_cats']
    W=tools.create_shared(out_size=n_cats,in_size=hidden_size,name="W_softmax")
    b=tools.create_shared(out_size=n_cats ,name="b_softmax")
    return Layer(W,b, activation=T.nnet.softmax)

def get_params(layers):
    params=[]
    for layer_i in layers:
        params+=layer_i.params()
    return params    