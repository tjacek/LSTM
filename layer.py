import theano, theano.tensor as T
import numpy as np
import tools

class Layer(object):

    def __init__(self, hyper_params,W,b, activation=T.nnet.sigmoid):
        self.hyper_params = hyper_params
        self.w=W
        self.b=b
        self.activation  = activation

    def linear(self,x):
        return T.dot(x,self.w)+self.b

    def params(self):
        return [self.w,self.b]

    def activate(self, x):
        if x.ndim > 1:
            return self.activation(
                T.dot(self.linear_matrix, x.T) + self.bias_matrix[:,None] ).T
        else:
            return self.activation(
                T.dot(self.linear_matrix, x) + self.bias_matrix )

class Gate(object):
    def __init__(self,w,u,b):
        self.w = w
        self.u = u
        self.b =b

    def linear(self,x,h):
        return T.dot(x,self.w) + T.dot(h,self.u) +self.b

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

def get_params(layers):
    params=[]
    for layer_i in layers:
        params+=layer_i.params()
    return params    