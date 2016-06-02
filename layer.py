import theano, theano.tensor as T
import numpy as np

class Layer(object):

    def __init__(self, input_size, hidden_size, activation=T.nnet.sigmoid):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.activation  = activation
        self.create_variables()

    def create_variables(self):
        self.linear_matrix = tools.create_shared(self.hidden_size, self.input_size, name="Layer.linear_matrix")
        self.bias_matrix   = tools.create_shared(self.hidden_size, name="Layer.bias_matrix")

    def activate(self, x):

        if x.ndim > 1:
            return self.activation(
                T.dot(self.linear_matrix, x.T) + self.bias_matrix[:,None] ).T
        else:
            return self.activation(
                T.dot(self.linear_matrix, x) + self.bias_matrix )