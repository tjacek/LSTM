import theano, theano.tensor as T
import numpy as np

class LSTM(object):
	def __init__(self,in_gate,in_gate2,forget_gate,out_gate):
		self.in_gate=in_gate
		self.in_gate2=in_gate
        self.forget_gate=forget_gate
        self.out_gate=out_gate