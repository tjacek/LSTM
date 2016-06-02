import theano, theano.tensor as T
import numpy as np

srng = theano.tensor.shared_randomstreams.RandomStreams(1234)
np_rng = np.random.RandomState(1234)

def create_shared(out_size, in_size=None, name=None):
    if in_size is None:
        return theano.shared(random_initialization((out_size, )), name=name)
    else:
        return theano.shared(random_initialization((out_size, in_size)), name=name)

def random_initialization(size):
    return (np_rng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)