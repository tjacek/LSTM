import theano, theano.tensor as T
import numpy as np

srng = theano.tensor.shared_randomstreams.RandomStreams(1234)
np_rng = np.random.RandomState(1234)

def create_shared(out_size, in_size=None, name=None):
    if in_size is None:
        return theano.shared(value=random_initialization((out_size, )), name=name)
    else:
        return theano.shared(value=random_initialization((out_size, in_size)), name=name)

def random_initialization(size):
    return (np_rng.standard_normal(size) * 1. / size[0]).astype(theano.config.floatX)

def get_batches(x,batch_size=5):
    n_batches=int(np.ceil(float(x.shape[0]) / float(batch_size)))
    return [x[i*batch_size:(i+1)*batch_size] 
               for i in range(n_batches)]
    