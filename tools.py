import theano, theano.tensor as T
import numpy as np

srng = theano.tensor.shared_randomstreams.RandomStreams(1234)
np_rng = np.random.RandomState(1234)

#def dir_of_params(params):
#    return dict([p ])

def create_shared(out_size, in_size=None, name=None,orth=True):
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

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def time_first(X):
    return np.transpose(X, (1, 0, 2))

def masked_dataset(seqs):
    lengths=[len(seq_i) for seq_i in seqs]
    max_len=max(lengths)
    def rescale_seq(seq_i):
        new_shape=list(seq_i.shape)
        new_shape[0]=max_len
        new_seq_i=np.zeros(tuple(new_shape),dtype=seq_i.dtype)
        new_seq_i[:seq_i.shape[0]]=seq_i
        return new_seq_i
    new_seqs=[rescale_seq(seq_i)
                for seq_i in seqs]
                
    mask=[ [1]*len_i + [0]*(max_len-len_i)
           for len_i,seq_i in zip(lengths,seqs)]
    return np.array(new_seqs),np.array(mask)
   