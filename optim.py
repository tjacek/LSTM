import theano, theano.tensor as T
import numpy as np

def basic_sgd(loss,hyper_params,params):
    learning_rate=hyper_params['learning_rate']
    diff=[ T.grad(loss, param_i) for param_i in params]
    updates=[(param_i, param_i - learning_rate * diff_i)
                for param_i,diff_i in zip(params,diff)]
    return updates

def momentum_sgd(loss,hyper_params,params):
    momentum=hyper_params['momentum']
    updates=basic_sgd(loss,hyper_params,params)
    new_updates=[]
    for param_i,update_i in updates:
        value = param_i.get_value(borrow=True)
        velocity_i = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param_i.broadcastable)
        x = momentum * velocity_i + update_i
        new_updates.append((velocity_i,x - param_i))
        new_updates.append((param_i,x))
    return new_updates