import numpy as np
import rnn,gen,tools

def test():
    dataset=gen.make_abc(dataset_size=21,seq_size=3)
    y=get_dim(dataset,dim=0)
    X=np.array(get_dim(dataset,dim=1))
    print("dataset created")
    rnn_model=rnn.build_rnn(rnn.default_params())
    print("model created")
    train_super(X,y,rnn_model)

def get_dim(dataset,dim=0):
    return [pair_i[dim] for pair_i in dataset]

def train_super(X,y,model):
    print(X.shape)
    x_batches=[x_i.shape for x_i in tools.get_batches(X)]
    #print(len(x_batches))
    print(x_batches)#x_batches[0].shape)
    #for i,y_i in enumerate(y):
    #    x_i=X[i]
    #    print(x_i.dtype)
    #    print(x_i.shape)
    #    print(x_i)
    #    print(model.pred(x_i))#,y_i)

test()