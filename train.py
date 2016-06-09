import numpy as np
import rnn,gen,tools

def test():
    dataset=gen.make_abc(dataset_size=21,seq_size=3)
    y=np.array(get_dim(dataset,dim=0))
    X=np.array(get_dim(dataset,dim=1))
    print("dataset created")
    rnn_model=rnn.build_rnn(rnn.default_params())
    print("model created")
    train_super(X,y,rnn_model)
    print("train")
    check_model(X,y,rnn_model)

def get_dim(dataset,dim=0):
    return [pair_i[dim] for pair_i in dataset]

def train_super(X,y,model,epochs=100):
    x_batches=tools.get_batches(X)
    y_batches=tools.get_batches(y)
    for i in range(epochs):
        for x_i,y_i in zip(x_batches,y_batches):
            xt_i=tools.time_first(x_i)
            print(xt_i.shape)
            #print(model.pred(xt_i))
            print(model.loss(xt_i,y_i))
            print(model.updates(xt_i,y_i))
    return model

def check_model(X,y,model):
    x_batches=tools.get_batches(X)
    y_batches=tools.get_batches(y)
    for i,y_i in enumerate(y_batches):
        xt_i=tools.time_first(x_batches[i])
        y_pred=model.pred(xt_i)
        print(y_i==y_pred) 

test()