import numpy as np
import rnn,gen,tools

def test():
    dataset=gen.make_abc(dataset_size=21,seq_size=10)
    y=np.array(get_dim(dataset,dim=0))
    #X=np.array(get_dim(dataset,dim=1))
    X=get_dim(dataset,dim=1)
    X,mask=tools.masked_dataset(X)
    print(mask.shape)
    print("dataset created")
    rnn_model=rnn.build_rnn(rnn.default_params())
    print("model created")
    train_super(X,y,mask,rnn_model)
    print("train")
    #check_model(X,y,rnn_model)

def get_dim(dataset,dim=0):
    return [pair_i[dim] for pair_i in dataset]

def train_super(X,y,mask,model,epochs=1):
    x_batches=tools.get_batches(X)
    y_batches=tools.get_batches(y)
    mask_batches=tools.get_batches(mask)
    for i in range(epochs):
        #for x_i,y_i in zip(x_batches,y_batches):
        for i,y_i in enumerate(y_batches):     
            x_i=x_batches[i]
            xt_i=tools.time_first(x_i)
            mask_i=mask_batches[i]
            maskt_i=tools.time_first(mask_i)
            print("#################")
            print(xt_i.shape)
            print(y_i.shape)
            print(maskt_i.shape)  
            value=model.pred(xt_i,maskt_i)          
            print(value.shape)#,maskt_i))
            print(value)
            #print(model.loss(xt_i,y_i))
            #print(model.updates(xt_i,y_i))
    return model

def check_model(X,y,model):
    x_batches=tools.get_batches(X)
    y_batches=tools.get_batches(y)
    for i,y_i in enumerate(y_batches):
        xt_i=tools.time_first(x_batches[i])
        y_pred=model.pred(xt_i)
        print(y_i==y_pred) 

test()