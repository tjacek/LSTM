import rnn,gen

def test():
    dataset=gen.make_abc(dataset_size=200,seq_size=20)
    y=get_dim(dataset,dim=0)
    X=get_dim(dataset,dim=1)
    print("dataset created")
    rnn_model=rnn.build_rnn(rnn.default_params())
    print("model created")
    train_super(X,y,rnn_model)

def get_dim(dataset,dim=0):
    return [pair_i[dim] for pair_i in dataset]

def train_super(X,y,model):
    for i,y_i in enumerate(y):
        x_i=X[i]
        print(x_i[0].shape)
        model.updates(x_i,y_i)

test()