import numpy as np    

def gen_dataset(size,max_length,min_length=3,reg=True):
    X = np.concatenate([np.random.uniform(size=(size,max_length, 1)),
                        np.zeros((size,max_length,1 ))],
                       axis=-1)
    y = np.zeros((size,),dtype=float)
    mask = np.zeros((size, max_length))
    for i in range(size):
        seq_i_length = np.random.randint(min_length, max_length)
        mask[i, :seq_i_length] = 1
        X[i, seq_i_length:, 0] = 0
        X[i, np.random.randint(0,seq_i_length/2), 1] = 1
        X[i, np.random.randint(seq_i_length/2,seq_i_length), 1] = 1
        y[i] = np.sum(X[i, :, 0]*X[i, :, 1])
    if(reg):
        X -= X.reshape(-1, 2).mean(axis=0)
        y -= y.mean()      
    return X,y,mask
