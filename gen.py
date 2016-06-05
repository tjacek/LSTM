import numpy as np 

def make_abc(dataset_size=100,seq_size=20):
    cats=[gen_abc,rand_gen]
    return make_dataset(cats,dataset_size,seq_size)

def make_dataset(cats,dataset_size,seq_size):
    dataset=[]
    for i,cat_i in enumerate(cats):
        dataset+=gen_data(cat_i,cat=i,size=dataset_size,max_size=seq_size)
    return dataset

def gen_data(make_seq,cat=0,size=100,max_size=10):
    sizes=[np.random.randint(3,max_size)
               for i in range(size)] 
    return [ (cat,make_seq(size_i)) 
                 for size_i in sizes] 

def gen_abc(rn_size):
    abc=[]
    for i in range(3):
        abc+=[i for j in range(rn_size)]
    return to_vectors(abc)

def rand_gen(max_size):
    size_i=np.random.randint(3,3*max_size)
    rand_seq=[np.random.randint(0,3) 
                for i in range(size_i)]
    return to_vectors(rand_seq)

def to_vectors(seq):
    size=max(seq)+1
    vectors=[ind_vector(seq_i,size)
               for seq_i in seq]
    return np.array(vectors)

def ind_vector(k,size):
    vec=np.zeros((size,),dtype=float)
    vec[k]=1.0
    return vec
#print(make_abc())