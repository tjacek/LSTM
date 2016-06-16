import numpy as np 

def make_abc(dataset_size=100,seq_size=20):
    cats=[gen_abc,rand_gen]
    return make_dataset(cats,dataset_size,seq_size)

def make_dataset(cats,dataset_size,seq_size):
    dataset=[]
    for i,cat_i in enumerate(cats):
        dataset+=gen_uneven(cat_i,cat=i,size=dataset_size,max_size=seq_size)
    return dataset

def gen_fixed(make_seq,cat=0,size=100,max_size=10):
    return [ (cat,make_seq(max_size)) 
                 for size_i in range(size)]

def gen_uneven(make_seq,cat=0,size=100,max_size=20):
    sizes=[np.random.randint(3,max_size)
               for i in range(size)] 
    return [ (cat,make_seq(3*size_i)) 
                 for size_i in sizes] 

def gen_abc(rn_size):
    abc=[]
    for i in range(3):
        abc+=[i for j in range(rn_size)]
    return to_vectors(abc)

def rand_gen(max_size):
    max_size*=3
    rand_seq=[np.random.randint(0,3) 
                for i in range(max_size)]
    return to_vectors(rand_seq)

def to_vectors(seq,size=3):
    vectors=[ind_vector(seq_i,size)
               for seq_i in seq]
    return np.array(vectors)

def ind_vector(k,size):
    vec=np.zeros((size,),dtype=float)
    vec[k]=1.0
    return vec