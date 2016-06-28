import numpy as np 

def gen_seq(size,model,chars=None):
    if(chars==None):
        chars=[char_as_vec(2)]
    for i in range(size):
        seq_i=np.array(chars)
        next_char=gen_next(seq_i,model)
        chars.append(next_char)  
    return np.array(chars)  

def gen_next(char_prev,model):
    #char_prev=char_as_vec(k)
    prob=model.pred(char_prev)[-1]
    if(is_space(char_prev)):     
        k_next=random_char(prob)
    else:
        k_next=determ_next_char(prob)
    return char_as_vec(k_next)

def determ_next_char(prob,n_cats=36):
    prob=prob.flatten()
    return np.argmax(prob)

def sample_next_char(prob,n_cats=36):
    prob=prob.flatten()
    return np.random.choice(n_cats, replace=True, p=prob)

def random_char(prob,n_cats=20):
    return np.random.randint(n_cats)+1

def char_as_vec(k,n_cats=36):
    #char_shape[1]=n_cats
    seed=np.zeros((1,n_cats),dtype=float)
    seed[0][k]=1.0
    return seed

def is_space(char_prev):
    last_char=char_prev[-1].flatten()
    return last_char[0]==1.0