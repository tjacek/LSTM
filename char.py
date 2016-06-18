# -*- coding: utf-8 -*-
import numpy as np 

ALPHA=u' abcdefghijklmnoprstuwyząęćńłśóżź'

def word_to_array(word):
    y=txt_to_int(word)
    vectors=[index_to_vector(y_i) 
               for y_i in y]
    x=np.array(vectors)
    return x,y

def txt_to_int(txt):
    return [ ALPHA.index(token_i)
                for token_i in txt]

def index_to_vector(i):
    max_i=len(ALPHA)
    vec_i=np.zeros((max_i,),dtype=float)
    vec_i[i]=1.0
    return vec_i

print(word_to_array(u'żółć'))