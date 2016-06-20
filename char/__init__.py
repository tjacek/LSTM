# -*- coding: utf-8 -*-
import numpy as np 
import codecs
ALPHA=u'abcdefghijklmnoprstuwyząęćńłśóżź'

def word_to_array(y):
    vectors=[index_to_vector(y_i) 
               for y_i in y]
    #x=np.array(vectors)
    return vectors

def txt_to_int(txt):
    return [ ALPHA.index(token_i)
                for token_i in txt]

def index_to_vector(i):
    max_i=len(ALPHA)
    vec_i=np.zeros((max_i,),dtype=float)
    vec_i[i]=1.0
    return vec_i

def read_dataset(filename):
    f = codecs.open(filename,'r','utf8')
    txt=f.read()
    lines=txt.split('\n')
    y=[txt_to_int(line_i)
        for line_i in lines]
    x=[word_to_array(y_i) 
               for y_i in y]
    #print(get_lengths(y))
    #print(get_lengths(x))
    for y_i in y:
        del y_i[0]
    for x_i in x:
        del x_i[-1]    
    #print(get_lengths(y))
    #print(get_lengths(x))  
    return 0#x#dataset

def get_lengths(y):
    return [len(y_i) for y_i in y]

print(read_dataset(u'data.txt'))