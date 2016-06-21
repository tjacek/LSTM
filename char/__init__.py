# -*- coding: utf-8 -*-
import numpy as np 
import codecs
import pap

ALPHA=u' abcdefghijklmnoprstuwyzvxqąęćńłśóżź'

def make_pap_dataset(filename='pap.txt'):
    notes=pap.read_notes(filename)
    samples=pap.select_samples(notes)
    x,y=lines_to_next(samples)
    print(x.shape)
    print(y.shape)
    return x,y

def lines_to_next(lines): 
    pairs=[ next_char(line_i) for line_i in lines]
    x_raw=[pair_i[0] for pair_i in pairs]
    y_raw=[pair_i[1] for pair_i in pairs]
    x=np.array(x_raw)
    y=np.array(y_raw)
    return x,y

def next_char(word):
    y=txt_to_int(word)
    x=word_to_array(y)
    del y[0]
    del x[-1]
    return np.array(x),np.array(y)

def word_to_array(y):
    vectors=[index_to_vector(y_i) 
               for y_i in y]
    return vectors

def txt_to_int(txt):
    for token_i in txt:
        if(not (token_i in ALPHA)):
            print(token_i)
    return [ ALPHA.index(token_i)
                for token_i in txt]

def index_to_vector(i):
    max_i=len(ALPHA)
    vec_i=np.zeros((max_i,),dtype=float)
    vec_i[i]=1.0
    return vec_i

make_pap_dataset()
