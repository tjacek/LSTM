# -*- coding: utf-8 -*-
import codecs,re
import numpy as np    

def read_notes(filename,min_size=120):
    txt = codecs.open(filename,'r','utf8')
    txt=txt.read()
    notes=re.split(u'#[0-9]+',txt)
    clean_notes=[ clean(note_i)
                  for note_i in notes]
    long_notes=[ note_i for note_i in clean_notes
                  if len(note_i)>min_size]
    return long_notes

def clean(txt):
    allowed_chars=u'[a-z|ż|ź|ć|ź|ś|ń|ó|ł|ą|ę]+'
    word_list=re.findall(allowed_chars,txt.lower()) 
    return u' '.join(word_list)

def get_lengths(y):
    return [len(y_i) for y_i in y]

def select_samples(notes,samples_size=100):
    if(samples_size>len(notes)):
        sample_size=len(notes)	
    samples=[]
    index=np.random.randint(0,len(notes))
    while(len(samples)<samples_size):
        sample_i=extract_sample(notes[index])	
        samples.append(sample_i)
        del notes[index]
    return samples	

def extract_sample(sample,max_size=100):
    return sample[0:max_size+1]

#notes=read_notes('pap.txt')
#print(notes[1])
#samples=select_samples(notes)
#print(samples[0])