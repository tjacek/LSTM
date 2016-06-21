# -*- coding: utf-8 -*-
import codecs,re
#import char    

def read_notes(filename,min_size=100):
    txt = codecs.open(filename,'r','utf8')
    txt=txt.read()
    notes=re.split(u'#[0-9]+',txt)
    long_notes=[ note_i for note_i in notes
                  if len(note_i)>min_size]
    return long_notes

def clean(txt):
    allowed_chars=u'[(a-z)|ż|ź|ć|ź|ś|ń|ó|ł|ą|ę]+'
    word_list=re.findall(allowed_chars,txt.lower()) 
    return u' '.join(word_list)

def get_lengths(y):
    return [len(y_i) for y_i in y]

note=read_notes('pap.txt')[0]
print(note)
print(clean(note))

