import sys
import rnn,char,gen_seq

def generate_note(n,seed='polska'):
    model=read_lstm()
    vec_seed=char.words_to_seq(seed)
    seq_m=gen_seq.gen_seq(n,model,vec_seed)
    note=char.seq_to_words(seq_m)
    return note

def read_lstm(filename='models/lstm300'):
    n_chars=len(char.ALPHA)
    pap_params={'n_cats':n_chars,'seq_dim':n_chars,'hidden_dim':3*n_chars,
            'cell_dim':3*n_chars,'learning_rate':0.1,'momentum':0.9}
    rnn_model=rnn.build_rnn(pap_params)
    rnn_model.read(filename)
    return rnn_model

if __name__ == "__main__":
    if(len(sys.argv)<2):
        n=100
    else:
        n=int(sys.argv[1])
    if(len(sys.argv)<2):
        seed=u'polska'
    else:
        seed=sys.argv[2]
    print(generate_note(n,seed))
