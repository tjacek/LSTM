import numpy as np
import rnn,gen,tools,lstm
import gen.add
import char,gen_seq

def pap():
    X,y=char.make_pap_dataset('char/pap.txt')
    n_chars=len(char.ALPHA)
    pap_params={'n_cats':n_chars,'seq_dim':n_chars,'hidden_dim':3*n_chars,
            'cell_dim':3*n_chars,'learning_rate':0.1,'momentum':0.9}
    rnn_model=rnn.build_rnn(pap_params)
    rnn_model.read('models/lstm300')
    train_super(X,y,rnn_model,50)
    rnn_model.save('models/lstm300')

def train_super(X,y,model,epochs=100,output='models/lstm7000'):
    x_batches=tools.get_batches(X)
    y_batches=tools.get_batches(y)
    for j in range(epochs):
        cost=[]
        for i,y_i in enumerate(y_batches):     
            x_i=x_batches[i]
            xt_i=tools.time_first(x_i)
            yt_i=tools.time_first(y_i)
            value=model.pred(xt_i)#,y_i.astype(int))            
            yt_i=tools.to_one_hot(yt_i,xt_i.shape[2])
            cost_i=model.updates(xt_i,yt_i)
            cost.append(cost_i)
        sum_i=sum(cost)/float(len(cost))
        print(str(j) + ' ' + str(sum_i))
        model.save(output)
    seq_m=gen_seq.gen_seq(100,model)
    print(char.seq_to_words(seq_m))
    return model

def train_super_mask(X,y,mask,model,output='models/lstm7000'):
    x_batches=tools.get_batches(X)
    y_batches=tools.get_batches(y)
    mask_batches=tools.get_batches(mask)
    for j in range(epochs):
        cost=[]
        for i,y_i in enumerate(y_batches):         
            x_i=x_batches[i]
            xt_i=tools.time_first(x_i)
            mask_i=mask_batches[i]
            maskt_i=tools.time_first(mask_i)
            value=model.pred(xt_i,maskt_i)      
            print(value.shape)
            print(y_i.shape)
            print(model.loss(xt_i,y_i.astype(int) ,maskt_i))
            print(model.updates(xt_i,y_i,maskt_i))
            cost.append(cost_i)
        sum_i=sum(cost)/float(len(cost))
        print(str(j) + ' ' + str(sum_i))
    model.save(output)

if __name__ == "__main__":
    pap()