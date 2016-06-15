import theano, theano.tensor as T
import numpy as np
import tools,layer

def mask_lstm(hyper_params,in_var,mask_var):
    forget_gate,in_gate,cell_gate,out_gate=lstm_params(hyper_params)
    def recurrence(x_t,h_t,c_t):
        f_t=T.nnet.sigmoid(forget_gate.linear(x_t,h_t))
        i_t=T.nnet.sigmoid(in_gate.linear(x_t,h_t))
        c_t_prop=T.tanh(cell_gate.linear(x_t,h_t))
        c_t_next=f_t*c_t+i_t*c_t_prop
        o_t=T.nnet.sigmoid(out_gate.linear(x_t,h_t))
        h_t_next=o_t*T.tanh(c_t_next)
        out_t=T.nnet.softmax(h_t_next)
        return [h_t,c_t,out_t]
    
    start_cell=T.zeros( (in_var.shape[1], hyper_params['cell_dim']),dtype=float)
    start_hidden=T.zeros( (in_var.shape[1],hyper_params['hidden_dim']), dtype=float)
    [h,c,s], updates = theano.scan(
            recurrence,
            sequences=in_var,
            outputs_info=[start_hidden,start_cell,None],
            n_steps=in_var.shape[0])
    final= mask_var*s#T.dot(mask_var,s)
    params=layer.get_params([forget_gate,in_gate,cell_gate,out_gate])
    return final,params

def lstm_params(hyper_params):
    input_dim=hyper_params['seq_dim']
    hidden_dim=hyper_params['hidden_dim']
    cell_dim=hyper_params['cell_dim']
    #con_dim=input_dim+hidden_dim
    forget_gate=layer.create_gate(out_size=cell_dim, x_size=input_dim, h_size=input_dim,name='forgot')
    in_gate=layer.create_gate(out_size=cell_dim, x_size=input_dim, h_size=input_dim,name='in')
    cell_gate=layer.create_gate(out_size=cell_dim, x_size=input_dim, h_size=input_dim, name='cell')
    out_gate=layer.create_gate(out_size=cell_dim, x_size=input_dim, h_size=input_dim, name='out')
    return forget_gate,in_gate,cell_gate,out_gate

def default_params():
    return {'n_cats':3,'seq_dim':3,'hidden_dim':3,
            'cell_dim':3,'learning_rate':0.1,'momentum':0.9}