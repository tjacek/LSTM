import theano, theano.tensor as T
import numpy as np
import tools,layer

class LSTMBuilder(object):
    def __init__(self,hyper_params):
        self.hyper_params=hyper_params
        forget_gate,in_gate,cell_gate,out_gate=lstm_params(hyper_params)
        self.forget_gate=forget_gate
        self.in_gate=in_gate
        self.cell_gate=cell_gate
        self.out_gate=out_gate

    def get_step(self):
        def step(x_t,h_t,c_t):
            f_t=T.nnet.sigmoid(self.forget_gate.linear(x_t,h_t))
            i_t=T.nnet.sigmoid(self.in_gate.linear(x_t,h_t))
            c_t_prop=T.tanh(self.cell_gate.linear(x_t,h_t))
            c_t_next=f_t*c_t+i_t*c_t_prop
            o_t=T.nnet.sigmoid(self.out_gate.linear(x_t,h_t))
            h_t_next=o_t*T.tanh(c_t_next)
            return [h_t_next,c_t_next]
        return step
        
    def get_output(self,in_var):
        start_cell=T.zeros( (in_var.shape[1], self.hyper_params['cell_dim']),dtype=float)
        start_hidden=T.zeros( (in_var.shape[1],self.hyper_params['hidden_dim']), dtype=float)
        rec_step=self.get_step()
        [h,c], updates = theano.scan(
            rec_step,
            sequences=in_var,
            outputs_info=[start_hidden,start_cell],#,None],
            n_steps=in_var.shape[0])
        return h

    def get_params(self):
        gates=[self.forget_gate,self.in_gate,self.cell_gate,self.out_gate]
        return layer.get_params(gates)

def lstm_params(hyper_params):
    input_dim=hyper_params['seq_dim']
    hidden_dim=hyper_params['hidden_dim']
    cell_dim=hyper_params['cell_dim']
    #con_dim=input_dim+hidden_dim
    forget_gate=layer.create_gate(out_size=cell_dim, x_size=input_dim, h_size=hidden_dim,name='forgot')
    in_gate=layer.create_gate(out_size=cell_dim, x_size=input_dim, h_size=hidden_dim,name='in')
    cell_gate=layer.create_gate(out_size=cell_dim, x_size=input_dim, h_size=hidden_dim, name='cell')
    out_gate=layer.create_gate(out_size=cell_dim, x_size=input_dim, h_size=hidden_dim, name='out')
    return forget_gate,in_gate,cell_gate,out_gate

def default_params():
    return {'n_cats':2,'seq_dim':2,'hidden_dim':3,
            'cell_dim':3,'learning_rate':0.1,'momentum':0.9}