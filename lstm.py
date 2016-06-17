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
        
    def get_output(self,nn_vars):
        in_var=nn_vars['in_var']
        start_cell,start_hidden=self.init_outputs(in_var)
        rec_step=self.get_step()
        [h,c], updates = theano.scan(
            rec_step,
            sequences=in_var,
            outputs_info=[start_hidden,start_cell],#,None],
            n_steps=in_var.shape[0])
        return h

    def init_outputs(self,in_var):
        cell_dim=self.hyper_params['cell_dim']
        hidden_dim=self.hyper_params['hidden_dim']
        start_cell=T.zeros( (in_var.shape[1],cell_dim),dtype=float)
        start_hidden=T.zeros( (in_var.shape[1],hidden_dim), dtype=float)
        return start_cell,start_hidden

    def get_params(self):
        gates=[self.forget_gate,self.in_gate,self.cell_gate,self.out_gate]
        return layer.get_params(gates)

class MaskLSTMBuilder(LSTMBuilder):
    def __init__(self, hyper_params):
        super(MaskLSTMBuilder, self).__init__(hyper_params)
        #self.arg = arg
    
    def get_step(self):
        old_step=super(MaskLSTMBuilder, self).get_step()
        def step(x_t,m_t,h_t,c_t):
            [h_t_next,c_t_next]=old_step(x_t,h_t,c_t)
            #mask_t=m_t.reshape((m_t.shape[0],1))
            mask_t=T.tile(m_t,(c_t.shape[1],1) )
            mask_t=mask_t.T
            masked_cell = T.switch(mask_t, c_t, c_t_next)
            masked_hid = T.switch(mask_t, h_t, h_t_next)
            return [masked_hid,masked_cell]
        return step    
    
    def get_output(self,nn_vars):
        in_var=nn_vars['in_var']
        mask_var=nn_vars['mask_var']
        start_cell,start_hidden=self.init_outputs(in_var)
        rec_step=self.get_step()
        [h,c], updates = theano.scan(
            rec_step,
            sequences=[in_var,mask_var],
            outputs_info=[start_hidden,start_cell],#,None],
            n_steps=in_var.shape[0])
        return h

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