import pdb
import mxnet as mx
from lstm import lstm_unroll


def get_lstm_o(num_lstm_layer=None, input_len=None,
            num_hidden = None, num_embed = None,
            num_label = None, dropout = 0.):
    def sym_gen(seq_len):
      return lstm_unroll(num_lstm_layer, seq_len, input_len,
                       num_hidden = num_hidden, num_embed = num_embed,
                       num_label = num_label, dropout = dropout)
                       # supress the nill label automatically
    return sym_gen

def get_lstm_sym(num_hidden=None,num_lstm_layer=None,num_embed=None,num_label=None,\
                 take_softmax = True, dropout = 0.):
  def sym_gen_lstm(seq_len):
    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_lstm_layer):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_'%i))
        #stack.add(mx.rnn.DropoutCell(dropout))
    data = mx.sym.Variable('data')
   # embed = mx.sym.Embedding(data=data, input_dim=28,
   #                          output_dim=num_embed, name='embed')

    stack.reset()
    #outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)
    outputs, states = stack.unroll(seq_len, inputs=data, merge_outputs=True)

    pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
    pred = mx.sym.FullyConnected(data=pred, num_hidden=num_label, name='pred')
    #pred = mx.sym.FullyConnected(data=outputs, num_hidden=num_label, name='pred')
    #pred = mx.sym.Reshape(pred, shape=(-1, seq_len))

    if take_softmax:
      label = mx.sym.Variable('softmax_label')
      #label = mx.sym.transpose(data=label)
      label = mx.sym.Reshape(label, shape=(-1,))
      pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
      return pred, ('data',), ('softmax_label',)
    else:
      return pred, ('data',), []

  return sym_gen_lstm
