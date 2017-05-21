import mxnet as mx

def get_lstm_sym(num_hidden=None,num_lstm_layer=None,num_embed=None):
  def sym_gen_lstm(seq_len):
    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_lstm_layer):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_'%i))
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=28,
                             output_dim=num_embed, name='embed')

    stack.reset()
    outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

    pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
    pred = mx.sym.FullyConnected(data=pred, num_hidden=28, name='pred')

    label = mx.sym.Reshape(label, shape=(-1,))
    pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return pred, ('data',), ('softmax_label',)

  return sym_gen_lstm

def get_lstm_sym_test(num_hidden=None,num_lstm_layer=None,num_embed=None):
  def sym_gen_lstm(seq_len):
    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_lstm_layer):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_'%i))
    data = mx.sym.Variable('data')
    embed = mx.sym.Embedding(data=data, input_dim=28,
                             output_dim=num_embed, name='embed')

    stack.reset()
    outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

    pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
    pred = mx.sym.FullyConnected(data=pred, num_hidden=28, name='pred')

    pred = mx.sym.SoftmaxActivation(data=pred, name='softmax')

    return pred

  return sym_gen_lstm
