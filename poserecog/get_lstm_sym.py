import mxnet as mx
from lstm import lstm_unroll


def get_lstm(num_lstm_layer=None, input_len=None,
            num_hidden = None, num_embed = None,
            num_label = None, dropout = 0.):
    def sym_gen(seq_len):
      return lstm_unroll(num_lstm_layer, seq_len, input_len,
                       num_hidden = num_hidden, num_embed = num_embed,
                       num_label = num_label, dropout = dropout)
    return sym_gen

