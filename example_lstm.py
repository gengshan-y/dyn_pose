import pdb
import numpy as np
import json
import mxnet as mx
from poserecog.bucket_io import BucketSentenceIter
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

contexts = mx.context.gpu(5)
batch_size = 1  #
num_epochs = 25
disp_batches = 5
buckets = [129]  #

num_hidden = 512
num_lstm_layer = 3
invalid_label = None

data_train = BucketSentenceIter(buckets, batch_size,dataPath = 'out' )

stack = mx.rnn.SequentialRNNCell()
for i in range(num_lstm_layer):
    stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_'%i))

def sym_gen(seq_len):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=28,
                             output_dim=200, name='embed')

    stack.reset()
    outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

    pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
    pred = mx.sym.FullyConnected(data=pred, num_hidden=28, name='pred')

    label = mx.sym.Reshape(label, shape=(-1,))
    pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return pred, ('data',), ('softmax_label',)

model = mx.mod.BucketingModule(
    sym_gen             = sym_gen,
    default_bucket_key  = data_train.default_bucket_key,
    context             = contexts)

model.fit(
    train_data          = data_train,
    eval_data           = data_train,
    eval_metric         = mx.metric.Perplexity(invalid_label),
    kvstore             = 'device',
    optimizer           = 'sgd',
    optimizer_params    = { 'learning_rate': 0.01,
                            'momentum': 0,
                            'wd': 0.00001 },
    initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
    num_epoch           = num_epochs,
    batch_end_callback  = mx.callback.Speedometer(batch_size, disp_batches))
