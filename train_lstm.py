#!/home/gengshan/workNov/env2/bin/python
import pdb
import numpy as np
import json
import mxnet as mx
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm_o
import logging
import tensorboard
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
logging.basicConfig(level=logging.DEBUG)


def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


contexts = mx.context.gpu(5)
batch_size = 16  #
num_epochs = 200
disp_batches = 1
buckets = [129]  #
invalid_label = -1

num_hidden = 512
num_lstm_layer = 3
num_embed = 256
num_label = 3


init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h


data_train = BucketSentenceIter(buckets, batch_size,dataPath = 'out' )
data_val = BucketSentenceIter(buckets, batch_size,dataPath = 'out' )
data_train.provide_data += init_states
data_val.provide_data += init_states

sym_gen = get_lstm_o(num_lstm_layer=num_lstm_layer, input_len=28,
            num_hidden = num_hidden, num_embed = num_embed,
            num_label = num_label, dropout = 0.)

model = mx.module.Module(sym_gen(buckets[0])[0], data_names = [x[0] for x in data_train.provide_data],\
                   label_names=('softmax_label',), context = contexts)
model.bind(data_shapes=data_train.provide_data,\
           label_shapes=data_train.provide_label,inputs_need_grad=True)

summary_writer = tensorboard.FileWriter('log/')
def monitor_train(param):
    metric = dict(param.eval_metric.get_name_value())
    summary_writer.add_summary(tensorboard.summary.scalar('perp',\
                                metric['Perplexity']))

def mon_grad(params):
  print params.locals['self']._exec_group.get_outputs()

batch_end_callbacks = [mx.callback.Speedometer(batch_size, disp_batches)]
batch_end_callbacks.append(monitor_train)
#batch_end_callbacks.append(mon_grad)

lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[90,120], factor=0.5)
optimizer = mx.optimizer.SGD(learning_rate = 0.0001, momentum = 0, wd = 0.0001,\
                             lr_scheduler = lr_scheduler)
model.fit(
    train_data          = data_train,
    eval_data           = data_val,
    eval_metric         = mx.metric.np(Perplexity),
    kvstore             = 'device',
    optimizer           = optimizer,
    initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
    num_epoch           = num_epochs,
    batch_end_callback  = batch_end_callbacks,
    epoch_end_callback  = mx.callback.do_checkpoint("model/pose_lstm"))
