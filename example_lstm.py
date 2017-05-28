#!/home/gengshan/workNov/env2/bin/python
import pdb
import numpy as np
import json
import mxnet as mx
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm_sym, get_lstm_o
import logging
import tensorboard
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


#def Perplexity(label, pred):
#  label = label.T.reshape((-1,))
#  idx = np.where(label > -1)[0][0] 
#  loss = 0.
#  for i in range(idx+1):
#    loss += -np.log(max(1e-10, pred[i][int(label[idx])]))
#  # print 'label:%f, loss:%f' % (label[idx],loss/(idx+1))
#  return np.exp(loss/(idx+1))




contexts = mx.context.gpu(5)
batch_size = 10  #
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
sym_gen_lstm = get_lstm_sym(num_hidden=num_hidden, num_lstm_layer=num_lstm_layer,\
                            num_embed=num_embed,num_label=num_label,dropout=0.2)

pdb.set_trace()
sym_gen = get_lstm_o(num_lstm_layer=num_lstm_layer, input_len=28,
            num_hidden = num_hidden, num_embed = num_embed,
            num_label = num_label, dropout = 0.2)

model = mx.mod.BucketingModule(
    sym_gen             = sym_gen,
    default_bucket_key  = data_train.default_bucket_key,
    context             = contexts)

summary_writer = tensorboard.FileWriter('log/')
def monitor_train(param):
    metric = dict(param.eval_metric.get_name_value())
    summary_writer.add_summary(tensorboard.summary.scalar('perp',\
                                metric['Perplexity']))

batch_end_callbacks = [mx.callback.Speedometer(batch_size, disp_batches)]
batch_end_callbacks.append(monitor_train)

lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[90,120], factor=0.5)
optimizer = mx.optimizer.SGD(learning_rate = 0.0001, momentum = 0, wd = 0.0001,\
                             lr_scheduler = lr_scheduler)
model.fit(
    train_data          = data_train,
    eval_data           = data_val,
    eval_metric         = mx.metric.np(Perplexity),
    #eval_metric         = mx.metric.Perplexity(invalid_label),
    kvstore             = 'device',
    optimizer           = optimizer,
    initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
    num_epoch           = num_epochs,
    batch_end_callback  = batch_end_callbacks,
    epoch_end_callback  = mx.callback.do_checkpoint("model/pose_lstm"))
