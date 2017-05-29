#!/home/gengshan/workNov/env2/bin/python
import pdb
import numpy as np
import mxnet as mx
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm
from poserecog.config import lstm_config as lcf
import logging
#head = '%(asctime)-15s %(message)s'
#logging.basicConfig(level=logging.DEBUG, format=head)
logging.basicConfig(level=logging.DEBUG)
import tensorboard
summary_writer = tensorboard.FileWriter('log/')


def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


def monitor_train(param):
    metric = dict(param.eval_metric.get_name_value())
    summary_writer.add_summary(tensorboard.summary.scalar('perp',\
                                metric['Perplexity']))

def mon_grad(params):
  print params.locals['self']._exec_group.get_outputs()


init_c = [('l%d_init_c'%l, (lcf.batch_size, lcf.num_hidden)) \
          for l in range(lcf.num_lstm_layer)]
init_h = [('l%d_init_h'%l, (lcf.batch_size, lcf.num_hidden)) \
          for l in range(lcf.num_lstm_layer)]
init_states = init_c + init_h

data_train = BucketSentenceIter(lcf.buckets, lcf.batch_size,\
                                dataPath = 'out', train = True )
data_val = BucketSentenceIter(lcf.buckets, lcf.batch_size,\
                                dataPath = 'out' )

data_train.provide_data += init_states
data_val.provide_data += init_states

sym_gen = get_lstm(num_lstm_layer=lcf.num_lstm_layer, input_len=lcf.input_dim,
                   num_hidden = lcf.num_hidden, num_embed = lcf.num_embed,
                   num_label = lcf.num_label, dropout = lcf.dropout)

model = mx.module.Module(sym_gen(lcf.buckets[0])[0], \
                         data_names = [x[0] for x in data_train.provide_data],\
                         label_names=('softmax_label',), context = mx.gpu(lcf.ctx))
model.bind(data_shapes=data_train.provide_data,\
           label_shapes=data_train.provide_label,inputs_need_grad=False)


batch_end_callbacks = [mx.callback.Speedometer(lcf.batch_size, lcf.disp_batches)]
batch_end_callbacks.append(monitor_train)
#batch_end_callbacks.append(mon_grad)

lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[5000,10000], factor=0.5)
optimizer = mx.optimizer.SGD(learning_rate = 0.01, momentum = 0, wd = 0.0001,\
                  lr_scheduler = lr_scheduler, rescale_grad = 1./lcf.batch_size)


model.fit(
    train_data          = data_train,
    eval_data           = data_val,
    eval_metric         = mx.metric.np(Perplexity),
    kvstore             = 'device',
    optimizer           = optimizer,
    initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
    num_epoch           = lcf.num_epochs,
    batch_end_callback  = batch_end_callbacks,
    epoch_end_callback  = mx.callback.do_checkpoint("model/pose_lstm"))
