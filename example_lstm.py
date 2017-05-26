#!/home/gengshan/workNov/env2/bin/python
import pdb
import numpy as np
import json
import mxnet as mx
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm_sym
import logging
import tensorboard
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


def Perplexity(label, pred):
    pdb.set_trace()
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
batch_size = 1  #
num_epochs = 25
disp_batches = 5
buckets = [38]  #
invalid_label = -1

num_hidden = 10
num_lstm_layer = 1
num_embed = 256

data_train = BucketSentenceIter(buckets, batch_size,dataPath = 'out' )
sym_gen_lstm = get_lstm_sym(num_hidden=num_hidden, num_lstm_layer=num_lstm_layer,\
                            num_embed=num_embed)

model = mx.mod.BucketingModule(
    sym_gen             = sym_gen_lstm,
    default_bucket_key  = data_train.default_bucket_key,
    context             = contexts)

summary_writer = tensorboard.FileWriter('log/')
def monitor_train(param):
    metric = dict(param.eval_metric.get_name_value())
    summary_writer.add_summary(tensorboard.summary.scalar('perp',\
                                metric['perplexity']))

batch_end_callbacks = [mx.callback.Speedometer(batch_size, disp_batches)]
batch_end_callbacks.append(monitor_train)

model.fit(
    train_data          = data_train,
    eval_data           = data_train,
    #eval_metric         = mx.metric.np(Perplexity),
    eval_metric         = mx.metric.Perplexity(invalid_label),
    kvstore             = 'device',
    optimizer           = 'sgd',
    optimizer_params    = { 'learning_rate': 0.001,
                            'momentum': 0,
                            'wd': 0.00001 },
    initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
    num_epoch           = num_epochs,
    batch_end_callback  = batch_end_callbacks,
    epoch_end_callback  = mx.callback.do_checkpoint("model/pose_lstm"))
