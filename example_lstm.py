import pdb
import numpy as np
import json
import mxnet as mx
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm_sym
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


contexts = mx.context.gpu(5)
batch_size = 1  #
num_epochs = 2
disp_batches = 5
buckets = [129]  #
invalid_label = None

num_hidden = 512
num_lstm_layer = 3
num_embed = 256

data_train = BucketSentenceIter(buckets, batch_size,dataPath = 'out' )
sym_gen_lstm = get_lstm_sym(num_hidden=num_hidden,\
                            num_lstm_layer=num_lstm_layer,num_embed=num_embed)

model = mx.mod.BucketingModule(
    sym_gen             = sym_gen_lstm,
    default_bucket_key  = data_train.default_bucket_key,
    context             = contexts)

pdb.set_trace()

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
    batch_end_callback  = mx.callback.Speedometer(batch_size, disp_batches),
    epoch_end_callback  = mx.callback.do_checkpoint("model/pose_lstm"))
