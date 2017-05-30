#!/home/gengshan/workNov/env2/bin/python
import pdb
import mxnet as mx
import numpy as np
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm
from poserecog.config import lstm_config as lcf

lcf.batch_size = 1

init_c = [('l%d_init_c'%l, (lcf.batch_size, lcf.num_hidden)) \
          for l in range(lcf.num_lstm_layer)]
init_h = [('l%d_init_h'%l, (lcf.batch_size, lcf.num_hidden)) \
          for l in range(lcf.num_lstm_layer)]
init_states = init_c + init_h

data_train = BucketSentenceIter(lcf.buckets, lcf.batch_size, dataPath = 'out')
data_train.provide_data += init_states

sym_gen = get_lstm(num_lstm_layer=lcf.num_lstm_layer, input_len=lcf.input_dim,
            num_hidden = lcf.num_hidden, num_embed = lcf.num_embed,
            num_label = data_train.cls_num + 1, dropout = lcf.dropout)

model = mx.mod.BucketingModule(
    sym_gen             = sym_gen,
    default_bucket_key  = data_train.default_bucket_key,
    context             = mx.gpu(lcf.ctx))

_, arg_params, aux_params = mx.model.load_checkpoint('model/pose_lstm',lcf.load_epoch)
model.bind(data_shapes=data_train.provide_data,label_shapes=data_train.provide_label,\
           for_training=False)
model.set_params(arg_params = arg_params, aux_params=aux_params)

pdb.set_trace()

data_iter = iter(data_train)
end_of_batch = False
next_data_batch = next(data_iter)
lb = [];pd = []
while not end_of_batch:
  lbs=next_data_batch.label[0].asnumpy().flatten()
  #print 'cate:%f' % lbs[np.where(lbs != -1)][0]
  #print next_data_batch.data[0].asnumpy()[0]
  model.forward(next_data_batch)
  pdb.set_trace()
  ret = model.get_outputs()[0].asnumpy()
  lb.append( lbs[np.where(lbs != 0)][0] )
  pd.append(np.argmax(np.sum(ret,axis=0)[1:])+1) 
  #print np.argmax(np.sum(ret,axis=0)[1:])+1
  try:
    next_data_batch = next(data_iter)
  except StopIteration:
    end_of_batch = True

from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
pdb.set_trace()
print confusion_matrix(lb,pd)
print precision_score(lb,pd,average=None)
print accuracy_score(lb,pd)
