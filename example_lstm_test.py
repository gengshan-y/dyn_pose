#!/home/gengshan/workNov/env2/bin/python
import pdb
import mxnet as mx
import numpy as np
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm_sym,get_lstm_o
from poserecog.rnn_model import LSTMInferenceModel

num_lstm_layer = 3
num_hidden = 512
num_embed = 256

batch_size=1
buckets=[129]
contexts = mx.gpu(5)
num_label = 3

init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h


data_train = BucketSentenceIter(buckets, batch_size,dataPath = 'out' )
data_train.provide_data += init_states
sym_gen_lstm = get_lstm_sym(num_hidden=num_hidden, num_lstm_layer=num_lstm_layer,\
                            num_embed=num_embed, num_label = num_label,\
                            take_softmax=False, dropout=0.2)

sym_gen = get_lstm_o(num_lstm_layer=num_lstm_layer, input_len=28,
            num_hidden = num_hidden, num_embed = num_embed,
            num_label = num_label, dropout = 0.2)

model = mx.mod.BucketingModule(
    sym_gen             = sym_gen,
    default_bucket_key  = data_train.default_bucket_key,
    context             = contexts)

_, arg_params, aux_params = mx.model.load_checkpoint('model/pose_lstm',10)
model.bind(data_shapes=data_train.provide_data,label_shapes=data_train.provide_label,for_training=False)
model.set_params(arg_params = arg_params, aux_params=aux_params)

pdb.set_trace()
model = LSTMInferenceModel(num_lstm_layer, 28,
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label = num_label, arg_params=arg_params, ctx=mx.gpu(), dropout=0.2)

data_iter = iter(data_train)
end_of_batch = False
next_data_batch = next(data_iter)
while not end_of_batch:
  lbs=next_data_batch.label[0].asnumpy().flatten()
  print 'cate:%f' % lbs[np.where(lbs != -1)][0]
  print next_data_batch.data[0].asnumpy()[0]
  #model.forward(next_data_batch)
  #ret = model.get_outputs()[0].asnumpy()
  ret = np.zeros((buckets[0],num_label))
  ret[0] = model.forward(mx.ndarray.reshape(next_data_batch.data[0][0][0],(1,-1)),True)
  for i,dt in enumerate(next_data_batch.data[0][0][1:]):
    ret[i+1] = model.forward(mx.ndarray.reshape(dt,(1,-1)))
  pdb.set_trace()
  print np.argmax(np.sum(ret,axis=0)[1:])+1
  try:
    next_data_batch = next(data_iter)
  except StopIteration:
    end_of_batch = True
