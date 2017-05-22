import pdb
import mxnet as mx
import numpy as np
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm_sym_test

num_lstm_layer = 3
num_hidden = 512
num_embed = 256

batch_size=1
buckets=[129]
contexts = mx.gpu(5)

data_train = BucketSentenceIter(buckets, batch_size,dataPath = 'out' )
pdb.set_trace()
sym_lstm = get_lstm_sym_test(num_hidden=num_hidden,\
                            num_lstm_layer=num_lstm_layer,num_embed=num_embed)
sym_lstm = sym_lstm(buckets[0])


t = sym_lstm.get_internals()
_, arg_params, aux_params = mx.model.load_checkpoint('model/pose_lstm',1)
model = mx.model.FeedForward(ctx=contexts,
                                 symbol=sym_lstm,
                                 arg_params=arg_params,
                                 aux_params=aux_params,
                                 numpy_batch_size=1)
data_iter = iter(data_train)
end_of_batch = False
next_data_batch = next(data_iter)
while not end_of_batch:
  data_batch = next_data_batch.data[0]
  print 'cate:%f' % next_data_batch.label[0].asnumpy().flatten()[0]
  # print data_batch.asnumpy()
  ret = model.predict(data_batch)
  pdb.set_trace()
  print ret
  print np.argmax(ret[0])
  try:
    next_data_batch = next(data_iter)
  except StopIteration:
    end_of_batch = True
