import pdb
import mxnet as mx
import numpy as np
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm_sym

num_lstm_layer = 1
num_hidden = 10
num_embed = 256

batch_size=1
buckets=[38]
contexts = mx.gpu(5)

data_train = BucketSentenceIter(buckets, batch_size,dataPath = 'out' )
sym_gen_lstm = get_lstm_sym(num_hidden=num_hidden, num_lstm_layer=num_lstm_layer,\
                            num_embed=num_embed, take_softmax=False)

model = mx.mod.BucketingModule(
    sym_gen             = sym_gen_lstm,
    default_bucket_key  = data_train.default_bucket_key,
    context             = contexts)

_, arg_params, aux_params = mx.model.load_checkpoint('model/pose_lstm',25)
model.bind(data_shapes=data_train.provide_data,label_shapes=None,for_training=False)
model.set_params(arg_params = arg_params, aux_params=aux_params)

data_iter = iter(data_train)
end_of_batch = False
next_data_batch = next(data_iter)
while not end_of_batch:
  lbs=next_data_batch.label[0].asnumpy().flatten()
  print 'cate:%f' % lbs[np.where(lbs != -1)][0]
  print next_data_batch.data[0].asnumpy()[0]
  model.forward(next_data_batch)
  ret = model.get_outputs()[0].asnumpy()
  print np.mean(ret,axis=0)
  pdb.set_trace()
  try:
    next_data_batch = next(data_iter)
  except StopIteration:
    end_of_batch = True
