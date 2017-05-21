import pdb
import mxnet as mx
from poserecog.bucket_io import BucketSentenceIter
from poserecog.get_lstm_sym import get_lstm_sym_test

num_lstm_layer = 3
num_hidden = 512
num_embed = 256

batch_size=1
buckets=[129]
contexts = mx.gpu(5)

data_train = BucketSentenceIter(buckets, batch_size,dataPath = 'out' )
sym_gen_lstm = get_lstm_sym_test(num_hidden=num_hidden,\
                            num_lstm_layer=num_lstm_layer,num_embed=num_embed)

pdb.set_trace()
_, arg_params, aux_params = mx.model.load_checkpoint('model/pose_lstm',1)
model = mx.model.FeedForward(ctx=contexts,
                                 symbol=sym_gen_lstm(buckets[0]),
                                 arg_params=arg_params,
                                 aux_params=aux_params,
                                 numpy_batch_size=1)
data_iter = iter(data_train)
batch = next(data_iter).data[0].asnumpy()
ret = model.predict(batch)
print ret
