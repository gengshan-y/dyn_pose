import pdb
import glob
import numpy as np
import mxnet as mx
import os
import json
from poserecog.util import recReadDir
from sklearn.preprocessing import normalize

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

class ModelParallelBatch(object):
    """Batch used for model parallelism"""
    def __init__(self, data, bucket_key):
        self.data = np.array(data)
        self.bucket_key = bucket_key

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch

class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, buckets, batch_size,
                 data_name='data', label_name='label',
                 model_parallel=False, dataPath = './data', train = False):
        super(BucketSentenceIter, self).__init__()

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)
        buckets.sort()
        self.buckets = buckets
        self.data = [[] for _ in buckets]
        self.tmpLabel = []
        self.train = train

        # save data path
        self.data_path = dataPath
        self.dLoader()
        trainX = self.trainX
        trainY = self.trainY


        self.data_name = data_name
        self.label_name = label_name
        self.model_parallel = model_parallel



        # for sentence in sentences:
        #     sentence = self.text2id(sentence, vocab)
        #     if len(sentence) == 0:
        #         continue
        for it, sentence in enumerate(trainX):
            for i, bkt in enumerate(buckets):
                if bkt >= len(sentence):
                    sentence = np.asarray(sentence)
                    self.data[i].append(sentence)
                    self.tmpLabel.append(trainY[it])
                    break
            # we just ignore the sentence it is longer than the maximum
            # bucket size here

        # convert data into ndarrays for better speed during training
        data = [np.zeros((len(x), buckets[i], x[0].shape[1])) for i, x in enumerate(self.data)]
        # now it is N * (736, 129, 54)
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                data[i_bucket][j, :sentence.shape[0], :] = sentence
        self.data = data


        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        self.featLen = data[0].shape[-1]
        self.make_data_iter_plan()

        self.provide_data = [('data', (batch_size, self.default_bucket_key, self.featLen \
                             ))]
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]
        print self.provide_data
        print self.provide_label

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append(len(self.data[i]) / self.batch_size)
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]
        #bucket_idx_all = [range(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            if not self.model_parallel:
                data = np.zeros((self.batch_size, self.buckets[i_bucket],\
                                 self.featLen))
                label = np.zeros((self.batch_size, self.buckets[i_bucket]))
                self.data_buffer.append(data)
                self.label_buffer.append(label)
            else:
                data = np.zeros((self.buckets[i_bucket], self.batch_size))
                self.data_buffer.append(data)

        if self.model_parallel:
            # Transpose data if model parallel 
            for i in range(len(self.data)):
                bucket_data = self.data[i]
                self.data[i] = np.transpose(bucket_data)

    def __iter__(self):
        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size

            # Model parallelism 
            if self.model_parallel:
                if self.data[i_bucket][:, idx].shape[1] == 0:
                    print "WARNING: detected shape " + str(self.data[i_bucket][:, idx].shape)
                    continue
                data[:] = self.data[i_bucket][:, idx]
                data_batch = ModelParallelBatch(data, self.buckets[i_bucket])
                yield data_batch
            
            # Data parallelism
            else:
                data[:] = self.data[i_bucket][idx]

                for sentence in data:
                    assert len(sentence) == self.buckets[i_bucket]
                

                label = self.label_buffer[i_bucket]
                # label[:, :-1] = data[:, 1:]
                # label[:, -1] = 0
                vecLabel = np.asarray(self.tmpLabel)[idx]
                label[:] = 0  # no overlap bet. batches
                for it, lb in enumerate(vecLabel):
                    label[it, :len(lb)] = lb
                #print label

                data_all = [mx.nd.array(data)]
                label_all = [mx.nd.array(label)]
                data_names = ['data']
                label_names = ['softmax_label']
                data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                         self.buckets[i_bucket])
                yield data_batch
       


    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]

    def dLoader(self):
        with open('split.json', 'r') as f:
          split = json.load(f)
        if self.train:
          split = split['train']
        else:
          split = split['val']
        print( 'reading from %s, phase train %d' % (self.data_path,self.train) )
        cates = set([x.split('_')[0] for x in split])
        cates.remove('stop');cates.remove('circle')
        pdb.set_trace()
        print 'categories: ' + str(cates)
        self.cls_num = len(cates)

        data = {}
        # get data path
        for cate in cates:
            data[cate] = sum([glob.glob('%s/%s*.json'%(self.data_path,x))\
                          for x in split if cate in x] ,[])

        trainX = []
        trainY = []
        for idx, k in enumerate(data.keys()):
            print 'label: %d, category: %s' % (idx+1,k)
            # for each class
            for it in data[k]:
                print it
                rawx = json.load(open(it, 'r'))
                anc = np.asarray( [it['p'] for it in rawx] )
                anc_head = anc[0][0]  # fr0 head
                anc_len = np.mean(np.linalg.norm(anc[:,0] - anc[:,1],axis=1) +\
                          np.linalg.norm(anc[:,8] - anc[:,9],axis=1) +\
                          np.linalg.norm(anc[:,11] - anc[:,12],axis=1))
                          #2d pose vert line stable
                x = [(it['p']-anc_head).reshape(-1,)/anc_len for it in rawx]
                #x = [it[:14] for it in x]
                
                y = np.asarray([0] * self.default_bucket_key)
                y[:len(rawx)] = idx + 1
                #y[len(x)-1] = idx
                #y =  [-1] * len(x)
                #y[-1] = idx # indicate len
                trainX.append(x)
                trainY.append(y)
        print str(len(trainY)) + ' samples'
    
        if self.train:
          for it in range(0, 5):
            trainX += trainX
            trainY += trainY    
        
        self.trainX = trainX
        self.trainY = trainY
