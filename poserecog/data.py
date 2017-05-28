import mxnet as mx
import cv2
import glob
import poserecog.util as util
import numpy as np

class CamIter:
  def __init__(self,boxsize=368,path=None,stream=False,batch_size=1):
    self.boxsize = boxsize
    self.iter_list = glob.glob(path)
    self.num_data = len(self.iter_list)
    self.batch_size = batch_size
    self.cursor = 0
  
  def reset(self):
    self.cursor = 0

  def get_batch_size(self):
    return self.batch_size

  def iter_next(self):
    rem = self.num_data - self.cursor
    if rem > self.batch_size:
      return self.batch_size
    else:
      return rem
 
  def _read_img(self,path):
    img = cv2.imread(path)
    scale = self.boxsize / (img.shape[0] * 1.0)
    imageToTest = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest)
    img1= np.transpose(np.float32(imageToTest_padded[:,:,:]), (2,0,1))/256 - 0.5;
    return img1,{'img':imageToTest,'path':path}

  def next(self):
    batch_size = self.iter_next()
    if batch_size:
      batches = [];imgdt_list=[]
      for i in range(batch_size):
        img1,img_data = self._read_img( self.iter_list[self.cursor + i] )
        batches.append(img1);imgdt_list.append(img_data)
      self.cursor += batch_size
      batches = np.asarray(batches)
      return mx.io.NDArrayIter(data = batches),imgdt_list
    else:
      raise StopIteration
