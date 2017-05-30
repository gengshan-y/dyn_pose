import pdb
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
    self.pixel_mean = np.array([103.939, 116.779, 123.68])
  
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

    im_array, im_scale = util.resize(img, 600, 1000)
    im_array = util.transform(im_array ,self.pixel_mean)  # to rgb, subtruct mean
    im_info = np.array([im_array.shape[1], im_array.shape[2], im_scale], dtype=np.float32)
    return [im_array, im_info ,{'img':imageToTest,'path':path, 'scale':im_scale / scale}]

  def next(self):
    batch_size = self.iter_next()
    if batch_size:
      data = [];im_info = [];imgdt_list=[]
      for i in range(batch_size):
        batch = self._read_img( self.iter_list[self.cursor + i] )
        data.append(batch[0]);im_info.append(batch[1]);imgdt_list.append(batch[2])
      self.cursor += batch_size
      return mx.io.NDArrayIter(data = {'data':np.asarray(data),\
                                       'im_info':np.asarray(im_info)}),imgdt_list
    else:
      raise StopIteration
