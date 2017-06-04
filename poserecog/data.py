import pdb
import mxnet as mx
import cv2
import glob
import poserecog.util as util
import numpy as np

class CamIter:
  def __init__(self,boxsize=368,path=None,stream=False,batch_size=1):
    self.boxsize = boxsize
    self.vid_cap = None
    if 'rtsp' in path:
      self.vid_cap = cv2.VideoCapture(path)
      if not self.vid_cap.isOpened():
        raise Exception("no video provided", 0)
    else:
      self.iter_list = glob.glob(path)
      self.cursor = 0
    self.batch_size = batch_size
    self.pixel_mean = np.array([103.939, 116.779, 123.68])
  
  def reset(self):
    self.cursor = 0

  def get_batch_size(self):
    return self.batch_size

  def iter_next(self):
    if self.vid_cap is None:
      rem = len(self.iter_list) - self.cursor
      if rem > self.batch_size:
        return self.batch_size
      else:
         return rem
    else:
      return True
 
  
  def _process_img(self, img):
    scale = self.boxsize / (img.shape[0] * 1.0)
    imageToTest = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    im_array, im_scale = util.resize(img, 600, 1000)
    im_array = util.transform(im_array ,self.pixel_mean)  # to rgb, subtruct mean
    im_info = np.array([im_array.shape[1], im_array.shape[2], im_scale], dtype=np.float32)
    return im_array,im_info,imageToTest, im_scale / scale


  def _read_img(self,path):
    img = cv2.imread(path)
    im_array, im_info, imageToTest, scale = self._process_img(img)
    return [im_array, im_info ,{'img':imageToTest,'path':path, 'scale':scale}]


  def _read_vid(self):
    rval = False
    while not rval:
      rval, frame = self.vid_cap.read() 
    im_array, im_info, imageToTest, scale = self._process_img(frame)
    return [im_array, im_info ,{'img':imageToTest,'path':'rstp', 'scale':scale}]


  def next(self):
    if self.vid_cap is None:
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

    else:
      batch_size = self.batch_size
      data = [];im_info = [];imgdt_list=[]
      for i in range(batch_size):
        batch = self._read_vid()
        data.append(batch[0]);im_info.append(batch[1]);imgdt_list.append(batch[2])
      return mx.io.NDArrayIter(data = {'data':np.asarray(data),\
                                       'im_info':np.asarray(im_info)}),imgdt_list
