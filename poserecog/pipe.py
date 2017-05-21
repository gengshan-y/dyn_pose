import pdb
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.ndimage
import scipy.signal
import time
import ConfigParser
import mxnet as mx
import poserecog.util as util

class Pipeline:
  def __init__(self, config_path):
    config = ConfigParser.ConfigParser()
    config.read(config_path)

    det_model_prefix = config.get('model', 'det_model_prefix')
    pose_model_prefix = config.get('model', 'pose_model_prefix')
    self.ctx = mx.gpu(int(config.get('model', 'ctx')))
    self.boxsize = int(config.get('model', 'boxsize'))
    self.sigma = int(config.get('model', 'sigma'))
    self.np = int(config.get('model', 'np'))

    # model base folder
    base_folder = os.path.split(os.path.abspath(config_path))[0]
    det_model_path = os.path.join(base_folder, det_model_prefix)
    pose_model_path = os.path.join(base_folder, pose_model_prefix)

    self.model_det = self.load_model(det_model_path)
    self.model_pose = self.load_model(pose_model_path)

    self.gauss = self.get_gauss()


  def get_gauss(self):
    gauss1d = scipy.signal.gaussian(self.boxsize,self.sigma)
    gauss2d = np.expand_dims(gauss1d,0) * np.expand_dims(gauss1d,1)
    return np.transpose(gauss2d[:,:,np.newaxis,np.newaxis], (3,2,0,1))


  def load_model(self, model_path):
    net, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
    model = mx.model.FeedForward(ctx=self.ctx,
                                 symbol=net,
                                 arg_params=arg_params,
                                 aux_params=aux_params,
                                 numpy_batch_size=1)
    return model

  def process(self, img):
    scale = self.boxsize / (img.shape[0] * 1.0)
    imageToTest = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest)
    img1= np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
    batch = mx.io.NDArrayIter(data = {'data':img1})
    start_time = time.time()
    output=self.model_det.predict(batch)
    print('Person net took %.2f ms.' % (1000 * (time.time() - start_time)))

    person_map = np.squeeze(output)
    person_map_resized = cv2.resize(person_map, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    x,y = util.get_maxima(person_map_resized)
    person_image=cv2.getRectSubPix(imageToTest,(self.boxsize,self.boxsize),(x,y))

    img1 = np.transpose(np.float32(person_image[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
    batch = mx.io.NDArrayIter(data = {'image':img1, 'center_map':self.gauss}) 
    start_time = time.time()
    output=self.model_pose.predict(batch)[-1]  # last layer
    print('Person net took %.2f ms.' % (1000 * (time.time() - start_time)))
    pdb.set_trace()
    ret=np.zeros((self.np,2))
    for part in range(self.np):
      part_map = output[0,part,:,:]
      part_map = cv2.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
      ret[part,1],ret[part,0] = np.unravel_index(part_map.argmax(),part_map.shape)
    self.disp_map(person_image.copy(), ret)
    return ret

  def display_pose(self, person_image, output):
    down_scaled_image = cv2.resize(person_image, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    canvas = np.empty(shape=(self.boxsize/2, 0, 3))
    for part in [0,3,7,10,12]: # sample 5 body parts: [head, right elbow, left wrist, right ankle, left knee]
      part_map = output[0,part,:,:]
      part_map_resized = cv2.resize(part_map, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC) #only for displaying
      part_map_color = util.colorize(part_map_resized)
      part_map_color_blend = part_map_color * 0.5 + down_scaled_image * 0.5
      canvas = np.concatenate((canvas, part_map_color_blend), axis=1)
    canvas = np.concatenate((canvas, 255 * np.ones((self.boxsize/2, 5, 3))), axis=1)
    cv2.imwrite('test.png',canvas)


  def disp_map(self,img, joints):
    pdb.set_trace()
    joints = joints.astype(int)
    for j in joints:
      cv2.circle(img, (j[0], j[1]), 3, (0,0,255), -1)
    cv2.imwrite('test.png',img) 
