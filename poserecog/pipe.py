import pdb
import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.ndimage
import scipy.signal
import time
import ConfigParser
import mxnet as mx
import poserecog.util as util
from poserecog.data import CamIter as CamIter


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
    self.length = []


  def get_gauss(self):
    gauss1d = scipy.signal.gaussian(self.boxsize,self.sigma)
    gauss2d = np.expand_dims(gauss1d,0) * np.expand_dims(gauss1d,1)
    return np.transpose(gauss2d[:,:,np.newaxis], (2,0,1))


  def load_model(self, model_path):
    net, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
    model = mx.model.FeedForward(ctx=self.ctx,
                                 symbol=net,
                                 arg_params=arg_params,
                                 aux_params=aux_params,
                                 numpy_batch_size=1)
    return model

  def process(self, img_path, write=True):
    cam_iter = CamIter(boxsize=self.boxsize,path=img_path,batch_size=1)
    write_list=[]
    while cam_iter.iter_next():
      batch,imgdt_list = cam_iter.next()
      start_time = time.time()
      output=self.model_det.predict(batch)
      print('Person net took %.2f ms.' % (1000 * (time.time() - start_time)))

      # form another batch
      batch_pose = [];gausses = []
      for i,it in enumerate(output):
        person_map = np.squeeze(it)
        person_map_resized = cv2.resize(person_map, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        x,y = util.get_maxima(person_map_resized)
        person_image=cv2.getRectSubPix(imgdt_list[i]['img'],(self.boxsize,self.boxsize),(x,y))
        img1 = np.transpose(np.float32(person_image), (2,0,1))/256 - 0.5
        batch_pose.append(img1);gausses.append(self.gauss)
      batch_pose = np.asarray(batch_pose);gausses = np.asarray(gausses)
      batch = mx.io.NDArrayIter(data={'image':batch_pose,'center_map':gausses})

      start_time = time.time()
      output=self.model_pose.predict(batch)[-1]  # last layer
      print('Pose net took %.2f ms.' % (1000 * (time.time() - start_time)))

      for i in range(len(output)):
        ret=[]
        for part in range(self.np):
          part_map = output[i,part,:,:]
          part_map = cv2.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
          y,x = np.unravel_index(part_map.argmax(),part_map.shape)
          ret.append( [x,y] )
        #self.disp_map(person_image.copy(), ret)
        write_list.append( {'id':imgdt_list[i]['path'],'p':ret} )
    if write:
      write_path = 'out/' + img_path.split('/')[-1].rsplit('_',1)[0] + '.json'
      print 'writing to %s' % write_path
      json.dump(write_list, open(write_path ,'w'))
    
    self.length.append( len(write_list) )


  def plot_len_dist(self):
    from matplotlib import pyplot as plt
    plt.hist(self.length, bins=50)
    plt.savefig('len_hist.png')
    


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
    joints = joints.astype(int)
    for j in joints:
      cv2.circle(img, (j[0], j[1]), 3, (0,0,255), -1)
    cv2.imwrite('test.png',img) 
