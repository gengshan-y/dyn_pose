import os
import pdb
from poserecog.util import recReadDir
import subprocess

cate_pos = -2
fps = 5
suffix = '.avi'
bin_path = '/usr/local/ffmpeg/bin/'
in_path = '/data/gengshan/har/weizmann/'
out_path = 'dataset/weizmann/' # 'dataset/pose_vid/' #'dataset/pose/'

vid_list = recReadDir([in_path], contain = suffix)
keys = list(set([x.split('/')[cate_pos] for x in vid_list]))
pdb.set_trace()
class_map = {}
for c in keys:
  class_map[c] = [v for v in vid_list if c == v.split('/')[cate_pos]]



from sklearn.model_selection import train_test_split
train_l = [];val_l=[]
for k in keys:
  lbs = ['%s_%s'%(k,x.split(suffix)[0].split('/')[-1]) for x in class_map[k]]
  train,val = train_test_split(lbs,test_size=0.3)
  train_l += train; val_l += val

pdb.set_trace()
import json
with open('split-weiz.json','w') as f:
  json.dump({'train':train_l,'val':val_l}, f)
