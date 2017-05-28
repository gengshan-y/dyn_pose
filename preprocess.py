import os
import pdb
from poserecog.util import recReadDir
import subprocess

cate_pos = 5
fps = 5
suffix = '.avi'
bin_path = '/usr/local/ffmpeg/bin/'
out_path = 'dataset/weizmann/'

vid_list = recReadDir(['/data/gengshan/har/weizmann/'], contain = '.avi')
keys = list(set([x.split('/')[cate_pos] for x in vid_list]))
class_map = {}
for c in keys:
  class_map[c] = [v for v in vid_list if c == v.split('/')[cate_pos]]

print 'has %d categories' % len(keys)
print keys
print class_map[keys[0]]


for k in keys:
  for v in class_map[k]:
    n = v.split(suffix)[0].split('/')[-1]
    img_path = '%s/%s_%s_%s' % (out_path, k, n, '%04d.jpg')
    cmd = '%s/ffmpeg -i %s -r %d %s' % (bin_path, v, fps, img_path)
    a = subprocess.check_output(cmd, shell = True, stderr=subprocess.STDOUT)
    print a
