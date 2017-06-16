import pdb
import poserecog.pipe as pipe
import glob

config_path = 'model/model.config'
base_path = 'dataset/pose'

pdb.set_trace()
piper = pipe.Pipeline(config_path)

img_list = glob.glob(base_path+'/*.jpg')
scls = set([x.split('/')[-1].rsplit('_',1)[0] for x in img_list])
print '%d videos' % len(scls)

pdb.set_trace()
#scls=['up_IPC_20170529121800']
pdb.set_trace()
for c in scls:
  path =  '%s/%s_*.jpg' % (base_path,c)
  print path
  piper.reset()
  piper.process(path)
