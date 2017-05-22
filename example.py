import pdb
import poserecog.pipe as pipe
import glob

config_path = 'model/model.config'
piper = pipe.Pipeline(config_path)

pdb.set_trace()
path = []
base_path = '/home/gengshan/workJan/poseEstm_rtpose/camData/'
for cls in glob.glob(base_path+'/*'):
  for subcls in glob.glob( cls+'/*'):
    path += glob.glob( subcls+'/*/')
pdb.set_trace()

for it in path:
  it = it[:-1];print it
  piper.process(it)
