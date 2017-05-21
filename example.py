import pdb
import poserecog.pipe as pipe
import cv2

pdb.set_trace()

config_path = 'model/model.config'
piper = pipe.Pipeline(config_path)

img = cv2.imread('./test/test.jpg')
ret = piper.process(img)
