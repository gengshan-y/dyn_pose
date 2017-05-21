import numpy as np
from cStringIO import StringIO
import scipy.ndimage

def get_maxima(dmap):
  data_max = scipy.ndimage.filters.maximum_filter(dmap, 3)
  y,x = np.unravel_index(np.argmax(data_max), data_max.shape)
  #maxima = (dmap == data_max)
  #diff = (data_max > 0.5)
  #maxima[diff == 0] = 0
  #x = np.nonzero(maxima)[1]
  #y = np.nonzero(maxima)[0]
  return x,y

def showBGRimage(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    a[:,:,[0,2]] = a[:,:,[2,0]] # for B,G,R order
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor(gray_img[y,x], 0, 1)
    return out

def padRightDownCorner(img):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%8==0) else 8 - (h % 8) # down
    pad[3] = 0 if (w%8==0) else 8 - (w % 8) # right
    
    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + 128, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]* + 128, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]* + 128, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + 128, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)): 
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
    return c

def recReadDir(baseDir, contain = ''):
    """ read filenames recursively """
    # no dir to expand, return
    if len([x for x in baseDir if os.path.isdir(x)]) == 0:
        # filter before return results
        baseDir = [x for x in baseDir if contain in x]
        return baseDir
    # expand dirs
    out = []
    for it in baseDir:
        if os.path.isdir(it):
            # expand dirs
            out += [os.path.abspath(os.path.join(it,x)) for x in os.listdir(it)]
        elif contain in it:
            # filter one more time to make efficient
            out += [it]
    return recReadDir(out, contain)
