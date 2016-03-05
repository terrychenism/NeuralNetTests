import os
CAFFE_ROOT = '../../'
if CAFFE_ROOT is not None:
    import sys
    sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
print "Importing caffe ..."

import caffe
import scipy.io
import numpy as np
from PIL import Image
import cv2
from skimage.io import imread, imsave
from skimage.color import label2rgb
from skimage import img_as_ubyte
from PIL import Image as PILImage

cmap = scipy.io.loadmat('./voc_gt_cmap.mat')['cmap']

# Network definitions
net_def = 'parsernn.deploy'
weights = 'snapshots/parsernn_iter_45000.caffemodel'

# Load Network
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(net_def, weights, caffe.TEST)

# Load Image
# im = Image.open('cat.jpg')
IMAGE_FILE = 'cat.jpg'
input_image = 255 * caffe.io.load_image(IMAGE_FILE)


width = input_image.shape[0]
height = input_image.shape[1]
maxDim = max(width,height)

image = PILImage.fromarray(np.uint8(input_image))
image = np.array(image)

mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

# Rearrange channels to form BGR
im = image[:,:,::-1]
# Subtract mean
im = im - reshaped_mean_vec


# Pad as necessary
cur_h, cur_w, cur_c = im.shape
pad_h = 500 - cur_h
pad_w = 500 - cur_w
im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

# im = np.array(im, dtype=np.float32)
# im = im[:,:,::-1]               # Change to BGR
# mean = np.array((104.00698793,116.66876762,122.67891434))
# im -= mean     # Mean Subtraction
im = im.transpose(2,0,1)        # Blob: C x H x W
im = im[None,:,:,:]

# Assign Data
# net.blobs['data'].reshape(*im.shape)
net.blobs['data'].data[...] = im

# Run forward
net.forward()

res = net.blobs['score'].data[0].argmax(axis=0)
print res

mat_predicted = np.asarray(res)
print net.blobs['score'].data.shape
# cv2.imshow("cropped", mat_predicted)
# cv2.waitKey()
# cv2.imwrite('result.jpg',mat_predicted)

segmentation = mat_predicted[0:cur_h, 0:cur_w]

imsave('result.jpg', label2rgb(segmentation, colors=cmap))


# Classes Predicted
print 'Classes Predicted:', np.unique(net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8))
print 'Result saved'
