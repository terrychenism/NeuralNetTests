# -*- coding: utf-8 -*-
"""
Created on Thu May 15 09:46:15 2014

@author: Terry

Function: convolution
"""

from theano.tensor.nnet import conv
import theano.tensor as T
import numpy, theano


rng = numpy.random.RandomState(23455)


input = T.tensor4(name = 'input')

w_shape = (2,3,9,9) 
w_bound = numpy.sqrt(3*9*9)
W = theano.shared(numpy.asarray(rng.uniform(low = -1.0/w_bound, high = 1.0/w_bound,size = w_shape),
                                dtype = input.dtype),name = 'W')

b_shape = (2,)
b = theano.shared(numpy.asarray(rng.uniform(low = -.5, high = .5, size = b_shape),
                                dtype = input.dtype),name = 'b')
                                
conv_out = conv.conv2d(input,W)


output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
f = theano.function([input],output)





import pylab
from PIL import Image
from matplotlib.pyplot import *

img = Image.open('C://Users//Terry//Desktop//img2.jpg')
width,height = img.size
img = numpy.asarray(img, dtype = 'float32')/256. # (height, width, 3)


img_rgb = img.swapaxes(0,2).swapaxes(1,2) 
minibatch_img = img_rgb.reshape(1,3,height,width)
filtered_img = f(minibatch_img)

pylab.figure(1)
pylab.subplot(1,3,1);pylab.axis('off');
pylab.imshow(img)
title('origin image')

pylab.gray()
pylab.subplot(2,3,2); pylab.axis("off")
pylab.imshow(filtered_img[0,0,:,:]) #0:minibatch_index; 0:1-st filter
title('convolution 1')

pylab.subplot(2,3,3); pylab.axis("off")
pylab.imshow(filtered_img[0,1,:,:]) #0:minibatch_index; 1:1-st filter
title('convolution 2')



# maxpooling
from theano.tensor.signal import downsample

input = T.tensor4('input')
maxpool_shape = (2,2)
pooled_img = downsample.max_pool_2d(input,maxpool_shape,ignore_border = False)

maxpool = theano.function(inputs = [input],
                          outputs = [pooled_img])

pooled_res = numpy.squeeze(maxpool(filtered_img))              
#pylab.figure(2)
pylab.subplot(235);pylab.axis('off');
pylab.imshow(pooled_res[0,:,:])
title('down sampled 1')

pylab.subplot(236);pylab.axis('off');
pylab.imshow(pooled_res[1,:,:])
title('down sampled 2')

pylab.show()

