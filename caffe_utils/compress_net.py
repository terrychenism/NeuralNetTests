import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import Image
from scipy import stats

import os
CAFFE_ROOT = '/home/tairuic/Downloads/caffe_pkg/caffe_work/caffe-crop-eval/'
if CAFFE_ROOT is not None:
    import sys
    sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))

import caffe




# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

base_model = 'VGG_ILSVRC_16_layers_deploy.prototxt'
base_weights = 'VGG_ILSVRC_16_layers.caffemodel'
base_net = caffe.Net(base_model, base_weights, caffe.TRAIN)

voc_model = 'VGG_ILSVRC_16_layers_deploy.prototxt'
voc_weights = 'VGG_ILSVRC_16_layers_compress.caffemodel'  # where result is going to go
voc_net = caffe.Net(voc_model, caffe.TEST)

# Source and destination paramteres, these are the same because the layers
# have the same names in base_net, voc_net
src_params = ['fc6', 'fc7','fc8']  # ignore fc8 because it will be re initialized
dest_params = ['fc6', 'fc7', 'fc8']

# First: copy shared layers
shared_layers = set(base_net.params.keys()) & set(voc_net.params.keys())
# shared_layers -= set(src_params + dest_params)

for layer in sorted(list(shared_layers)):
    print("Copying shared layer",layer)
    print base_net.params[layer][0].data.shape
    # print base_net.params[layer][0].data
    low_values_indices = abs(base_net.params[layer][0].data) < 3e-03 
    base_net.params[layer][0].data[low_values_indices] = 0
    non_zero_cnt = np.count_nonzero(base_net.params[layer][0].data)
    shape = base_net.params[layer][0].data.shape
    if 'fc' in layer:
        total_cnt = shape[0] * shape[1]
    else:
        total_cnt = shape[0] * shape[1] * shape[2] * shape[3]
    print 1-float(non_zero_cnt)/total_cnt

    # print base_net.params[layer][0].data

    voc_net.params[layer][0].data[...] = base_net.params[layer][0].data
    voc_net.params[layer][1].data[...] = base_net.params[layer][1].data

# Second: copy over the fully connected layers
# fc_params = {name: (weights, biases)}

# fc_params = {}
# for pr in src_params:
#     fc_params[pr] = (base_net.params[pr][0].data, base_net.params[pr][1].data)

# # conv_params = {name: (weights, biases)}
# conv_params = {}
# for pr in dest_params:
#     conv_params[pr] = (voc_net.params[pr][0].data, voc_net.params[pr][1].data)

# for pr, pr_conv in zip(src_params, dest_params):
#     print('(source) {} weights are {} dimensional and biases are {} dimensional'\
#       .format(pr, fc_params[pr][0].shape, fc_params[pr][1].shape))
#     print('(destn.) {} weights are {} dimensional and biases are {} dimensional'\
#       .format(pr_conv, conv_params[pr_conv][0].shape, conv_params[pr_conv][1].shape))

#     conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
#     conv_params[pr_conv][1][...] = fc_params[pr][1]

voc_net.save(voc_weights)


