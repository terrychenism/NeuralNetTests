# pylint: skip-file
from data import ilsvrc12_iterator
import mxnet as mx
import logging
workspace_default = 1024
input_data = mx.symbol.Variable(name="data")
# group 1
conv1_1 = mx.symbol.Convolution(
    data=input_data, kernel=(7, 7), pad=(3, 3), stride=(2,2), num_filter=64, name="conv1_1")
relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
# conv1_2 = mx.symbol.Convolution(
#     data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
# relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
pool1 = mx.symbol.Pooling(
    data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
# group 2
conv2_1 = mx.symbol.Convolution(
    data=pool1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=128, name="conv2_1")
relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
conv2_2 = mx.symbol.Convolution(
    data=relu2_1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=128, name="conv2_2")
relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
conv2_3 = mx.symbol.Convolution(
    data=relu2_2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=128, name="conv2_3")
relu2_3 = mx.symbol.Activation(data=conv2_3, act_type="relu", name="relu2_3")
conv2_4 = mx.symbol.Convolution(
    data=relu2_3, kernel=(1, 1), stride=(1, 1), num_filter=256, name="conv2_4")
relu2_4 = mx.symbol.Activation(data=conv2_4, act_type="relu", name="relu2_4")

pool2 = mx.symbol.Pooling(
    data=relu2_4, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
# group 3
conv3_1 = mx.symbol.Convolution(
    data=pool2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="conv3_1")
relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
conv3_2 = mx.symbol.Convolution(
    data=relu3_1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="conv3_2")
relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
conv3_3 = mx.symbol.Convolution(
    data=relu3_2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="conv3_3")
relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
conv3_4 = mx.symbol.Convolution(
    data=relu3_3, kernel=(1, 1), stride=(1, 1), num_filter=512, name="conv3_4")
relu3_4 = mx.symbol.Activation(data=conv3_4, act_type="relu", name="relu3_4")
pool3 = mx.symbol.Pooling(
    data=relu3_4, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2,2), name="pool3")
# group 4
conv4_1 = mx.symbol.Convolution(
    data=pool3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=512, name="conv4_1")
relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
conv4_2 = mx.symbol.Convolution(
    data=relu4_1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=512, name="conv4_2")
relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
conv4_3 = mx.symbol.Convolution(
    data=relu4_2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=512, name="conv4_3")
relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
conv4_4 = mx.symbol.Convolution(
    data=relu4_3, kernel=(1, 1), stride=(1, 1), num_filter=1024, name="conv4_4")
relu4_4 = mx.symbol.Activation(data=conv4_4, act_type="relu", name="relu4_4")
pool4 = mx.symbol.Pooling(
    data=relu4_4, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2,2), name="pool4")
# group 5
conv5_1 = mx.symbol.Convolution(
    data=pool4, kernel=(1, 1), stride=(1, 1), num_filter=1024, name="conv5_1")
relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
conv5_2 = mx.symbol.Convolution(
    data=relu5_1, kernel=(1, 1), stride=(1, 1), num_filter=1024, name="conv5_2")
relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
# conv5_3 = mx.symbol.Convolution(
#     data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
# relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
pool5 = mx.symbol.Pooling(
    data=relu5_2, pool_type="ave", kernel=(7, 7), stride=(1,1), name="pool5")
drop5 = mx.symbol.Dropout(data=pool5, p=0.5, name="drop5")
# group 6
conv6 = mx.symbol.Convolution(
    data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=401, name="conv6")
softmax = mx.symbol.SoftmaxOutput(data=conv6, name='softmax')


## data
batch_size = 16
train, val = ilsvrc12_iterator(batch_size=batch_size, input_shape=(3,224,224))

## train
num_gpus = 1
gpus = [mx.gpu(i) for i in range(num_gpus)]
model = mx.model.FeedForward(
    ctx           = gpus,
    symbol        = softmax,
    num_epoch     = 20,
    learning_rate = 0.01,
    momentum      = 0.9,
    wd            = 0.00001)
logging.basicConfig(level = logging.DEBUG)
model.fit(X = train, eval_data = val,
          batch_end_callback = mx.callback.Speedometer(batch_size=batch_size))
