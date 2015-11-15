# pylint: skip-file
from data import ilsvrc12_iterator
import mxnet as mx
import logging

def get_vgg16_symbol(nhidden):
    data = mx.symbol.Variable(name="data")
    conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=False)
    relu1 = mx.symbol.Activation(name='relu1', data=conv1 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
    conv2_3 = mx.symbol.Convolution(name='conv2_3', data=relu2_2 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu2_3 = mx.symbol.Activation(name='relu2_3', data=conv2_3 , act_type='relu')
    conv2_4 = mx.symbol.Convolution(name='conv2_4', data=relu2_3 , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu2_4 = mx.symbol.Activation(name='relu2_4', data=conv2_4 , act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
    conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3 , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4 , act_type='relu')
    pool3 = mx.symbol.Pooling(name='pool3', data=relu3_4 , pad=(1,1), kernel=(3,3), stride=(2,2), pool_type='max')
    
    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
    conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3 , act_type='relu')
    conv4_4 = mx.symbol.Convolution(name='conv4_4', data=relu4_3 , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu4_4 = mx.symbol.Activation(name='relu4_4', data=conv4_4 , act_type='relu')
    pool4 = mx.symbol.Pooling(name='pool4', data=relu4_4 , pad=(1,1), kernel=(3,3), stride=(2,2), pool_type='max')
    
    conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4 , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1 , act_type='relu')
    conv5_2 = mx.symbol.Convolution(name='conv5_2', data=relu5_1 , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu5_2 = mx.symbol.Activation(name='relu5_2', data=conv5_2 , act_type='relu')
    pool5 = mx.symbol.Pooling(name='pool5', data=relu5_2 , pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
    drop5 = mx.symbol.Dropout(name='drop5', data=pool5 , p=0.500000)
    
    flatten=mx.symbol.Flatten(name='flatten', data=drop5)
    fc8 = mx.symbol.FullyConnected(name='fc8', data=flatten, num_hidden=nhidden, no_bias=False)
    softmax = mx.symbol.SoftmaxOutput(data=fc8, name='softmax')
    return softmax
    
softmax = get_vgg16_symbol(1000)  
## data
batch_size = 64
train, val = ilsvrc12_iterator(batch_size=batch_size, input_shape=(3,224,224))
model_prefix = "model/VGG"
## train
num_gpus = 1
gpus = [mx.gpu(i) for i in range(num_gpus)]
model = mx.model.FeedForward(
    ctx           = gpus,
    symbol        = softmax,
    num_epoch     = 20,
    learning_rate = 0.01,
    momentum      = 0.9,
    wd            = 0.0002)

logging.basicConfig(level = logging.DEBUG)
model.fit(X = train, eval_data = val,
          eval_metric="acc",
          batch_end_callback=[mx.callback.Speedometer(batch_size), mx.callback.log_train_metric(50)],
          epoch_end_callback=mx.callback.do_checkpoint(model_prefix))
