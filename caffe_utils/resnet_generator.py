#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('solver_file',
                        help='Output solver.prototxt file')
    parser.add_argument('train_val_file',
                        help='Output train_val.prototxt file')
    parser.add_argument('-l', '--layer_number', nargs='*',type=int,
                        help=('Layer number for each layer stage.'),
                        default=[3, 8, 36, 3])
    parser.add_argument('-t', '--type', type=int,
                        help=('0 for deploy.prototxt, 1 for train_val.prototxt.'),
                        default=1)

    args = parser.parse_args()
    return args

def generate_data_layer():
    data_layer_str = '''name: "ResNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 224
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "/mnt/data/imagenet/ilsvrc12_train_lmdb"
    batch_size: 8
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 224
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "/mnt/data/imagenet/ilsvrc12_val_lmdb"
    batch_size: 5
    backend: LMDB
  }
}'''
    return data_layer_str

def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra"):
    conv_layer_str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
     lr_mult: 2
     decay_mult: 0
  }
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
    weight_filler {
      type: "%s"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
'''%(layer_name, bottom, top, kernel_num, pad, kernel_size, stride, filler)
    return conv_layer_str

def generate_pooling_layer(kernel_size, stride, pool_type, layer_name, bottom, top):
    pool_layer_str = '''layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}
'''%(layer_name, bottom, top, pool_type, kernel_size, stride)
    return pool_layer_str

def generate_fc_layer(num_output, layer_name, bottom, top, filler="msra"):
    fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
     num_output: %d
     weight_filler {
       type: "%s"
       std: 0.001
     }
     bias_filler {
       type: "constant"
       value: 0
     }
  }
}
'''%(layer_name, bottom, top, num_output, filler)
    return fc_layer_str

def generate_eltwise_layer(layer_name, bottom_1, bottom_2, top, op_type="SUM"):
    eltwise_layer_str = '''layer {
  name: "%s"
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  eltwise_param {
    operation: %s
  }
}
'''%(layer_name, bottom_1, bottom_2, top, op_type)
    return eltwise_layer_str

def generate_activation_layer(layer_name, bottom, top, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  top: "%s"
}
'''%(layer_name, act_type, bottom, top)
    return act_layer_str

def generate_softmax_loss(bottom):
    softmax_loss_str = '''layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss"
}
layer {
  name: "acc/top-1"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "acc/top-1"
  include {
    phase: TEST
  }
}
layer {
  name: "acc/top-5"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "acc/top-5"
  include {
    phase: TEST
  }
  accuracy_param {
    top_k: 5
  }
}
'''%(bottom, bottom, bottom)
    return softmax_loss_str

def generate_bn_layer(layer_name, bottom, top):
    bn_layer_str = '''layer {
  name: "%s"
  type: "BatchNorm"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }
}
'''%(layer_name, bottom, top)
    return bn_layer_str

def generate_train_val():
    args = parse_args()
    network_str = generate_data_layer()
    '''before stage'''
    last_top = 'data'
    network_str += generate_conv_layer(7, 64, 2, 0, 'conv1', last_top, 'conv1')
    network_str += generate_bn_layer('conv1_bn', 'conv1', 'conv1_bn')
    network_str += generate_activation_layer('conv1_relu', 'conv1_bn', 'conv1_bn', 'ReLU')
    network_str += generate_pooling_layer(3, 2, 'MAX', 'pool1', 'conv1_bn', 'pool1')
    '''stage 1'''
    last_top = 'pool1'
    network_str += generate_conv_layer(1, 256, 1, 0, 'conv1_output', last_top, 'conv1_output')
    last_output = 'conv1_output'
    for l in xrange(1, args.layer_number[0]+1):
        network_str += generate_conv_layer(1, 64, 1, 0, 'conv2_%d_1'%l, last_top, 'conv2_%d_1'%l)
        network_str += generate_bn_layer('conv2_%d_1_bn'%l, 'conv2_%d_1'%l, 'conv2_%d_1_bn'%l)
        network_str += generate_activation_layer('conv2_%d_1_relu'%l, 'conv2_%d_1_bn'%l, 'conv2_%d_1_bn'%l, 'ReLU')
        network_str += generate_conv_layer(3, 64, 1, 1, 'conv2_%d_2'%l, 'conv2_%d_1_bn'%l, 'conv2_%d_2'%l)
        network_str += generate_bn_layer('conv2_%d_2_bn'%l, 'conv2_%d_2'%l, 'conv2_%d_2_bn'%l)
        network_str += generate_activation_layer('conv2_%d_2_relu'%l, 'conv2_%d_2_bn'%l, 'conv2_%d_2_bn'%l, 'ReLU')
        network_str += generate_conv_layer(1, 256, 1, 0, 'conv2_%d_3'%l, 'conv2_%d_2_bn'%l, 'conv2_%d_3'%l)
        network_str += generate_eltwise_layer('conv2_%d_sum'%l, last_output, 'conv2_%d_3'%l, 'conv2_%d_sum'%l, 'SUM')
        network_str += generate_bn_layer('conv2_%d_sum_bn'%l, 'conv2_%d_sum'%l, 'conv2_%d_sum_bn'%l)
        network_str += generate_activation_layer('conv2_%d_sum_relu'%l, 'conv2_%d_sum_bn'%l, 'conv2_%d_sum_bn'%l, 'ReLU')
        last_top = 'conv2_%d_sum_bn'%l
        last_output = 'conv2_%d_sum_bn'%l
    network_str += generate_conv_layer(1, 512, 2, 0, 'conv2_output', last_top, 'conv2_output')
    last_output = 'conv2_output'
    '''stage 2'''
    network_str += generate_conv_layer(1, 128, 2, 0, 'conv3_1_1', last_top, 'conv3_1_1')
    network_str += generate_bn_layer('conv3_1_1_bn', 'conv3_1_1', 'conv3_1_1_bn')
    network_str += generate_activation_layer('conv3_1_1_relu', 'conv3_1_1_bn', 'conv3_1_1_bn', 'ReLU')
    network_str += generate_conv_layer(3, 128, 1, 1, 'conv3_1_2', 'conv3_1_1_bn', 'conv3_1_2')
    network_str += generate_bn_layer('conv3_1_2_bn', 'conv3_1_2', 'conv3_1_2_bn')
    network_str += generate_activation_layer('conv3_1_2_relu', 'conv3_1_2_bn', 'conv3_1_2_bn', 'ReLU')
    network_str += generate_conv_layer(1, 512, 1, 0, 'conv3_1_3', 'conv3_1_2_bn', 'conv3_1_3')
    network_str += generate_eltwise_layer('conv3_1_sum', last_output, 'conv3_1_3', 'conv3_1_sum', 'SUM')
    network_str += generate_bn_layer('conv3_1_sum_bn', 'conv3_1_sum', 'conv3_1_sum_bn')
    network_str += generate_activation_layer('conv3_1_sum_relu', 'conv3_1_sum_bn', 'conv3_1_sum_bn', 'ReLU')
    last_top = 'conv3_1_sum_bn'
    for l in xrange(2, args.layer_number[1]+1):
        network_str += generate_conv_layer(1, 128, 1, 0, 'conv3_%d_1'%l, last_top, 'conv3_%d_1'%l)
        network_str += generate_bn_layer('conv3_%d_1_bn'%l, 'conv3_%d_1'%l, 'conv3_%d_1_bn'%l)
        network_str += generate_activation_layer('conv3_%d_1_relu'%l, 'conv3_%d_1_bn'%l, 'conv3_%d_1_bn'%l, 'ReLU')
        network_str += generate_conv_layer(3, 128, 1, 1, 'conv3_%d_2'%l, 'conv3_%d_1_bn'%l, 'conv3_%d_2'%l)
        network_str += generate_bn_layer('conv3_%d_2_bn'%l, 'conv3_%d_2'%l, 'conv3_%d_2_bn'%l)
        network_str += generate_activation_layer('conv3_%d_2_relu'%l, 'conv3_%d_2_bn'%l, 'conv3_%d_2_bn'%l, 'ReLU')
        network_str += generate_conv_layer(1, 512, 1, 0, 'conv3_%d_3'%l, 'conv3_%d_2_bn'%l, 'conv3_%d_3'%l)
        network_str += generate_eltwise_layer('conv3_%d_sum'%l, last_top, 'conv3_%d_3'%l, 'conv3_%d_sum'%l, 'SUM')
        network_str += generate_bn_layer('conv3_%d_sum_bn'%l, 'conv3_%d_sum'%l, 'conv3_%d_sum_bn'%l)
        network_str += generate_activation_layer('conv3_%d_sum_relu'%l, 'conv3_%d_sum_bn'%l, 'conv3_%d_sum_bn'%l, 'ReLU')
        last_top = 'conv3_%d_sum_bn'%l
    network_str += generate_conv_layer(1, 1024, 2, 0, 'conv3_output', last_top, 'conv3_output')
    last_output = 'conv3_output'
    '''stage 3'''
    network_str += generate_conv_layer(1, 256, 2, 0, 'conv4_1_1', last_top, 'conv4_1_1')
    network_str += generate_bn_layer('conv4_1_1_bn', 'conv4_1_1', 'conv4_1_1_bn')
    network_str += generate_activation_layer('conv4_1_1_relu', 'conv4_1_1_bn', 'conv4_1_1_bn', 'ReLU')
    network_str += generate_conv_layer(3, 256, 1, 1, 'conv4_1_2', 'conv4_1_1_bn', 'conv4_1_2')
    network_str += generate_bn_layer('conv4_1_2_bn', 'conv4_1_2', 'conv4_1_2_bn')
    network_str += generate_activation_layer('conv4_1_2_relu', 'conv4_1_2_bn', 'conv4_1_2_bn', 'ReLU')
    network_str += generate_conv_layer(1, 1024, 1, 0, 'conv4_1_3', 'conv4_1_2_bn', 'conv4_1_3')
    network_str += generate_eltwise_layer('conv4_1_sum', last_output, 'conv4_1_3', 'conv4_1_sum', 'SUM')
    network_str += generate_bn_layer('conv4_1_sum_bn', 'conv4_1_sum', 'conv4_1_sum_bn')
    network_str += generate_activation_layer('conv4_1_sum_relu', 'conv4_1_sum_bn', 'conv4_1_sum_bn', 'ReLU')
    last_top = 'conv4_1_sum_bn'
    for l in xrange(2, args.layer_number[2]+1):
        network_str += generate_conv_layer(1, 256, 1, 0, 'conv4_%d_1'%l, last_top, 'conv4_%d_1'%l)
        network_str += generate_bn_layer('conv4_%d_1_bn'%l, 'conv4_%d_1'%l, 'conv4_%d_1_bn'%l)
        network_str += generate_activation_layer('conv4_%d_1_relu'%l, 'conv4_%d_1_bn'%l, 'conv4_%d_1_bn'%l, 'ReLU')
        network_str += generate_conv_layer(3, 256, 1, 1, 'conv4_%d_2'%l, 'conv4_%d_1_bn'%l, 'conv4_%d_2'%l)
        network_str += generate_bn_layer('conv4_%d_2_bn'%l, 'conv4_%d_2'%l, 'conv4_%d_2_bn'%l)
        network_str += generate_activation_layer('conv4_%d_2_relu'%l, 'conv4_%d_2_bn'%l, 'conv4_%d_2_bn'%l, 'ReLU')
        network_str += generate_conv_layer(1, 1024, 1, 0, 'conv4_%d_3'%l, 'conv4_%d_2_bn'%l, 'conv4_%d_3'%l)
        network_str += generate_eltwise_layer('conv4_%d_sum'%l, last_top, 'conv4_%d_3'%l, 'conv4_%d_sum'%l, 'SUM')
        network_str += generate_bn_layer('conv4_%d_sum_bn'%l, 'conv4_%d_sum'%l, 'conv4_%d_sum_bn'%l)
        network_str += generate_activation_layer('conv4_%d_sum_relu'%l, 'conv4_%d_sum_bn'%l, 'conv4_%d_sum_bn'%l, 'ReLU')
        last_top = 'conv4_%d_sum_bn'%l
    network_str += generate_conv_layer(1, 2048, 2, 0, 'conv4_output', last_top, 'conv4_output')
    last_output = 'conv4_output'
    '''stage 4'''
    network_str += generate_conv_layer(1, 512, 2, 0, 'conv5_1_1', last_top, 'conv5_1_1')
    network_str += generate_bn_layer('conv5_1_1_bn', 'conv5_1_1', 'conv5_1_1_bn')
    network_str += generate_activation_layer('conv5_1_1_relu', 'conv5_1_1_bn', 'conv5_1_1_bn', 'ReLU')
    network_str += generate_conv_layer(3, 512, 1, 1, 'conv5_1_2', 'conv5_1_1_bn', 'conv5_1_2')
    network_str += generate_bn_layer('conv5_1_2_bn', 'conv5_1_2', 'conv5_1_2_bn')
    network_str += generate_activation_layer('conv5_1_2_relu', 'conv5_1_2_bn', 'conv5_1_2_bn', 'ReLU')
    network_str += generate_conv_layer(1, 2048, 1, 0, 'conv5_1_3', 'conv5_1_2_bn', 'conv5_1_3')
    network_str += generate_eltwise_layer('conv5_1_sum', last_output, 'conv5_1_3', 'conv5_1_sum', 'SUM')
    network_str += generate_bn_layer('conv5_1_sum_bn', 'conv5_1_sum', 'conv5_1_sum_bn')
    network_str += generate_activation_layer('conv5_1_sum_relu', 'conv5_1_sum_bn', 'conv5_1_sum_bn', 'ReLU')
    last_top = 'conv5_1_sum_bn'
    for l in xrange(2, args.layer_number[3]+1):
        network_str += generate_conv_layer(1, 512, 1, 0, 'conv5_%d_1'%l, last_top, 'conv5_%d_1'%l)
        network_str += generate_bn_layer('conv5_%d_1_bn'%l, 'conv5_%d_1'%l, 'conv5_%d_1_bn'%l)
        network_str += generate_activation_layer('conv5_%d_1_relu'%l, 'conv5_%d_1_bn'%l, 'conv5_%d_1_bn'%l, 'ReLU')
        network_str += generate_conv_layer(3, 512, 1, 1, 'conv5_%d_2'%l, 'conv5_%d_1_bn'%l, 'conv5_%d_2'%l)
        network_str += generate_bn_layer('conv5_%d_2_bn'%l, 'conv5_%d_2'%l, 'conv5_%d_2_bn'%l)
        network_str += generate_activation_layer('conv5_%d_2_relu'%l, 'conv5_%d_2_bn'%l, 'conv5_%d_2_bn'%l, 'ReLU')
        network_str += generate_conv_layer(1, 2048, 1, 0, 'conv5_%d_3'%l, 'conv5_%d_2_bn'%l, 'conv5_%d_3'%l)
        network_str += generate_eltwise_layer('conv5_%d_sum'%l, last_top, 'conv5_%d_3'%l, 'conv5_%d_sum'%l, 'SUM')
        network_str += generate_bn_layer('conv5_%d_sum_bn'%l, 'conv5_%d_sum'%l, 'conv5_%d_sum_bn'%l)
        network_str += generate_activation_layer('conv5_%d_sum_relu'%l, 'conv5_%d_sum_bn'%l, 'conv5_%d_sum_bn'%l, 'ReLU')
        last_top = 'conv5_%d_sum_bn'%l
    network_str += generate_pooling_layer(7, 1, 'AVE', 'pool2', last_top, 'pool2')
    network_str += generate_fc_layer(1000, 'fc', 'pool2', 'fc', 'gaussian')
    network_str += generate_softmax_loss('fc')
    return network_str

def generate_solver(train_val_name):
    solver_str = '''net: "%s"
test_iter: 1000
test_interval: 6000
test_initialization: false
display: 60
base_lr: 0.1
lr_policy: "multistep"
stepvalue: 300000
stepvalue: 500000
gamma: 0.1
max_iter: 600000
momentum: 0.9
weight_decay: 0.0001
snapshot: 6000
snapshot_prefix: "resnet"
solver_mode: GPU'''%(train_val_name)
    return solver_str

def main():
  
    """
    python ResNet_Generator.py -t 1 solver.prototxt train_val.prototxt -l 3 4 6 3
    """
    args = parse_args()
    solver_str = generate_solver(args.train_val_file)
    network_str = generate_train_val()
    fp = open(args.solver_file, 'w')
    fp.write(solver_str)
    fp.close()
    fp = open(args.train_val_file, 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()

    
