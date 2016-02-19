import numpy as np
import sys
import os
import os.path as osp
import google.protobuf as pb
from argparse import ArgumentParser
caffe_root = '/home/caffe-bn/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


def main(args):
    # Set default output file names
    if args.output_model is None:
        file_name = osp.splitext(args.model)[0]
        args.output_model = file_name + '_inference.prototxt'
    if args.output_weights is None:
        file_name = osp.splitext(args.weights)[0]
        args.output_weights = file_name + '_inference.caffemodel'
    with open(args.model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Parse(f.read(), model)

    # Get the BN layers to be absorbed
    to_be_absorbed = []
    for i, layer in enumerate(model.layer):
        if layer.type != 'BN': continue
        bottom = layer.bottom[0]
        top = layer.top[0]
        can_be_absorbed = True
        for j in xrange(i - 1, -1, -1):
            bottom_layer = model.layer[j]
            if bottom in bottom_layer.top:
                if bottom_layer.type not in ['Convolution', 'InnerProduct']:
                    can_be_absorbed = False
        if can_be_absorbed:
            to_be_absorbed.append(layer.name)
            # Rename the top blobs
            for j in xrange(i + 1, len(model.layer)):
                top_layer = model.layer[j]
                if top in top_layer.bottom:
                    names = list(top_layer.bottom)
                    names[names.index(top)] = bottom
                    del(top_layer.bottom[:])
                    top_layer.bottom.extend(names)
                if top in top_layer.top:
                    names = list(top_layer.top)
                    names[names.index(top)] = bottom
                    del(top_layer.top[:])
                    top_layer.top.extend(names)
        else:
            # Freeze the BN layer
            # layer.bn_param.frozen = True
            del(layer.param[:])
            param = caffe.proto.caffe_pb2.ParamSpec()
            param.lr_mult = 0
            param.decay_mult = 0
            layer.param.extend([param] * 2)

    # Save the prototxt
    output_model_layers = [layer for layer in model.layer
                           if layer.name not in to_be_absorbed]
    output_model = caffe.proto.caffe_pb2.NetParameter()
    output_model.CopyFrom(model)
    del(output_model.layer[:])
    output_model.layer.extend(output_model_layers)
    with open(args.output_model, 'w') as f:
        f.write(pb.text_format.MessageToString(output_model))

    # Absorb the BN parameters
    weights = caffe.Net(args.model, args.weights, caffe.TEST)
    for i, layer in enumerate(model.layer):
        if layer.name not in to_be_absorbed: continue
        scale, bias, mean, var = [p.data.ravel()
                                     for p in weights.params[layer.name]]
        eps = 1e-5
        invstd = 1./np.sqrt( var + eps )
        for j in xrange(i - 1, -1, -1):
            bottom_layer = model.layer[j]
            if layer.bottom[0] in bottom_layer.top:
                W, b = weights.params[bottom_layer.name]
                num = W.data.shape[0]
                W.data[...] = (W.data * scale[:, None, None, None]
                                      * invstd[:, None, None, None])
                b.data[...] = (b.data[...] - mean) * scale * invstd + bias

    # Save the caffemodel
    output_weights = caffe.Net(args.output_model, caffe.TEST)
    for name in output_weights.params:
        for i in xrange(len(output_weights.params[name])):
            output_weights.params[name][i].data[...] = weights.params[name][i].data.copy()
    output_weights.save(args.output_weights)


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Generate Batch Normalized model for inference")
    parser.add_argument('model', help="The net definition prototxt")
    parser.add_argument('weights', help="The weights caffemodel")
    parser.add_argument('--output_model')
    parser.add_argument('--output_weights')
    args = parser.parse_args()
    main(args)
