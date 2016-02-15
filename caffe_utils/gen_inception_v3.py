filename = 'train_val.body'
f = open(filename, 'w')

def Conv(data, num_filter, kernel=1, stride=1, pad=0, name=None, suffix=''):
    path=addConv(bottom=data, top='%s%s' %(name, suffix), outChannels=num_filter, kernel=kernel, stride=stride, pad=pad)
    path=addBatchNorm(bottom=path, top='%s%s_bn' %(name, suffix))
    path=addScale(bottom=path, top='%s%s_bn_sc' %(name, suffix))
    path=addReLU(bottom=path)
    return path

def Conv2(data, num_filter, kernel_h=1, kernel_w=1, stride=1, pad_h=0,pad_w=0,name=None, suffix=''):
    path=addConv2(bottom=data, top='%s%s' %(name, suffix), outChannels=num_filter, kernel_h=kernel_h, kernel_w=kernel_w, stride=stride, pad_h=pad_h,pad_w=pad_w)
    path=addBatchNorm(bottom=path, top='%s%s_bn' %(name, suffix))
    path=addScale(bottom=path, top='%s%s_bn_sc' %(name, suffix))
    path=addReLU(bottom=path)
    return path

def addLinear(bottom, top, num_output): 
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}"\n'.format(top));
    f.write('  name: "{}"\n'.format(top));
    f.write('  type: "InnerProduct"\n');
    f.write('  param {\n');
    f.write('    lr_mult: 1\n');
    f.write('    decay_mult: 1\n');
    f.write('  }\n');
    f.write('  inner_product_param {\n');
    f.write('    num_output: {}\n'.format(num_output));
    f.write('    weight_filler {\n');
    f.write('      type: "xavier"\n');
    f.write('    }\n');
    f.write('    bias_term: false\n');
    f.write('  }\n');
    f.write('}\n');
    return top

def addLoss(bottom, top, weight):
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  bottom: "label"\n');
    f.write('  top: "{}/loss"\n'.format(top));
    f.write('  name: "{}/loss"\n'.format(top));
    f.write('  type: "SoftmaxWithLoss"\n');
    f.write('  loss_weight: {}\n'.format(weight));
    f.write('}\n');
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}/prob"\n'.format(top));
    f.write('  name: "{}/prob"\n'.format(top));
    f.write('  type: "Softmax"\n');
    f.write('  include {\n');
    f.write('    phase: TEST\n');
    f.write('  }\n');
    f.write('}\n');
    f.write('layer {\n');
    f.write('  bottom: "{}/prob"\n'.format(top));
    f.write('  bottom: "label"\n');
    f.write('  top: "{}/top-1"\n'.format(top));
    f.write('  name: "{}/top-1"\n'.format(top));
    f.write('  type: "Accuracy"\n');
    f.write('  include {\n');
    f.write('    phase: TEST\n');
    f.write('  }\n');
    f.write('}\n');
    f.write('layer {\n');
    f.write('  bottom: "{}/prob"\n'.format(top));
    f.write('  bottom: "label"\n');
    f.write('  top: "{}/top-5"\n'.format(top));
    f.write('  name: "{}/top-5"\n'.format(top));
    f.write('  type: "Accuracy"\n');
    f.write('  accuracy_param {\n');
    f.write('    top_k: 5\n');
    f.write('  }\n');
    f.write('  include {\n');
    f.write('    phase: TEST\n');
    f.write('  }\n');
    f.write('}\n');
    return

def addReLU(bottom):
   #bottom = 'inception_5b/pool_proj/bn'
   f.write('layer {\n');
   f.write('  bottom: "{}"\n'.format(bottom));
   f.write('  top: "{}"\n'.format(bottom));
   f.write('  name: "{}/relu"\n'.format(bottom));
   f.write('  type: "ReLU"\n');
   f.write('}\n');
   return bottom 

def addPooling(bottom, top, kernel, stride=1, pad=0, pool='MAX'):
   #bottom = 'conv1/7x7_s2/sc' 
   #top = 'pool1/3x3_s2'
   #pool = 'MAX'
   f.write('layer {\n');
   f.write('  bottom: "{}"\n'.format(bottom));
   f.write('  top: "{}"\n'.format(top));
   f.write('  name: "{}"\n'.format(top));
   f.write('  type: "Pooling"\n');
   f.write('  pooling_param {\n');
   f.write('    pool: {}\n'.format(pool));
   f.write('    kernel_size: {}\n'.format(kernel));
   f.write('    stride: {}\n'.format(stride));
   if pad > 0:
       f.write('    pad: {}\n'.format(pad));
   f.write('  }\n');
   f.write('}\n');
   return top

def addConv(bottom, top, outChannels, kernel, stride=1, pad=0):
   #bottom = 'data'
   #top = 'conv1/7x7_s2'
   f.write('layer {\n');
   f.write('  bottom: "{}"\n'.format(bottom));
   f.write('  top: "{}"\n'.format(top));
   f.write('  name: "{}"\n'.format(top));
   f.write('  type: "Convolution"\n');
   f.write('  param {\n');
   f.write('    lr_mult: 1\n');
   f.write('    decay_mult: 1\n');
   f.write('  }\n');
   f.write('  convolution_param {\n');
   f.write('    num_output: {}\n'.format(outChannels));
   f.write('    pad: {}\n'.format(pad));
   f.write('    kernel_size: {}\n'.format(kernel));
   f.write('    stride: {}\n'.format(stride));
   f.write('    weight_filler {\n');
   f.write('      type: "msra"\n');
   f.write('    }\n');
   f.write('    bias_term: false\n');
   f.write('  }\n');
   f.write('}\n');
   return top 


def addConv2(bottom, top, outChannels, kernel_h, kernel_w, stride=1, pad_h=0,pad_w=0):
   #bottom = 'data'
   #top = 'conv1/7x7_s2'
   f.write('layer {\n');
   f.write('  bottom: "{}"\n'.format(bottom));
   f.write('  top: "{}"\n'.format(top));
   f.write('  name: "{}"\n'.format(top));
   f.write('  type: "Convolution"\n');
   f.write('  param {\n');
   f.write('    lr_mult: 1\n');
   f.write('    decay_mult: 1\n');
   f.write('  }\n');
   f.write('  convolution_param {\n');
   f.write('    num_output: {}\n'.format(outChannels));
   f.write('    pad_h: {}\n'.format(pad_h));
   f.write('    pad_w: {}\n'.format(pad_w));
   f.write('    kernel_h: {}\n'.format(kernel_h));
   f.write('    kernel_w: {}\n'.format(kernel_w));
   f.write('    stride: {}\n'.format(stride));
   f.write('    weight_filler {\n');
   f.write('      type: "xavier"\n');
   f.write('    }\n');
   f.write('    bias_term: false\n');
   f.write('  }\n');
   f.write('}\n');
   return top

def addBatchNorm(bottom, top): 
    #bottom = 'conv1/7x7_s2'
    #top = 'conv1/7x7_s2/bn'
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  name: "{}"\n'.format(top));
    f.write('  top: "{}"\n'.format(top));
    f.write('  type: "BatchNorm"\n');
    #f.write('  batch_norm_param {\n');
    #f.write('    use_global_stats: true\n');
    #f.write('  }\n');
    f.write('}\n');
    return top 

def addScale(bottom, top): 
    #bottom = 'conv1/7x7_s2/bn'
    #top = 'conv1/7x7_s2/sc'
    f.write('layer {\n');
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}"\n'.format(top));
    f.write('  name: "{}"\n'.format(top));
    f.write('  type: "Scale"\n');
    f.write('  scale_param {\n');
    f.write('    bias_term: true\n');
    f.write('  }\n');
    f.write('}\n');
    return top 

def addConcat(bottoms, top):
    f.write('layer {\n');
    for bottom in bottoms:
        f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}"\n'.format(top));
    f.write('  name: "{}"\n'.format(top));
    f.write('  type: "Concat"\n');
    f.write('}\n');
    return top

def addInception(bottom, prefix, num_out_1x1, num_out_3x3_reduce, num_out_3x3, num_out_double3x3_reduce, num_out_double3x3, num_out_pool, pool, stride):
   #3a:  64,     64,  64,     64,  96,     32, 'AVE',   1
   #3b:  64,     64,  96,     64,  96,     64, 'AVE',   1
   #3c:   0,    128, 160,     64,  96,      0, 'MAX',   2

   #4a: 224,     64,  96,     96, 128,    128, 'AVE',   1
   #4b: 192,     96, 128,     96, 128,    128, 'AVE',   1
   #4c: 160,    128, 160,    128, 160,     96, 'AVE',   1
   #4d:  96,    128, 192,    160, 192,     96, 'AVE',   1
   #4e:   0,    128, 192,    192, 256,      0, 'MAX',   2

   #5a: 352,    192, 320,    160, 224,    128, 'AVE',   1
   #5b: 352,    192, 320,    192, 224,    128, 'MAX',   1

   # 1x1
   if num_out_1x1 > 0: 
       top1=addConv(bottom=bottom, top='inception_{}/1x1'.format(prefix), outChannels=num_out_1x1, kernel=1, stride=stride, pad=0)
       top1=addBatchNorm(bottom=top1, top='{}/bn'.format(top1))
       top1=addScale(bottom=top1, top='{}/sc'.format(top1))
       top1=addReLU(bottom=top1)
      
   # 3x3
   top2=addConv(bottom=bottom, top='inception_{}/3x3_reduce'.format(prefix), outChannels=num_out_3x3_reduce, kernel=1, stride=1, pad=0)
   top2=addBatchNorm(bottom=top2, top='{}/bn'.format(top2))
   top2=addScale(bottom=top2, top='{}/sc'.format(top2))
   top2=addReLU(bottom=top2)

   top2=addConv(bottom=top2, top='inception_{}/3x3'.format(prefix), outChannels=num_out_3x3, kernel=3, stride=stride, pad=1)
   top2=addBatchNorm(bottom=top2, top='{}/bn'.format(top2))
   top2=addScale(bottom=top2, top='{}/sc'.format(top2))
   top2=addReLU(bottom=top2)

   # double 3x3
   top3=addConv(bottom=bottom, top='inception_{}/double3x3_reduce'.format(prefix), outChannels=num_out_double3x3_reduce, kernel=1, stride=1, pad=0)
   top3=addBatchNorm(bottom=top3, top='{}/bn'.format(top3))
   top3=addScale(bottom=top3, top='{}/sc'.format(top3))
   top3=addReLU(bottom=top3)

   top3=addConv(bottom=top3, top='inception_{}/double3x3a'.format(prefix), outChannels=num_out_double3x3, kernel=3, stride=1, pad=1)
   top3=addBatchNorm(bottom=top3, top='{}/bn'.format(top3))
   top3=addScale(bottom=top3, top='{}/sc'.format(top3))
   top3=addReLU(bottom=top3)

   top3=addConv(bottom=top3, top='inception_{}/double3x3b'.format(prefix), outChannels=num_out_double3x3, kernel=3, stride=stride, pad=1)
   top3=addBatchNorm(bottom=top3, top='{}/bn'.format(top3))
   top3=addScale(bottom=top3, top='{}/sc'.format(top3))
   top3=addReLU(bottom=top3)

   # pool projection
   if stride == 1:
       top4=addPooling(bottom=bottom, top='inception_{}/pool'.format(prefix), kernel=3, stride=stride, pad=1, pool=pool) 
   elif stride == 2:
       top4=addPooling(bottom=bottom, top='inception_{}/pool'.format(prefix), kernel=3, stride=stride, pad=0, pool=pool) 
   else: 
       raise ValueError('stride is either 1 or 2. in inception layer.') 
   if num_out_pool > 0:
       top4=addConv(bottom=top4, top='inception_{}/pool_proj'.format(prefix), outChannels=num_out_pool, kernel=1, stride=1, pad=0)
       top4=addBatchNorm(bottom=top4, top='{}/bn'.format(top4))
       top4=addScale(bottom=top4, top='{}/sc'.format(top4))
       top4=addReLU(bottom=top4)

   # concat
   tops = [top2, top3, top4]
   if num_out_1x1 > 0:
       tops.insert(0, top1)
   top=addConcat(bottoms=tops, top='inception_{}/output'.format(prefix))

   return top 

def addClassifier(bottom, top, num_output):
    f.write('layer {\n'); 
    f.write('  bottom: "{}"\n'.format(bottom));
    f.write('  top: "{}"\n'.format(top));
    f.write('  name: "{}"\n'.format(top));
    f.write('  type: "InnerProduct"\n');
    f.write('  param {\n');
    f.write('    lr_mult: 1\n');
    f.write('    decay_mult: 1\n');
    f.write('  }\n');
    f.write('  param {\n');
    f.write('    lr_mult: 2\n');
    f.write('    decay_mult: 0\n');
    f.write('  }\n');
    f.write('  inner_product_param {\n');
    f.write('    num_output: {}\n'.format(num_output));
    f.write('    weight_filler {\n');
    f.write('      type: "xavier"\n');
    f.write('    }\n');
    f.write('    bias_filler {\n');
    f.write('      type: "constant"\n');
    f.write('      value: 0\n');
    f.write('    }\n');
    f.write('  }\n');
    f.write('}\n');
    return top


def Inception7A(data,
                num_1x1,
                num_3x3_red, num_3x3_1, num_3x3_2,
                num_5x5_red, num_5x5,
                pool, proj,
                name):
    tower_1x1 = Conv(data, num_1x1, name=('%s_conv' % name))
    tower_5x5 = Conv(data, num_5x5_red, name=('%s_tower' % name), suffix='_conv')
    tower_5x5 = Conv(tower_5x5, num_5x5, kernel=5, pad=2, name=('%s_tower' % name), suffix='_conv_1')
    tower_3x3 = Conv(data, num_3x3_red, name=('%s_tower_1' % name), suffix='_conv')
    tower_3x3 = Conv(tower_3x3, num_3x3_1, kernel=3, pad=1, name=('%s_tower_1' % name), suffix='_conv_1')
    tower_3x3 = Conv(tower_3x3, num_3x3_2, kernel=3, pad=1, name=('%s_tower_1' % name), suffix='_conv_2')
    pooling=addPooling(bottom=data, top=('%s_pool_%s_pool' % (pool, name)), kernel=3, stride=1, pad=1, pool='MAX')
    cproj = Conv(pooling, proj, name=('%s_tower_2' %  name), suffix='_conv')
    tops = [tower_1x1, tower_5x5, tower_3x3, cproj]
    concat=addConcat(bottoms=tops, top='ch_concat_%s_chconcat' % name)
    return concat


# First Downsample
def Inception7B(data,
                num_3x3,
                num_d3x3_red, num_d3x3_1, num_d3x3_2,
                pool,
                name):
    tower_3x3 = Conv(data, num_3x3, kernel=3, pad=0, stride=2, name=('%s_conv' % name))
    tower_d3x3 = Conv(data, num_d3x3_red, name=('%s_tower' % name), suffix='_conv')
    tower_d3x3 = Conv(tower_d3x3, num_d3x3_1, kernel=3, pad=1, stride=1, name=('%s_tower' % name), suffix='_conv_1')
    tower_d3x3 = Conv(tower_d3x3, num_d3x3_2, kernel=3, pad=0, stride=2, name=('%s_tower' % name), suffix='_conv_2')
    pooling=addPooling(bottom=data, top=('max_pool_%s_pool' % name), kernel=3, stride=2, pad=0, pool='MAX')
    tops = [tower_3x3, tower_d3x3, pooling]
    concat=addConcat(bottoms=tops, top='ch_concat_%s_chconcat' % name)
    return concat


def Inception7C(data,
                num_1x1,
                num_d7_red, num_d7_1, num_d7_2,
                num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4,
                pool, proj,
                name):
    tower_1x1 = Conv(data=data, num_filter=num_1x1, kernel=1, name=('%s_conv' % name))
    tower_d7 = Conv(data=data, num_filter=num_d7_red, name=('%s_tower' % name), suffix='_conv')
    tower_d7 = Conv2(data=tower_d7, num_filter=num_d7_1, kernel_h=1, kernel_w=7, pad_h=0,pad_w=3, name=('%s_tower' % name), suffix='_conv_1')
    tower_d7 = Conv2(data=tower_d7, num_filter=num_d7_2, kernel_h=7, kernel_w=1, pad_h=3,pad_w=0, name=('%s_tower' % name), suffix='_conv_2')
    tower_q7 = Conv2(data=data, num_filter=num_q7_red, name=('%s_tower_1' % name), suffix='_conv')
    tower_q7 = Conv2(data=tower_q7, num_filter=num_q7_1, kernel_h=7, kernel_w=1, pad_h=3,pad_w=0, name=('%s_tower_1' % name), suffix='_conv_1')
    tower_q7 = Conv2(data=tower_q7, num_filter=num_q7_2, kernel_h=1, kernel_w=7, pad_h=0,pad_w=3, name=('%s_tower_1' % name), suffix='_conv_2')
    tower_q7 = Conv2(data=tower_q7, num_filter=num_q7_3, kernel_h=7, kernel_w=1, pad_h=3,pad_w=0, name=('%s_tower_1' % name), suffix='_conv_3')
    tower_q7 = Conv2(data=tower_q7, num_filter=num_q7_4, kernel_h=1, kernel_w=7, pad_h=0,pad_w=3, name=('%s_tower_1' % name), suffix='_conv_4')

    pooling=addPooling(bottom=data, top=('%s_pool_%s_pool' % (pool, name)), kernel=3, stride=1, pad=1, pool='MAX')
    cproj = Conv(pooling, proj, name=('%s_tower_2' %  name), suffix='_conv')
    tops = [tower_1x1, tower_d7, tower_q7, cproj]
    concat=addConcat(bottoms=tops, top='ch_concat_%s_chconcat' % name)
    return concat

def Inception7D(data,
                num_3x3_red, num_3x3,
                num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3,
                pool,
                name):
    tower_3x3 = Conv(data=data, num_filter=num_3x3_red, name=('%s_tower' % name), suffix='_conv')
    tower_3x3 = Conv(data=tower_3x3, num_filter=num_3x3, kernel=3, pad=0, stride=2, name=('%s_tower' % name), suffix='_conv_1')
    tower_d7_3x3 = Conv(data=data, num_filter=num_d7_3x3_red, name=('%s_tower_1' % name), suffix='_conv')
    tower_d7_3x3 = Conv2(data=tower_d7_3x3, num_filter=num_d7_1, kernel_h=1, kernel_w=7, pad_h=0,pad_w=3, name=('%s_tower_1' % name), suffix='_conv_1')
    tower_d7_3x3 = Conv2(data=tower_d7_3x3, num_filter=num_d7_2, kernel_h=7, kernel_w=1, pad_h=3,pad_w=0, name=('%s_tower_1' % name), suffix='_conv_2')
    tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=num_d7_3x3, kernel=3, stride=2, name=('%s_tower_1' % name), suffix='_conv_3')
    pooling=addPooling(bottom=data, top=('%s_pool_%s_pool' % (pool, name)), kernel=3, stride=2, pad=0, pool='MAX')
    tops = [tower_3x3, tower_d7_3x3, pooling]
    concat=addConcat(bottoms=tops, top='ch_concat_%s_chconcat' % name)
    return concat


def Inception7E(data,
                num_1x1,
                num_d3_red, num_d3_1, num_d3_2,
                num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2,
                pool, proj,
                name):
    tower_1x1 = Conv(data=data, num_filter=num_1x1, kernel=1, name=('%s_conv' % name))
    tower_d3 = Conv(data=data, num_filter=num_d3_red, name=('%s_tower' % name), suffix='_conv')
    tower_d3_a = Conv2(data=tower_d3, num_filter=num_d3_1, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1, name=('%s_tower' % name), suffix='_mixed_conv')
    tower_d3_b = Conv2(data=tower_d3, num_filter=num_d3_2, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0, name=('%s_tower' % name), suffix='_mixed_conv_1')
    tower_3x3_d3 = Conv(data=data, num_filter=num_3x3_d3_red, name=('%s_tower_1' % name), suffix='_conv')
    tower_3x3_d3 = Conv(data=tower_3x3_d3, num_filter=num_3x3, kernel=3, pad=1, name=('%s_tower_1' % name), suffix='_conv_1')
    tower_3x3_d3_a = Conv2(data=tower_3x3_d3, num_filter=num_3x3_d3_1, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1, name=('%s_tower_1' % name), suffix='_mixed_conv')
    tower_3x3_d3_b = Conv2(data=tower_3x3_d3, num_filter=num_3x3_d3_2, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0, name=('%s_tower_1' % name), suffix='_mixed_conv_1')

    pooling=addPooling(bottom=data, top=('%s_pool_%s_pool' % (pool, name)), kernel=3, stride=1, pad=1, pool='MAX')
    cproj = Conv(pooling, proj, name=('%s_tower_2' %  name), suffix='_conv')
    tops = [tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj]
    concat=addConcat(bottoms=tops, top='ch_concat_%s_chconcat' % name)
    return concat



# main
# stage 1
conv=Conv(data='data', num_filter=32, kernel=3, stride=2,name='conv' )
conv_1=Conv(data=conv, num_filter=32, kernel=3, name='conv_1')
conv_2=Conv(data=conv_1, num_filter=64, kernel=3, pad=1,name='conv_2')
pool=addPooling(bottom=conv_2, top='pool', kernel=3, stride=2, pad=0, pool='MAX')

# stage 2
conv_3=Conv(data=pool, num_filter=80, kernel=1, pad=1, name='conv_3') #299 pad:0
conv_4=Conv(data=conv_3, num_filter=192, kernel=3, name='conv_4')
pool1=addPooling(bottom=conv_4, top='poo11', kernel=3, stride=2, pad=0, pool='MAX')



# stage 3
in3a = Inception7A(pool1, 64,
                   64, 96, 96,
                   48, 64,
                   "avg", 32, "mixed")
in3b = Inception7A(in3a, 64,
                   64, 96, 96,
                   48, 64,
                   "avg", 64, "mixed_1")
in3c = Inception7A(in3b, 64,
                   64, 96, 96,
                   48, 64,
                   "avg", 64, "mixed_2")
in3d = Inception7B(in3c, 384,
                   64, 96, 96,
                   "max", "mixed_3")
# stage 4
in4a = Inception7C(in3d, 192,
                   128, 128, 192,
                   128, 128, 128, 128, 192,
                   "avg", 192, "mixed_4")
in4b = Inception7C(in4a, 192,
                   160, 160, 192,
                   160, 160, 160, 160, 192,
                   "avg", 192, "mixed_5")
in4c = Inception7C(in4b, 192,
                   160, 160, 192,
                   160, 160, 160, 160, 192,
                   "avg", 192, "mixed_6")
in4d = Inception7C(in4c, 192,
                   192, 192, 192,
                   192, 192, 192, 192, 192,
                   "avg", 192, "mixed_7")
in4e = Inception7D(in4d, 192, 320,
                   192, 192, 192, 192,
                   "max", "mixed_8")
# stage 5
in5a = Inception7E(in4e, 320,
                   384, 384, 384,
                   448, 384, 384, 384,
                   "avg", 192, "mixed_9")
in5b = Inception7E(in5a, 320,
                   384, 384, 384,
                   448, 384, 384, 384,
                   "max", 192, "mixed_10")



# main classifier
pool=addPooling(bottom=in5b, top="global_pool", kernel=6, stride=1, pad=0, pool='AVE')#299 kernel=7
fc1000=addClassifier(bottom=pool, top='fc1000', num_output=1000)
addLoss(bottom=fc1000, top='loss', weight=1)


