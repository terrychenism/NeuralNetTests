# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import numpy as np
import caffe

caffe.set_mode_gpu()
net_full_conv = caffe.Net('net_surgery/inception_v3_conv.prototxt', 
		                          'net_surgery/inception_v3_conv.caffemodel',
		                          caffe.TEST)
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

filename = '/home/tairuic/Downloads/caffe_pkg/caffe/data/ilsvrc12/val.txt'

top1_acc = 0
top5_acc = 0
cnt = 0
with open(filename,"r") as f:
    for line in f:
    	cnt += 1
    	# print cls
        img_name, label = line.split()
        #print top5_acc, top1_acc
        img_full_name = '/home/tairuic/Downloads/caffe_pkg/caffe/data/ilsvrc12/ILSVRC2012_img_val/' + img_name
        im = caffe.io.load_image(img_full_name)	

        transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
        transformer.set_mean('data', np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255.0)

        out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

        #print out['prob'][0].argmax(axis=0)
        a = out['prob'].mean(axis=2)
        score = a.mean(axis=2)
        sort_index = np.argsort(score)[:,::-1]
        top_k = sort_index[:,0:5]

        #print int(label), int(top_k[:,0])
        if int(top_k[:,0]) == int(label):
        	top1_acc = top1_acc + 1

       	for pred in np.nditer(top_k):
        	#print pred
        	if int(pred) == int(label):
        		top5_acc = top5_acc + 1
        		break

       	#print "top1 acc: %.5f, top5 acc: %.5f, cnt: %d" % (top1_acc, top5_acc, cnt)
        print "num: %d, top1 acc: %.5f, top5 acc: %.5f" % (cnt, top1_acc/float(cnt), top5_acc/float(cnt))
        
        #print "top 5: ", top_k
        #print labels[top_k-1]


print "final top1 acc: %.3f, top5 acc: %.3f" % (top1_acc/500, top5_acc/500)		

# a = out['prob'].sum(axis=2)
# score = a.sum(axis=2)
# sort_index = np.argsort(score)[:,::-1]
# top_k = sort_index[:,0:5]
# print "top 5: ", top_k
# print labels[top_k-1]




