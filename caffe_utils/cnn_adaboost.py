#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Tairui Chen


import numpy as np
import os
import sys
import argparse
import glob
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

import caffe

g_rnd = np.random.randint(100000)

def create_weighted_db(X, y, weights, name='boost'):
    X = X.reshape(-1, 3, 32, 32)
    train_fn = os.path.join(DIR, name + '.h5')

    dd.io.save(train_fn, dict(data=X,
                              label=y.astype(np.float32),
                              sample_weight=weights), compress=False)
    with open(os.path.join(DIR, name + '.txt'), 'w') as f:
        print(train_fn, file=f)


class CNN(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def get_params(self, deep=False):
        return {}

    def fit(self, X, y, sample_weight=None):
        global g_seed
        global g_loop
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], np.float32)
            print('Calling fit with sample_weight None')
        else:
            sample_weight *= X.shape[0]
            print('Calling fit with sample_weight sum', sample_weight.sum())

        #sample_weight = np.ones(X.shape[0], np.float32)

        #II = sample_weight > 0
        #X = X[II]
        #y = y[II]
        #sample_weight = sample_weight[II]

        #sample_weight = np.ones(X.shape[0])
        w = sample_weight
        #sample_weight[:10] = 0.0
        #w[:1000] = 0.0
        #w = sample_weight
        #w0 = w / w.sum()
        #print('Weight entropy:', -np.sum(w0 * np.log2(w0)))
        print('Weight max:', w.max())
        print('Weight min:', w.min())
        #import sys; sys.exit(0)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Set up weighted database
        create_weighted_db(X, y, sample_weight)

        #steps = [(0.001, 2000, 2000)]
        steps = [(0.001, 0.004, 60000), (0.0001, 0.004, 5000), (0.00001, 0.004, 5000)]
        #steps = [(0.00001, 10000, 10000), (0.000001, 5000, 15000), (0.0000001, 5000, 20000)]
        #steps = [(0.001, 10000, 10000)]
        #steps = [(0.001, 200, 1000)]

        name = os.path.join(CONF_DIR, 'adaboost_{}_loop{}'.format(g_rnd, g_loop))
        bare_conf_fn = os.path.join(CONF_DIR, 'boost_bare.prototxt')
        conf_fn = os.path.join(CONF_DIR, 'solver.prototxt.template')
        #bare_conf_fn = 'regaug_bare.prototxt'
        #conf_fn = 'regaug_solver.prototxt.template'

        net, info = train_model(name, conf_fn, bare_conf_fn, steps,
                                seed=g_seed, device_id=DEVICE_ID)

        loss_fn = 'info/info_{}_loop{}.h5'.format(g_rnd, g_loop)
        dd.io.save(loss_fn, info)
        print('Saved to', loss_fn)

        g_loop += 1

        print('Classifier set up')

        self.net_ = net

    def predict_proba(self, X):
        X = X.reshape(-1, 3, 32, 32)
        #X = X.transpose(0, 2, 3, 1)
        prob = np.zeros((X.shape[0], self.n_classes_))

        M = 2500
        for k in range(int(np.ceil(X.shape[0] / M))):
            y = self.net_.forward_all(data=X[k*M:(k+1)*M]).values()[0].squeeze(axis=(2,3))
            prob[k*M:(k+1)*M] = y

        T = 30.0

        eps = 0.0001

        #prob = prob.clip(eps, 1-eps)

        log_prob = np.log(prob)
        print('log_prob', log_prob.min(), log_prob.max())
        #log_prob = log_prob.clip(min=-4, max=4)
        new_prob = np.exp(log_prob / T)
        new_prob /= dd.apply_once(np.sum, new_prob, [1])

        return new_prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return prob.argmax(-1)




train_data = np.load('G:/EDU/_SOURCE_CODE/chainer/examples/cifar10/data/train_data.npy')
train_labels = np.load('G:/EDU/_SOURCE_CODE/chainer/examples/cifar10/data/train_labels.npy')

model_path = 'cifar10/' # substitute your path here
# GoogleNet
net_fn   = model_path + 'VGG_mini_ABN.prototxt'
param_fn = model_path + 'cifar10_vgg_iter_120000.caffemodel'

caffe.set_mode_cpu()
net = caffe.Classifier(net_fn, param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB


def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']


for i in range(10):
	img = train_data[i].transpose((1, 2, 0)) * 255
	img = img.astype(np.uint8)[:, :, ::-1]
	end = 'prob'
	h, w = img.shape[:2]
	src, dst = net.blobs['data'], net.blobs[end]
	src.data[0] = preprocess(net, img)
	net.forward(end=end)
	features = dst.data[0].copy()
 
 
X = train_data
y = train_labels
X *= 255.0
mean_x = X.mean(0)
X -= mean_x

te_X= np.load('G:/EDU/_SOURCE_CODE/chainer/examples/cifar10/data/test_data.npy')
te_y = np.load('G:/EDU/_SOURCE_CODE/chainer/examples/cifar10/data/test_labels.npy')

create_weighted_db(te_X, te_y, np.ones(te_X.shape[0], dtype=np.float32), name='test')  

clf = AdaBoostClassifier(base_estimator=CNN(), algorithm='SAMME.R', n_estimators=10,
                                 random_state=0)
clf.fit(X.reshape(X.shape[0], -1), y)

for i, score in enumerate(clf.staged_score(X.reshape(X.shape[0], -1), y)):
                print(i+1, 'train score', score)

for i, score in enumerate(clf.staged_score(te_X.reshape(te_X.shape[0], -1), te_y)):
                print(i+1, 'test score', score)
