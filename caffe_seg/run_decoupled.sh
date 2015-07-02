#!/usr/bin/env sh

rm -rf log

TOOLS=../../build/tools

GLOG_logtostderr=1 

"../../bin/Decoupled.exe" \
pixel.txt \
F:/Coding/DecoupledNet/model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference_deploy_raw.prototxt \
F:/Coding/DecoupledNet/model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference.caffemodel \
F:/Coding/DecoupledNet/inference/data/VOC2012_TEST/JPEGImages/2008_000048.jpg
