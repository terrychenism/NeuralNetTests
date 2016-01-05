#!/bin/bash

CAFFE_ROOT=../..

for f in snapshots/*.caffemodel; 
do 
	echo "Processing $f file.."; 
	$CAFFE_ROOT/build/tools/caffe test \
	--gpu=0 \
	--model=train_val_v3.prototxt \
	--iterations=500 \
	--weights=$f \
	2>&1 | tee log_test.txt
	echo $f >> logs.txt
	tail -n 2 log_test.txt >> logs.txt
done

