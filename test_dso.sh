#!/bin/sh
seq=${1}

./build/bin/dso_dataset \
	files=/home/wyw/lyh/data/slam/$seq/image_0 \
	calib=/home/wyw/lyh/data/slam/camera.txt \
	gamma=/home/wyw/lyh/data/slam/pcalib.txt \
	vignette=/home/wyw/lyh/data/slam/vignette.png \
	preset=0 \
	mode=1\
	quiet=1
