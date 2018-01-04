#!/bin/sh
data_home=/home/wyw/lyh/data/slam
seq=${1}
pose_file=${2}
echo $data_home
./build/bin/dso_show_result \
	files=$data_home/$seq/image_0 \
	calib=$data_home/camera.txt \
	gamma=$data_home/pcalib.txt \
	datadir=$data_home/$seq \
	posefile=$pose_file \
	vignette=$data_home/vignette.png \
	preset=0 \
	mode=1\
	quiet=1
