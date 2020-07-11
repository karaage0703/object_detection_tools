#!/bin/sh
if [ ! -e coco-labels-paper.txt ]; then
    wget https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt
fi
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
tar xvzf efficientdet_d0_coco17_tpu-32.tar.gz
rm efficientdet_d0_coco17_tpu-32.tar.gz