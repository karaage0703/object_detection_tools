#!/bin/sh
if [ ! -e coco-labels-paper.txt ]; then
    wget https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt
fi
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_coco17_tpu-8.tar.gz
tar xvzf centernet_hg104_512x512_coco17_tpu-8.tar.gz
rm centernet_hg104_512x512_coco17_tpu-8.tar.gz