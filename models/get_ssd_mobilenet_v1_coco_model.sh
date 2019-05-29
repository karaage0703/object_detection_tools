#!/bin/sh
if [ ! -e coco-labels-paper.txt ]; then
    wget https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt
fi
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
rm ssd_mobilenet_v1_coco_2018_01_28.tar.gz