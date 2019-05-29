#!/bin/sh
if [ ! -e coco-labels-paper.txt ]; then
    wget https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt
fi
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar xvzf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
rm ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz