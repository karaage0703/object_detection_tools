# Object Detection Tools
This repository is useful tools for TensorFlow Object Detection API.

# Only Demo
　For only demo. Setup Python3.0, TensorFlow.
 
  Then execute following commnads, you can get object detection demo on Mac/Linux PC/Jetson Nano/Raspberry Pi.

```sh
$ cd && git clone https://github.com/karaage0703/object_detection_tools
$ cd ~/object_detection_tools/models
$ ./get_ssdlite_mobilenet_v2_coco_model.sh
$ cd ~/object_detection_tools
$ python3 scripts/object_detection.py -l='models/coco-labels-paper.txt' -m='models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
```

# Setup
Setup Python3.0 and TensorFlow environment.

And get TensorFlow Models repository.

Execute following commands for download TensorFlow Object Detection API and change directory:
```sh
$ git clone https://github.com/tensorflow/models
$ cd models/research
```

Go to `models/research` directory

# Usage

## Download this repository
Execute following command:
```sh
$ git clone https://github.com/karaage0703/object_detection_tools
```

## Model download
Change directory `object_detection_tools/models` and execute download script for downloading model file.

For example:

```sh
$ ./get_ssd_inception_v2_coco_model.sh
```

## Test Prediction
Execute following commands at `object_detection_tools` after downloading ssd_inception_v2_coco_model data:
```sh
$ python scripts/object_detection.py -l='models/coco-labels-paper.txt' -m='models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
```

## Train

## Annotate data
Using VoTT is recommended.

Export tfrecord data.

## Convert tf record file name
Put tfrecord data `./data/train` and `./data/val` directory.

Then, execute following command at `object_detection_tools/data` directory:

```sh
$ ./change_tfrecord_filename.sh
```

## Train Models
SSD inception v2 example(fine tuning)

Change directory `object_detection_tools/models` and execute download script for downloading model file:
```sh
$ ./get_ssd_inception_v2
```

Execute following commands for training model:
```sh
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection/model_main.py --pipeline_config_path="./object_detection_tools/config/ssd_inception_v2_coco.config" --model_dir="./saved_model_01" --num_train_steps=1000 --alsologtostderr
```

notice: `model_dir` must be empty before training

## Convert Model
Convert from ckpt to graph file.

Execute following commands for converting from ckpt to graph file:
```sh
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path object_detection_tools/config/ssd_inception_v2_coco.config --trained_checkpoint_prefix saved_model_01/model.ckpt-1000 --output_directory exported_graphs
```

## Convert Label
Convert from pbtxt data to label data.

Execute follwing commands for converting from pbtxt data to label data:
```sh
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection_tools/scripts/convert_pbtxt_label.py -l='object_detection_tools/data/tf_labl_map.pbtxt' > ./exported_graphs/labels.txt
```

## Test trained model
Execute following command for testing trained model:
```sh
$ python object_detection_tools/scripts/object_detection.py -l='./exported_graphs/labels.txt' -m='./exported_graphs/frozen_inference_graph.pb'
```

# License
This software is released under the Apache 2.0 License, see LICENSE.


# References
- https://github.com/tensorflow/models
- https://github.com/tsutof/tf-trt-ssd
