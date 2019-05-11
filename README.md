# Object Detection Tools
This repository is useful tools for TensorFlow Object Detection API.

# Setup
Setup Python3.0 and TensorFlow environment.

And get TensorFlow Models repository.

Execute following commands for download TensorFlow Object Detection API and change directory:
```sh
$ git clone https://github.com/tensorflow/models
$ cd models/research
```

Go ahead under `models/research` directory

# Usage

## Download this repository

```sh
$ git clone https://github.com/karaage0703/object_detection_tools
$ ln -sf $PWD/object_detection_tools/scripts/object_detection_tutorial.py $PWD/object_detection/object_detection_tutorial.py
```

## Model download
Change directory `models` and execute download scripts.

For example:

```sh
$ cd models
$ ./get_ssd_inception_v2_coco_model.sh
```

## Test Prediction
Execute following commands at `models/research` after downloading ssd_inception_v2_coco_model data:
```sh
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection/object_detection_tutorial.py -l='object_detection/data/mscoco_label_map.pbtxt' -m='object_detection_tools/models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
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


### ssd inception v2 example
Download ssd model
```sh
$ cd models
$ ./get_ssd_inception_v2
$ cd ..
```

Train model
```sh
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection/model_main.py --pipeline_config_path="./object_detection_tools/config/ssd_inception_v2_coco.config" --model_dir="./saved_model_01" --num_train_steps=1000 --alsologtostderr
```

### faster rcnn example
```sh
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection/model_main.py --pipeline_config_path="./object_detection_tools/config/faster_rcnn_resnet101_pets.config" --model_dir="./saved_model_01" --num_train_steps=1000 --alsologtostderr
```

## Convert Model
Convert from ckpt to graph file:

### ssd inception v2 example
```sh
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path object_detection_tools/config/ssd_inception_v2_coco.config --trained_checkpoint_prefix saved_model_01/model.ckpt-1000 --output_directory exported_graphs
```

## Test trained models

```sh
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection/object_detection_tutorial.py -l='./object_detection_tools/data/tf_label_map.pbtxt' -m='./exported_graphs/frozen_inference_graph.pb'
```

# License
This software is released under the Apache 2.0 License, see LICENSE.


# References
- https://github.com/tensorflow/models