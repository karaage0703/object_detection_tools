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
$ cp object_detection_tools/scripts/object_detection_tutorial.py ./object_detection/
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
$ python object_detection/object_detection_tutorial.py
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
```sh
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path models/model/faster_rcnn_resnet101_pets.config --trained_checkpoint_prefix models/model/model.ckpt-10 --output_directory exported_graphs
```

# License
This software is released under the Apache 2.0 License, see LICENSE.


# References
- https://github.com/tensorflow/models