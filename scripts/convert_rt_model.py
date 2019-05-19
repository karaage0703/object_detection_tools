# coding: utf-8
import argparse
import sys
import os
import tensorflow.contrib.tensorrt as trt
from tf_trt_models.detection import download_detection_model, build_detection_graph

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='convert_rt_model.')
parser.add_argument('-c', '--config', default='./parallels.config')
parser.add_argument('-m', '--model', default='./model.ckpt')
parser.add_argument('-o', '--output', default='./exported_graphs/frozen_inference_graph_trt.pb')

args = parser.parse_args()

frozen_graph, input_names, output_names = build_detection_graph(
    config=args.config,
    checkpoint=args.model,
    score_threshold=0.3,
    batch_size=1
)

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

with open(args.output, 'wb') as f:
    f.write(trt_graph.SerializeToString())
