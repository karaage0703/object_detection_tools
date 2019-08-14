# coding: utf-8
# convert from pbtxt to label
import argparse
from object_detection.utils import label_map_util

# Path to label
parser = argparse.ArgumentParser(description='object_detection_tutorial.')
parser.add_argument('-l', '--labels', default='./object_detection_tools/data/tf_label_map.pbtxt')
args = parser.parse_args()

category_index = label_map_util.create_category_index_from_labelmap(args.labels, use_display_name=True)
# print(category_index)

for i in range(len(category_index)):
  print(category_index[i+1]['name'])
