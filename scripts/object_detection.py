# coding: utf-8
# Object Detection Demo
import argparse
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='object_detection_tutorial.')
parser.add_argument('-l', '--labels', default='./object_detection_tools/data/tf_label_map.pbtxt')
parser.add_argument('-m', '--model', default='./exported_graphs/frozen_inference_graph.pb')
parser.add_argument('-d', '--device', default='normal_cam')

args = parser.parse_args()

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(args.model, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

# Switch camera according to device
if args.device == 'normal_cam':
  cam = cv2.VideoCapture(0)
elif args.device == 'jetson_nano_raspi_cam':
  GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
  cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
elif args.device == 'jetson_nano_web_cam':
  cam = cv2.VideoCapture(1)
else:
  print('wrong device')
  sys.exit()


count_max = 0

if __name__ == '__main__':
    count = 0

    labels = ['blank']
    with open(args.labels,'r') as f:
        for line in f:
            labels.append(line.rstrip())

    while True:
        ret, img = cam.read()
        if not ret:
            print('error')
            break
        key = cv2.waitKey(1)
        if key == 27: # when ESC key is pressed break
            break

        count += 1
        if count > count_max:
            img_bgr = cv2.resize(img, (300, 300))

            # convert bgr to rgb
            image_np = img_bgr[:,:,::-1]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

            for i in range(output_dict['num_detections']):
              class_id = output_dict['detection_classes'][i]
              if class_id < len(labels):
                label = labels[class_id]
              else:
                label = 'unknown'

              detection_score = output_dict['detection_scores'][i]

              # Draw bounding box
              h, w, c = img.shape
              box = output_dict['detection_boxes'][i] * np.array( \
                  [h, w,  h, w])
              box = box.astype(np.int)
              cv2.rectangle(img, \
                  (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 3)

              # Put label near bounding box
              information = '%s: %f' % (label, output_dict['detection_scores'][i])
              cv2.putText(img, information, (box[1], box[2]), \
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('detection result', img)
            count = 0

    cam.release()
    cv2.destroyAllWindows()
