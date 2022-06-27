#!/usr/bin/env python

# python test_model.py IMG_DIR XML_DIR path_to_expoted_models path_to_labels fold_number

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import sys
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from shapely.geometry import Polygon 
import xml.etree.ElementTree as ET
import cv2
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


print("vis utils installation", viz_utils.__file__)

# python test_model.py /home/ajouffray/Data/web_blurry/ /home/ajouffray/Data/web_blurry_annotated/ /home/ajouffray/TF-Object_detection/exported/blurry_mask_rcnn/fold0/ /home/ajouffray/Data/output_soft_4/label.pbtxt 0

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

IMG_DIR = sys.argv[1]
XML_DIR = sys.argv[2]
IMAGE_PATHS = os.listdir(IMG_DIR)

path = sys.argv[3]
PATH_TO_LABELS = sys.argv[4]
fold = sys.argv[5]

name = path.split("/")[-3]

if not path.endswith("/"):

	path = path + "/"

models = os.listdir(path)


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
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

# takes a path to an xml file
def getBndBoxes(xmlfile):

	# create element tree object
	tree = ET.parse(xmlfile)
  
	# get root element
	root = tree.getroot()
	bnd = root.findall("./object/bndbox")

	print(bnd)
	all_bnd = []
	for box in bnd:

		coordinates = []
		for child in box:

			num = int(child.text)
			print("child in box", child.text)
			coordinates.append(num)

		all_bnd.append(coordinates)

	return all_bnd



# measure the intersection / union accuracy
def measureOverlap(p1, p2):

	overlap_area = p1.intersection(p2).area

	union = (p1.area - overlap_area) + p2.area

	overlap = (overlap_area / union) * 100

	return overlap


# checks for overlapping bounding boxes and consolidates them
def checkOverlap(bnd_boxes):

	# get all corners

	for i in range(len(bnd_boxes)):

		print("this is the index", i)
		print("this is the length", len(bnd_boxes))
        
		try:
			rect = bnd_boxes.pop(i)

			# compare the rect to all other:
		    
			for other in bnd_boxes:
			
				p1 = Polygon([(rect[0],rect[1]), (rect[2],rect[1]),(rect[2],rect[3]),(rect[0],rect[3])])
				p2 = Polygon([(other[0],other[1]), (other[2],other[1]),(other[2],other[3]),(other[0],other[3])])

				if p1.intersects(p2):

					print("overlap")
					# combine the two, delete the other and reinsert the new rect.

					overlap = measureOverlap(p1, p2)
					print(overlap, "% overlap accuracy")

					new_box = [None] * 4
			    
					if rect[0] > other[0]:
						new_box[0] = other[0]
					else: 
						new_box[0] = rect[0]

					if rect[1] > other[1]:
						new_box[1] = other[1]
					else:
						new_box[1] = rect[1]

					if rect[2] < other[2]:
						new_box[2] = other[2]
					else:
						new_box[2] = rect[2]

					if rect[3] < other[3]:
						new_box[3] = other[3]
					else:
						new_box[3] = rect[3]


					rect = new_box
					bnd_boxes.remove(other)

				else:
					pass

			bnd_boxes.insert(i, rect)

		except Exception as e:

		    break



	return bnd_boxes


models_average_accuracy = []


for model in models:

	if model.startswith("auto"):

		model_box_accuracy = []

		PATH_TO_MODEL_DIR = path + model
	
		# Load the model
		# ~~~~~~~~~~~~~~
		PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model/saved_model.pb"

		print('Loading model...', end='')
		start_time = time.time()

		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.compat.v1.GraphDef()
			with tf.io.gfile.GFile(PATH_TO_SAVED_MODEL, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		end_time = time.time()
		elapsed_time = end_time - start_time
		print('Done! Took {} seconds'.format(elapsed_time))

		category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
		warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

		def load_image_into_numpy_array(path):

			return np.array(Image.open(path))

		count = 0

		save_dir = "/home/ajouffray/TF-Object-Detection/testing/"+name+"/fold"+fold+"/"+model+"/"

		os.makedirs(save_dir)

		for image_path in IMAGE_PATHS:

			print('Running inference for {}... '.format(IMG_DIR+"/"+image_path), end='')

			image_np = load_image_into_numpy_array(IMG_DIR+"/"+image_path)
			xml_counterpart = XML_DIR+"/"+image_path[:len(image_path) - 3]+"xml"

			print("testing image: ", IMG_DIR+"/"+image_path)
			print("ground truth: ", XML_DIR+"/"+image_path[:len(image_path) - 3]+"xml")

			image_np_expanded = np.expand_dims(image_np, axis=0)
			# Actual detection.
			output_dict = run_inference_for_single_image(image_np, detection_graph)
			# Visualization of the results of a detection.
			vis_util.visualize_boxes_and_labels_on_image_array(
		   		image_np,
		   		output_dict['detection_boxes'],
		   		output_dict['detection_classes'],
		   		output_dict['detection_scores'],
		   		category_index,
		   		instance_masks=output_dict.get('detection_masks'),
		   		use_normalized_coordinates=True,
		   		line_thickness=8)
		
			cv.imwrite(save_dir+"image" + str(count) + ".jpg", image_np)
	
			count +=1

		average = sum(model_box_accuracy) / len(model_box_accuracy)

		models_average_accuracy.append([model, average])

print("\n======================= RESULTS =======================\n")
print(models_average_accuracy)
print("\n=======================================================")


