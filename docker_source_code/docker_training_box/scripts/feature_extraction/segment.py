#!/usr/bin/env python
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
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_util
from object_detection.utils import ops as utils_ops
from shapely.geometry import Polygon 
import xml.etree.ElementTree as ET
import cv2
import argparse
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

print("vis utils installation", viz_util.__file__)

sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print(cuda_version)
print(sys_details["cudnn_version"])
print(tf.__version__)
print(sys_details)


'''
	inputs:
	-i --input: path to the video to be processes
	-m --model: path to the model to use
	-l --label: label of the label in the object
	-n --name: name of the project these labels belong to

	output:
	an avi file where each frame contain the blacked-out object of interest, adn ready to be used as an input for the data augmentation algoritm

	TODO: Maybe generate the label.pbtxt file as well? 
'''


# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument

parser.add_argument("-i", "--input", help = "The path to the input MP4 or AVI video to be processed")
parser.add_argument("-m", "--model", help = "The path to the model to use")
parser.add_argument("-l", "--label", help = "The name of the object featured in the video")
parser.add_argument("-n", "--name", help = "The name of the project these videos are for")

args = parser.parse_args()
 
if not args.input:
	print("ERROR: Missing the input argument")
	exit()

if not args.model:
	print("ERROR: Missing the model argument")
	exit()

if not args.label:
	print("ERROR: Missing the label argument")
	exit()
 
if not args.name:
	print("ERROR: Missing the name argument")
	exit()


# this can be changed lated to be user defined, this is the general working dir for this function
DEFAULT_SAVE_PATH = "/videos"

# the path to the label map should be defined by the model we will use but for now it can be hardcoded
PATH_TO_LABELS = "/scripts/feature_extraction/label.pbtxt"

# create the save path for the project
try:
	os.mkdir(DEFAULT_SAVE_PATH+args.name)
except Exception as e:
	print(e)

# create the save path for these labels
try:
	os.mkdir(DEFAULT_SAVE_PATH+args.name+"/"+args.label)
except Exception as e:
	print(e)

# path to save the videos

save_path = os.path.join(DEFAULT_SAVE_PATH, (args.name+"/"+args.label))

# get the model
PATH_TO_SAVED_MODEL = args.model + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# get the labels
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


def get_total_frames():
    
	total = 0
    
	for vid in videos:

	    vid_path = os.path.join(args.input, vid)

	    cap = cv2.VideoCapture(vid_path)

	    subtotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	    total += subtotal

	return total


def getRatio(mask, w, h):

	pixels = cv2.countNonZero(mask)
	# pixels = len(np.column_stack(np.where(thresh > 0)))

	image_area = w * h
	area_ratio = (pixels / image_area) * 100

	return area_ratio

# create the video reader, get the video params

videos = os.listdir(args.input)

frames_to_process = get_total_frames()
processed_frames = 0
total_process_time = 0
average_process_time = 0
time_remaining = 0

def current_milli_time():
	return round(time.time() * 1000)

def update_average_time(total_process_time):
    
	average_process_time = int(total_process_time / processed_frames)
	time_remaining = int((frames_to_process - processed_frames) * average_process_time)

	return time_remaining

def print_stats():

	millis = time_remaining
	seconds=(millis/1000)%60
	seconds = int(seconds)
	minutes=(millis/(1000*60))%60
	minutes = int(minutes)
	hours=int((millis/(1000*60*60))%24)

	percent_done = int((processed_frames / frames_to_process) * 100)
	percent_remaining = 100 - percent_done

	UP = "\x1B[3A"
	CLR = "\x1B[0K"

	progress = f"progress: [" + (f"="*percent_done) + (f" "*percent_remaining) + f"]"

	print ((f" Time remaining: {hours:2d}:{minutes:2d}:{seconds:2d}" + f"    {progress:>5}"), flush=True)


print("files to process: ", videos)
for vid in videos:
	print("video file: ", vid)
	if vid.lower().endswith(".avi") or vid.lower().endswith(".mp4") or vid.lower().endswith(".mov"):

		#print("extracting features")

		vid_path = os.path.join(args.input, vid)

		cap = cv2.VideoCapture(vid_path)
		width = cap.get(3)
		height = cap.get(4)
		fps = cap.get(5)
		size = (int(width), int(height))
		size_np = (int(height), int(width), 3)

		#print("size = ", size)

		# get the number of files in the save path
		file_count = 0
		try:
			files = os.listdir(save_path)
			for f in files:	
				if f.endswith("com.avi"):
					file_count +=1
		except:
			print("output path does not exist, creating it...")
			os.makedirs(save_path)

		com_writer = cv2.VideoWriter(save_path+'/output'+str(file_count)+'_com.avi', cv2.VideoWriter_fourcc(*'MJPG'),fps, size)

		print("output path: ", save_path)

		count = 0

		min_score = 0.45

		# minimum percentage of screen taken by the object, we need the object to appear big in the images
		min_ratio = 8.0
		alpha = 1.0
			
		while True:

			ret, image_np  = cap.read()
			
			if image_np is None:
				break

			processed_frames += 1
			start = current_milli_time()

			# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
			input_tensor = tf.convert_to_tensor(image_np)
			# The model expects a batch of images, so add an axis with `tf.newaxis`.
			input_tensor = input_tensor[tf.newaxis, ...]

			# input_tensor = np.expand_dims(image_np, 0)
			detections = detect_fn(input_tensor)

			num_detections = int(detections.pop('num_detections'))

			need_detection_key = ['detection_classes','detection_boxes','detection_masks','detection_scores']
			detections = {key: detections[key][0, :num_detections].numpy()
				       for key in need_detection_key}
			detections['num_detections'] = num_detections
			# detection_classes should be ints.
			detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

			
			# change the datatypes
			if 'detection_masks' in detections:
			    # Reframe the the bbox mask to the image size.
			    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
				tf.convert_to_tensor(detections['detection_masks']), detections['detection_boxes'],
				image_np.shape[0], image_np.shape[1])
			    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
							       tf.uint8)
			    detections['detection_masks_reframed'] = detection_masks_reframed.numpy()

			im_height, im_width, depth = image_np.shape

			# this is a list of all masks associated with bounding boxes, many of those masks are too small and have bad confidence scores so we need to 
			masks = detections.get('detection_masks_reframed')
			i = 0

			
			# create a black frame for the mask
			blk = np.zeros((size_np), np.uint8)

			# check each masks
			for mask in masks:
				
				score = detections['detection_scores'][i]
				if score >= min_score:

				
					#print("w, h", width, height)
					
					ratio = getRatio(mask, width, height)
					if ratio >= min_ratio:
						
						# create two black frames for the combined mask and image, and the output
						out = np.zeros((size_np), np.uint8)

						
						# draw the contour on the image
						rgb = ImageColor.getrgb('white')
						pil_image = Image.fromarray(blk)

						solid_color = np.expand_dims(
						np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
						pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
						pil_mask = Image.fromarray(np.uint8(255.0*alpha*(mask > 0))).convert('L')
						pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
						np.copyto(blk, np.array(pil_image.convert('RGB')))

						com = cv2.bitwise_and(image_np, blk)

						# get the coordinates of the bounding box
						box = detections['detection_boxes'][i]
						x1, x2, y1, y2 = (box[1] * im_width, box[3] * im_width, box[0] * im_height, box[2] * im_height)
						x1 = int(x1)
						x2 = int(x2)
						y1 = int(y1)
						y2 = int(y2)

						#print("out size", out.shape)
						#print("com size", com.shape)

						#print(x1, x2, y1, y2)

						# cut out the bounding box and paste it onto the out file
						sml = com[y1:y2, x1:x2]
						
						# scale up the object in the image:
					
						h1, w1 = sml.shape[:2]
						
						# out height is 1080, and width is 1920
						# we use slightly smaller values to make sure thay scale but are still smaller than out

						scale_from_height = (height - 20) / h1
						scale_from_width = (width - 20) / w1
				
						new_height = int(h1 * scale_from_height)
						new_width = int(w1 * scale_from_height)
						
						# this could mean the smaller object is wider than the out image, so we should scale fvrom width and not height
						if new_width > width:
							new_height = int(h1 * scale_from_width)
							new_width = int(w1 * scale_from_width)

						sml = cv2.resize(sml, (new_width-1, new_height-1))
						sml_height, sml_width, channels = sml.shape

						#print(h1, w1)
						out[0:sml_height, 0:sml_width] = sml

						
						com_writer.write(out)	
					
				
				i +=1
			
			count +=1

			end = current_milli_time() 
			runtime = end - start
			total_process_time += runtime
			time_remaining = update_average_time(total_process_time)
			print_stats()


		cap.release()
		com_writer.release()

