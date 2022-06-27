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
import re
import collections
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


# takes a path to an xml file
def getBndBoxes(xmlfile):

        # create element tree object
        tree = ET.parse(xmlfile)
  
        # get root element
        root = tree.getroot()
        objects = root.findall("./object")


        all_bnd = []
        for element in objects:

                name = element.find('name').text

                coordinates = []

                box = element.find('bndbox')

                for child in box:

                        num = int(child.text)
                        print("child in box", child.text)
                        coordinates.append(num)

                coordinates.append(name)

                all_bnd.append(coordinates)
                print("gt box info (x1 y1 x2 y2 name)", coordinates)

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

                try:
                        rect = bnd_boxes.pop(i)

                        # compare the rect to all other:
                    
                        for other in bnd_boxes:

                                p1 = Polygon([(rect[0],rect[1]), (rect[2],rect[1]),(rect[2],rect[3]),(rect[0],rect[3])])
                                p2 = Polygon([(other[0],other[1]), (other[2],other[1]),(other[2],other[3]),(other[0],other[3])])

                                if p1.intersects(p2):

                                        # combine the two, delete the other and reinsert the new rect.

                                        overlap = measureOverlap(p1, p2)

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


models_average_accuracy = {}
models_average_pf_accuracy = {}


for model in models:

        if model.startswith("auto"):

                model_box_accuracy = []
                model_pf_accuracy = []

                PATH_TO_MODEL_DIR = path + model

                # Load the model
                # ~~~~~~~~~~~~~~
                PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

                print('Loading model...', end='')
                start_time = time.time()

                # Load saved model and build the detection function
                detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Done! Took {} seconds'.format(elapsed_time))

                category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
                warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

                def load_image_into_numpy_array(path):

                        return np.array(Image.open(path))

                count = 0

                save_dir = "/workdir/testing/"+name+"/fold"+fold+"/"+model+"/"

                try:

                        os.makedirs(save_dir)

                except:

                        print("Error: could not create the save directory, ignore if it already exists")

                for image_path in IMAGE_PATHS:

                        print('Running inference for {}... '.format(IMG_DIR+"/"+image_path))

                        image_np = load_image_into_numpy_array(IMG_DIR+"/"+image_path)
                        xml_counterpart = XML_DIR+"/"+image_path[:len(image_path) - 3]+"xml"

                        print("xml counterpart:", xml_counterpart)

                        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                        input_tensor = tf.convert_to_tensor(image_np)
                        # The model expects a batch of images, so add an axis with `tf.newaxis`.
                        input_tensor = input_tensor[tf.newaxis, ...]

                        # input_tensor = np.expand_dims(image_np, 0)
                        detections = detect_fn(input_tensor)

                        # All outputs are batches tensors.
                        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
                        # We're only interested in the first num_detections.
                        num_detections = int(detections.pop('num_detections'))
                        detections = {key: value[0, :num_detections].numpy()
                                   for key, value in detections.items()}
                        detections['num_detections'] = num_detections

                        # detection_classes should be ints.
                        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                        image_np_with_detections = image_np.copy()

                        im_height, im_width, depth = image_np.shape

                        bnd_boxes = []

                        # get all the boxes
                        for idx, box in enumerate(detections['detection_boxes']):

                                #if detections['detection_scores'][idx] > .30:

                                # reformating tensors to be xy coordinates:
                                x1, x2, y1, y2 = (box[1] * im_width, box[3] * im_width, box[0] * im_height, box[2] * im_height)

                                bnd = [x1, y1, x2, y2]

                                bnd_boxes.append(bnd)

                        ground_truth_boxes = getBndBoxes(xml_counterpart)

                        image_np_with_detections = image_np.copy()
                        gt_check = image_np.copy()


                        # get ground truth:

                                # find all prediction intersection with the same class
                                        # add the accuracy 
                                # if no intesection 
                                        # add a 0 to the accuracy list 

                        # go though the predictions

                                # got through the ground truths
                                # all predictions with no intersections to ground truth of the same class are added as a 0 to the list


                        # accuracy list:
                        image_accuracy = []
                        image_pf_accuracy = []

                        # finds all ground truth that have a prediction of the correct class, and all the ground truth that were missed
                        for idx, gt_box in enumerate(ground_truth_boxes):

                                gt_name = gt_box[4]
                                gt_poly = Polygon([(int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (int(gt_box[0]), int(gt_box[3]))])

                                cv2.rectangle(gt_check,(int(gt_box[0]),int(gt_box[1])),(int(gt_box[2]),int(gt_box[3])),(100,100,100),2)


                                is_found = False

                                # predicted boxes
                                for idj, box in enumerate(bnd_boxes):

                                        if detections['detection_scores'][idj] > .50:

                                                pred_name = detections['detection_classes'][idj]

                                                # if they are the same class

                                                if str(pred_name) == "1":
                                                        pred_name = "apples"
                                                elif str(pred_name) == "2":
                                                        pred_name = "citrus"
                                                elif str(pred_name) == "3":
                                                        pred_name = "pears"

                                                if pred_name == gt_name:

                                                        pred_poly = Polygon([(int(box[0]),int(box[1])), (int(box[2]),int(box[1])),(int(box[2]),int(box[3])),(int(box[0]),int(box[3]))])

                                                        if pred_poly.intersects(gt_poly):


                                                                accuracy = measureOverlap(pred_poly, gt_poly)


                                                                # pass fail accuracy
                                                                # if the accuracy is above 60% then it's a pass
                                                                if accuracy >= 60:
                                                                        cv2.rectangle(image_np,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),3)
                                                                        image_pf_accuracy.append(100)
                                                                        #cv2.putText(image_np, str(idx), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2, cv2.LINE_AA)
                                                                        is_found = True
                                                                        image_accuracy.append(accuracy)
                                                                        break
                                                                # if the accuracy is between 60% and 30% it's a fail
                                                                elif accuracy >= 30:

                                                                        cv2.rectangle(image_np,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(200,200,0),3)
                                                                        #cv2.putText(image_np, str(idx), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2, cv2.LINE_AA)
                                                                        image_pf_accuracy.append(0)
                                                                        is_found = True
                                                                        image_accuracy.append(accuracy)
                                                                        break
                                                                # if the accuracy is less than 30% we assume it's a nearby bounding box overlapping
                                                                else:
                                                                        pass

                                                                # area accuracy

                                # if no predictions overlapped then the model did not find that obejct
                                if not is_found:

                                        cv2.rectangle(image_np,(int(gt_box[0]),int(gt_box[1])),(int(gt_box[2]),int(gt_box[3])),(255,0,0),3)

                                        #cv2.putText(image_np, str(idx), (int(gt_box[0]), int(gt_box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                                        image_accuracy.append(0)
                                        image_pf_accuracy.append(0)


                        # check for all innacurate predictions
                        for idx, box in enumerate(bnd_boxes):

                                on_something = False

                                if detections['detection_scores'][idx] > .50:

                                                pred_name = detections['detection_classes'][idx]

                                                pred_poly = Polygon([(int(box[0]),int(box[1])), (int(box[2]),int(box[1])),(int(box[2]),int(box[3])),(int(box[0]),int(box[3]))])

                                                for idj, gt_box in enumerate(ground_truth_boxes):

                                                        gt_name = gt_box[4]
                                                        gt_poly = Polygon([(int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (int(gt_box[0]), int(gt_box[3]))])

                                                        if pred_poly.intersects(gt_poly):
                                                                on_something = True


                                                if not on_something:


                                                        cv2.rectangle(image_np,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),3)
                                                        #cv2.putText(image_np, "F", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


                                                        image_accuracy.append(0)
                                                        image_pf_accuracy.append(0)

                        final_img_accuracy = sum(image_accuracy) / len(image_accuracy)
                        final_pf_accuracy = sum(image_pf_accuracy) / len(image_pf_accuracy)

                        model_box_accuracy.append(final_img_accuracy)
                        model_pf_accuracy.append(final_pf_accuracy)


                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
                        gt_check = cv2.cvtColor(gt_check, cv2.COLOR_RGB2BGR)

                        viz_utils.visualize_boxes_and_labels_on_image_array(
                                image_np_with_detections,
                                detections['detection_boxes'],
                                detections['detection_classes'],
                                detections['detection_scores'],
                                category_index,
                                use_normalized_coordinates=True,
                                max_boxes_to_draw=200,
                                min_score_thresh=.50,
                                agnostic_mode=False
                        )

                        cv.imwrite(save_dir+"predicted_image" + str(count) + ".jpg", image_np_with_detections)
                        cv.imwrite(save_dir+"analyzed_image" + str(count) + ".jpg", image_np)
                        cv.imwrite(save_dir+"gt_check_image" + str(count) + ".jpg", gt_check)

                        count +=1


                average = sum(model_box_accuracy) / len(model_box_accuracy)
                pf_average = sum(model_pf_accuracy) / len(model_pf_accuracy)

                model_num = int(re.sub(r"\D", "", model))


                models_average_accuracy[model_num] = average
                models_average_pf_accuracy[model_num] = pf_average

# sort the models_average_accuracy


ordered = collections.OrderedDict(sorted(models_average_accuracy.items()))
ordered_pf = collections.OrderedDict(sorted(models_average_pf_accuracy.items()))

print("\n======================= RESULTS I Over U =======================\n")
for k, v in ordered.items(): print(k, v)
print("\n==================================================================")


print("\n======================= RESULTS PASS / FAIL =======================\n")
for k, v in ordered_pf.items(): print(k, v)
print("\n==================================================================")

with open("/workdir/testing/"+name+"/fold"+fold+"/IoU_accuracy.txt", 'w') as f:
        for k, v in ordered.items(): f.write('%s:%s\n' % (k, v)) 
 
with open("/workdir/testing/"+name+"/fold"+fold+"/PF_accuracy.txt", 'w') as f:
        for k, v in ordered_pf.items(): f.write('%s:%s\n' % (k, v)) 
