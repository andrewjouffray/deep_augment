import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import random_shapes
import cv2
import os
import random

# looks or pixels in a certain range of color and returns a mask
def black_mask_return_mask(img):

    # Color range for black
    keepl = np.array([255, 230, 230])
    keeph = np.array([255, 255, 255])
    keep_mask = cv2.inRange(img, keepl, keeph)

    # Returns an image where all the pixels that were light blue are now all white
    return keep_mask


def black_to_image(blacked_img_og, background_image):


    # makes a copy of the blacked image
    blacked_img = blacked_img_og.copy()

    # gets dimensions of the blacked_image
    height, width, channels = blacked_img.shape

    # resize the background image to match the blacked out image
    background_image = cv2.resize(background_image, (width, height))


    # get all the blacked pixels
    blk_mask = black_mask_return_mask(blacked_img)

    # all pixels that were not black are going to have the blacked out image on them
    blacked_img[blk_mask != 0] = [0, 0, 0] # possibly useless

    # all pixels that were black are going to have the background image on them
    background_image[blk_mask == 0] = [0, 0, 0]

    # add the two together
    img = background_image + blacked_img


    return img


def getBackgroundPaths(basePath):

    if not basePath.endswith("/"):

        basePath += "/"

    backgrounds = []

    for path in  os.listdir(basePath):

        backgrounds.append(basePath + path)

    return backgrounds

def getRandBackground(backgrounds):

    index = random.randint(0, len(backgrounds) -1)

    return backgrounds[index]

backgrounds = getBackgroundPaths("./object_textures/")



for i in range(1000):

    image, _ = random_shapes((400, 620), min_shapes=250, max_shapes=270, min_size=45, max_size=55, allow_overlap=True, num_channels = 3, intensity_range = ((0, 0),))
    background = cv2.imread(getRandBackground(backgrounds))

    image = cv2.bitwise_not(image)

    image = black_to_image(image, background)

    cv2.imwrite("../uncompiled_random_shapes/shape"+str(i)+".jpg", image)



