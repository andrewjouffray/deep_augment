import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom
from skimage.draw import random_shapes
import cv2
import os
import random



# looks or pixels in a certain range of color and returns a mask
def white_mask_return_mask(img):

    # Color range for black
    keepl = np.array([255, 230, 230])
    keeph = np.array([255, 255, 255])
    keep_mask = cv2.inRange(img, keepl, keeph)

    # Returns an image where all the pixels that were light blue are now all white
    return keep_mask


def white_to_image(blacked_img_og, background_image):


    # makes a copy of the blacked image
    blacked_img = blacked_img_og.copy()

    # gets dimensions of the blacked_image
    height, width, channels = blacked_img.shape

    # resize the background image to match the blacked out image
    background_image = cv2.resize(background_image, (width, height))


    # get all the blacked pixels
    blk_mask = white_mask_return_mask(blacked_img)

    # all pixels that were not black are going to have the blacked out image on them
    blacked_img[blk_mask != 0] = [0, 0, 0] # possibly useless

    # all pixels that were black are going to have the background image on them
    background_image[blk_mask == 0] = [0, 0, 0]

    # add the two together
    img = background_image + blacked_img


    return img


def black_mask_return_mask(img):

    # Color range for black
    keepl = np.array([0, 0, 0])
    keeph = np.array([5, 5, 5])
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
    blacked_img[blk_mask != 0] = [0, 0, 0]

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
filaments = getBackgroundPaths("./filaments/")


#https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib
bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)




rad = 0.05
edgy = 0.01
image_size = 500

out = cv2.VideoWriter('../compiled_shapes_organic_filaments.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1920,1080))

for i in range(10000):


    # get a random filament
    filament = cv2.imread(getRandBackground(filaments))
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    canvas.fill(255)

    for j in range(random.randint(2, 8)):

        local_canvas = np.zeros((1080, 1920, 3), np.uint8)
        local_canvas.fill(255)

        a = get_random_points(n=45, scale=1)

        print("new shape: ", i)

        # set of py coordinates, and set of x cooridinates
        x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)

        contours = []
        for j in range(len(x) - 1):

            point = [None, None]
            point[0] = int(x[j] * image_size)
            point[1] = int(y[j] * image_size)
            contours.append(point)

        contours = np.asarray(contours)

        # shapes is fully created here, black on white background
        image = np.zeros((image_size, image_size, 3), np.uint8)
        image.fill(255)
        cv2.fillPoly(image, pts = [contours], color =(0,0,0))

        x_offset = random.randint(0, 1400)
        y_offset = random.randint(0, 500)

        local_canvas[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

        canvas = white_to_image(canvas, local_canvas)


        #cv2.imshow("filledPolygon", canvas)


        #cv2.waitKey(0)

        #closing all open windows
        #cv2.destroyAllWindows()

    background = cv2.imread(getRandBackground(backgrounds))

    image = white_to_image(canvas, filament)
    image = black_to_image(image, background)

    # replace pure white with black
    white_pixels = np.where(
        (image[:, :, 0] >= 210) &
        (image[:, :, 1] >= 200) &
        (image[:, :, 2] >= 210)
    )

    # set those pixels to black
    image[white_pixels] = [0, 0, 0]

    #cv2.imshow("filledPolygon", image)


    #cv2.waitKey(0)

    out.write(image)

out.release()
