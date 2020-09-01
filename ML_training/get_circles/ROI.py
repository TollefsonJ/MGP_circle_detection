# This script grabs and outputs circles from maps in the "input" folder
# Circle regions are output to "ROI"
# Circle regions are then hand-coded to train ML model


# import the necessary packages
import numpy as np
import argparse
import glob
import os,sys
import cv2
from PIL import Image, ImageOps

#################### define functions ##############
# define change_to_ROI function
def change_to_ROI(path):
    return os.path.join(os.path.split(os.path.dirname(os.path.dirname(path)))[0], 'ROI', os.path.basename(path))

# define circles function: finds circle ROIs and outputs to output path
def find_circles(rmin, rmax, min_dist, p2):
    circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,min_dist,
                                param1=p1,param2=p2,minRadius=rmin,maxRadius=rmax)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            box = int(mult*r)
            ROI = copy[y-box:y+box, x-box:x+box]
            circ_roi = "-".join(os.path.splitext(imgname))
            circ_roi = str(x).join(os.path.splitext(circ_roi))
            circ_roi = ",".join(os.path.splitext(circ_roi))
            circ_roi = str(y).join(os.path.splitext(circ_roi))
            circ_roi = change_to_ROI(circ_roi)
            try:
                cv2.imwrite(circ_roi, ROI)
            except:
                continue


################################## parameters ##################################
# global circles params
p1 = 20
# blur
b1 = 3
b2 = 3

# how much to increase bounding box around circle ROI
mult = 1.2

# define pass 1
min1 = 15
max1 = 80
dist1 = 20
p2_1 = 70

# define pass 2
min2 = 81
max2 = 150
dist2 = 60
p2_2 = 70

###############################################################################
#########################  find circles!!!!!!!!  ##############################
###############################################################################
# load filenames
imgnames = sorted(glob.glob("analysis_images/input/*.jpg"))

# exclude files that include 'ind', 'titl', or 'covr' in filenames
imgnames = [ x for x in imgnames if "ind" not in x and "titl" not in x and 'covr' not in x and 'cbd' not in x and '0000' not in x]

# load images
for imgname in imgnames:
    image = cv2.imread(imgname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(b1,b2),0)
# find the circles!
    find_circles(min1, max1, dist1, p2_1)
    find_circles(min2, max2, dist2, p2_2)
