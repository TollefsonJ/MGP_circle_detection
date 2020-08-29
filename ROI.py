##### This script outputs each detected circle as an image to code for ML algorithm #####

#### FOLDER STRUCTURE###
# level 1: input folder, output folder, this script
# input folder: includes subfolders downloaded from LOC database, using LOC API. Folder names begin with "saveTo".

################## ARE THERE COMPASSES ON THE MAPS? 0 for no, 1 for yes #################
compass = 0
###################################################


# import the necessary packages
import numpy as np
import argparse
import glob
import os,sys
import cv2

# set function to change output paths for images and ROI crops

def change_to_output(path):
    return os.path.join(os.path.split(os.path.dirname(os.path.dirname(path)))[0], 'output', os.path.basename(path))
def change_to_ROI(path):
    return os.path.join(os.path.split(os.path.dirname(os.path.dirname(path)))[0], 'ROI', os.path.basename(path))

################################## parameters ##################################
# params for circle bin radii, houghircles, GaussianBlur, and circle drawn by cv2 are defined here


# params for circlesompass
## Radii
rmin = 20
rmax = 80

rmin_2 = 81
rmax_2 = 150
# min distance between circles
min_dist = 20
min_dist_2 = 60
# params
p1 = 20
p2 = 75
# blur
b1 = 3
b2 = 3

# thickness for circle that is drawn for you to see
draw_stroke = 8
text_size = 0.5
text_stroke = 1

# how much to increase bounding box around circle ROI
box = 15
########################################


# load filenames
imgnames = sorted(glob.glob("input/***/*.jpg"))

# exclude files that include 'ind', 'titl', or 'covr' in filenames
imgnames = [ x for x in imgnames if "ind" not in x and "titl" not in x and 'covr' not in x and '0000' not in x]

# load the images, clone for output
for imgname in imgnames:
    image = cv2.imread(imgname)




######### resize to fit params ###################################################
#
#    height, width = image.shape[:2]
#    target_height = 2067
#    target_width = 1433
#
#    # enlarge image by factor
#    if (target_height * target_width) < (height * width):
#        # get scaling factor
#        scaling_factor = (target_height + target_width) / (float(height) + float(width))
#
#        # resize image
#        output = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
#
#    # enlarge image by a calculated factor so it's the same resolution as the target
#    if (target_height * target_width) > (height * width):
#        # get scaling factor
#        scaling_factor = (target_height + target_width) / (float(height) + float(width))
#
#        # resize image
#        output = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
#

## add blur and convert to grayscale
#    copy = output.copy()
#    blur = cv2.GaussianBlur(output,(b1,b2),0)
#    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#
########### end of resize to fit params ########################################

# comment out the line below if using resize function. otherwise, leave it in

# add blur and convert to grayscale
    output = image.copy()
    copy = image.copy()
    blur = cv2.GaussianBlur(image,(b1,b2),0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

################################### detect circles in the image, 3 iterations ##################################

# Compass? iteration
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,min_dist,
                                param1=p1,param2=p2,minRadius=rmin,maxRadius=rmax)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            ROI = copy[y-r-box:y+r+box, x-r-box:x+r+box]
            circ_roi = "-".join(os.path.splitext(imgname))
            circ_roi = str(x).join(os.path.splitext(circ_roi))
            circ_roi = ",".join(os.path.splitext(circ_roi))
            circ_roi = str(y).join(os.path.splitext(circ_roi))
            circ_roi = change_to_ROI(circ_roi)
            cv2.imwrite(circ_roi, ROI)

            cv2.circle(output,(x, y), r, (0, 255, 0), draw_stroke)
            # draw the radius
            cv2.putText(output,str(x)+','+str(y),(x,y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0), text_stroke, cv2.LINE_AA)


    circles2 = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,min_dist_2,
                                param1=p1,param2=p2,minRadius=rmin_2,maxRadius=rmax_2)
    if circles2 is not None:
        circles2 = np.round(circles2[0, :]).astype("int")
        for (x, y, r) in circles2:
            ROI = copy[y-r-box:y+r+box, x-r-box:x+r+box]
            circ_roi = "-".join(os.path.splitext(imgname))
            circ_roi = str(x).join(os.path.splitext(circ_roi))
            circ_roi = ",".join(os.path.splitext(circ_roi))
            circ_roi = str(y).join(os.path.splitext(circ_roi))
            circ_roi = change_to_ROI(circ_roi)
            cv2.imwrite(circ_roi, ROI)

            cv2.circle(output,(x, y), r, (0, 255, 0), draw_stroke)
            # draw the radius
            cv2.putText(output,str(x)+','+str(y),(x,y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0), text_stroke, cv2.LINE_AA)


# define bottom threshold for how many circles to find
    if circles is not None:
        no_of_circles = int(len(circles))
    else: no_of_circles = int(0)


    if circles2 is not None:
        no_of_circles2 = int(len(circles2))
    else: no_of_circles2 = int(0)

# number_circles it the total number of circles found, minus the number of compasses on the map
    number_circles = no_of_circles + no_of_circles2

# if there is a circle, save the file with "_out" and the number of circles appended to the filename
    if number_circles > 0:
        imgname1 = "_out".join(os.path.splitext(imgname))
        imgname1 = str(number_circles).join(os.path.splitext(imgname1))
        imgname1 = change_to_output(imgname1)
        cv2.imwrite(imgname1, output)
    else:
        imgname2 = imgname
