#### FOLDER STRUCTURE###
# level 1: input folder, this script
# input folder: includes subfolders downloaded from LOC database, using LOC API. Folder names begin with "saveTo".
# place output folder WITHIN input folder

################## ARE THERE COMPASSES ON THE MAPS? 0 for no, 1 for yes #################
compass = 0
###################################################


# import the necessary packages
import numpy as np
import argparse
import glob
import os,sys
import cv2

# set function to change output path
def change_to_output(path):
    return os.path.join(os.path.split(os.path.dirname(path))[0], 'output', os.path.basename(path))

################################## parameters ##################################
# params for circle bin radii, hough_circles, GaussianBlur, and circle drawn by cv2 are defined here

# params for circles_compass
## Radii
rmin_c = 25
rmax_c = 40
# min distance between circles
min_dist_c = 15
# params
p1_c = 20
p2_c = 75
# blur
b1_c = 3
b2_c = 3

# params for circles_small
## Radii
rmin_s = 41
rmax_s = 65
# min distance between circles
min_dist_s = 15
# params
p1_s = 20
p2_s = 60
# blur
b1_s = 7
b2_s = 7

# params for circles_large
## Radii
rmin_l = 66
rmax_l = 100
# min distance between circles
min_dist_l = 55
# params
p1_l = 20
p2_l = 60
# blur
b1_l = 7
b2_l = 7

# thickness for circle that is drawn for you to see
draw_stroke = 8
text_size = 0.5
text_stroke = 1

################################## end of params ##################################


# load filenames
## including /saveTo**/ also searches one level of sub-directories below "input" that start with saveTo
imgnames = sorted(glob.glob("input/***/*.jpg"))

# load the images, clone for output
for imgname in imgnames:
    image = cv2.imread(imgname)
    copy = image.copy()



######## resize to fit params ###################################################

    height, width = copy.shape[:2]
    target_height = 2008
    target_width = 1390

    # enlarge image by factor
    if (target_height * target_width) < (height * width):
        # get scaling factor
        scaling_factor = (target_height + target_width) / (float(height) + float(width))

        # resize image
        output = cv2.resize(copy, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # enlarge image by a calculated factor so it's the same resolution as the target
    if (target_height * target_width) > (height * width):
        # get scaling factor
        scaling_factor = (target_height + target_width) / (float(height) + float(width))

        # resize image
        output = cv2.resize(copy, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

########## end of resize to fit params ########################################


# add blur and convert to grayscale
    blur_c = cv2.GaussianBlur(output,(b1_c,b2_c),0)
    blur_s = cv2.GaussianBlur(output,(b1_s,b2_s),0)
    blur_l = cv2.GaussianBlur(output,(b1_l,b2_l),0)

    gray_c = cv2.cvtColor(blur_c, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.cvtColor(blur_s, cv2.COLOR_BGR2GRAY)
    gray_l = cv2.cvtColor(blur_l, cv2.COLOR_BGR2GRAY)


################################### detect circles in the image, 3 iterations ##################################

# Compass? iteration
    circles_c = cv2.HoughCircles(gray_c,cv2.HOUGH_GRADIENT,1,min_dist_c,
                                param1=p1_c,param2=p2_c,minRadius=rmin_c,maxRadius=rmax_c)
    if circles_c is not None:
        circles_c = np.round(circles_c[0, :]).astype("int")

        for (x, y, r) in circles_c:
            # draw the outer circle
            cv2.circle(output,(x, y), r, (0, 255, 0), draw_stroke)
            # draw the radius
            cv2.putText(output,str(r),(x,y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0), text_stroke, cv2.LINE_AA)

        #   draw the center of the circle
        #   cv2.circle(output,(x, y), 2, (0, 0, 255), 3)

# Small iteration
    circles = cv2.HoughCircles(gray_s,cv2.HOUGH_GRADIENT,1,min_dist_s,
                                param1=p1_s,param2=p2_s,minRadius=rmin_s,maxRadius=rmax_s)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # draw the outer circle
            cv2.circle(output,(x, y), r, (0, 255, 0), draw_stroke)
            # draw the radius
            cv2.putText(output,str(r),(x,y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0), text_stroke, cv2.LINE_AA)

        #   draw the center of the circle
        #   cv2.circle(output,(x, y), 2, (0, 0, 255), 3)


# Large iteration
    circles_1 = cv2.HoughCircles(gray_l,cv2.HOUGH_GRADIENT,1,min_dist_l,
                                param1=p1_l,param2=p2_l,minRadius=rmin_l,maxRadius=rmax_l)
    if circles_1 is not None:
        circles_1 = np.round(circles_1[0, :]).astype("int")

        for (x, y, r) in circles_1:
            # draw the outer circle
            cv2.circle(output,(x, y), r, (0, 255, 0), draw_stroke)
            # draw the radius
            cv2.putText(output,str(r),(x,y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0), text_stroke, cv2.LINE_AA)

        #   draw the center of the circle
        #   cv2.circle(output,(x, y), 2, (0, 0, 255), 3)


##################################### set conditions and output files with circles ##################################

# define bottom threshold for how many circles to find
    if circles_c is not None:
        no_of_circles_c = int(len(circles_c))
    else: no_of_circles_c = int(0)

    if circles is not None:
        no_of_circles = int(len(circles))
    else: no_of_circles = int(0)

    if circles_1 is not None:
        no_of_circles_1 = int(len(circles_1))
    else: no_of_circles_1 = int(0)

# number_circles it the total number of circles found, minus the number of compasses on the map
    number_circles = no_of_circles + no_of_circles_1 + no_of_circles_c - compass

# if there is a circle, save the file with "_out" and the number of circles appended to the filename
    if number_circles > 0:
        imgname1 = "_out".join(os.path.splitext(imgname))
        imgname1 = str(number_circles).join(os.path.splitext(imgname1))
        imgname1 = change_to_output(imgname1)
        cv2.imwrite(imgname1, output)
    else:
        imgname2 = imgname
