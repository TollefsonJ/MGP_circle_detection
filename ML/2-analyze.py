# this script passes circles from Hough.Circles to the ML model,
# and output a CSV with filenames of



##### This script outputs each detected circle as an image to code for ML algorithm #####

#### FOLDER STRUCTURE###
# level 1: input folder, output folder, this script
# input folder: includes subfolders downloaded from LOC database, using LOC API. Folder names begin with "saveTo".


# import the necessary packages
import numpy as np
import argparse
import glob
import os,sys
import cv2
from PIL import Image, ImageOps


# set up arrays for output
all_circles_as_array = []
filename = []

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
p2 = 85
# blur
b1 = 3
b2 = 3

# thickness for circle that is drawn for you to see
draw_stroke = 8
text_size = 0.5
text_stroke = 1

# how much to increase bounding box around circle ROI
mult = 1.2
########################################

# load filenames
imgnames = sorted(glob.glob("analysis_images/resized/*.jpg"))

# exclude files that include 'ind', 'titl', or 'covr' in filenames
imgnames = [ x for x in imgnames if "ind" not in x and "titl" not in x and 'covr' not in x and 'cbd' not in x and '0000' not in x]

# load the images, clone for output
for imgname in imgnames:
    image = cv2.imread(imgname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(b1,b2),0)


################################### detect circles in the image, 3 iterations ##################################

# Compass? iteration
    circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,min_dist,
                                param1=p1,param2=p2,minRadius=rmin,maxRadius=rmax)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            box = int(mult*r)
            box = int(mult*r)
            ROI = gray[y-box:y+box, x-box:x+box]
            ROI = cv2.resize(ROI, (64,64))
            all_circles_as_array.append(ROI)
            filename.append(imgname)


    circles2 = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,min_dist_2,
                                param1=p1,param2=p2,minRadius=rmin_2,maxRadius=rmax_2)
    if circles2 is not None:
        circles2 = np.round(circles2[0, :]).astype("int")
        for (x, y, r) in circles2:
            box = int(mult*r)
            box = int(mult*r)
            ROI = gray[y-box:y+box, x-box:x+box]
            ROI = cv2.resize(ROI, (64,64))
            all_circles_as_array.append(ROI)
            filename.append(imgname)


# # Construct the arrays
X , z = np.array(all_circles_as_array), np.array(filename)
print(X.shape)

# reshape X array to 2-dimensions
nsamples, nx, ny = X.shape
X = X.reshape((nsamples, nx*ny))
print(X.shape)


####### 2. scale and run through model ######
import pickle

# define scaler and apply
scaler = pickle.load(open('training/scaler.pkl', 'rb'))
X_scaled = scaler.transform(X)

model = pickle.load(open('training/model.pkl', 'rb'))
y_pred = model.predict(X_scaled)

print(y_pred)
# turn arrays into pandas dataframe of results
import pandas as pd
dfz = pd.DataFrame(z)
dfy = pd.DataFrame(y_pred)
df = pd.concat([dfz, dfy], axis=1)
df.columns = ['file', 'prediction']
df.to_csv (r'output.csv', index = False)
