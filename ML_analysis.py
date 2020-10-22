########### NOTE: this is INCOMPLETE ########



# this script passes circles from Hough.Circles to the ML model,
# and outputs:
    # CSV with ML predictions:
        # p(pos) = probability that circle is an MGP
        # p(neg) = probability that circle is NOT an MGP
    # copies of all positive maps (defined as p(pos) above a certain threshold), in the "output" folder

# Images are placed in "analysis_images"


# import the necessary packages
import numpy as np
import argparse
import glob
import os,sys
import cv2


# set up arrays for output
all_circles_as_array = []
filename = []
cx = []
cy = []
cr = []

################################## parameters ##################################

################ hough.circles parameters ############
# global circles params
p1 = 20
# blur
b1 = 3
b2 = 3

# how much to increase bounding box around circle ROI
mult = 1.2

# define pass 1
min1 = 15
max1 = 40
dist1 = 20
p2_1 = 65

# define pass 2
min2 = 41
max2 = 70
dist2 = 40
p2_2 = 70

# define pass 3
min3 = 71
max3 = 130
dist3 = 70
p2_3 = 75

################## ML predict_proba cutoff parameter ############
# "cutoff" sets the minimum ML probability prediction
# for circles to be returned as positives

cutoff = 0.5

#################### define hough.circles function ##############

def find_circles(rmin, rmax, min_dist, p2):
    circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,min_dist,
                                param1=p1,param2=p2,minRadius=rmin,maxRadius=rmax)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            try:
                box = int(mult*r)
                box = int(mult*r)
                ROI = gray[y-box:y+box, x-box:x+box]
                ROI = cv2.resize(ROI, (64,64))
                all_circles_as_array.append(ROI)
                filename.append(imgname)
                cx.append(x)
                cy.append(y)
                cr.append(r)
            except:
                continue


###############################################################################
#########################  find circles!!!!!!!!  ##############################
###############################################################################

# load filenames
imgnames = sorted(glob.glob("input/***/*.jpg"))

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
    find_circles(min3, max3, dist3, p2_3)

# Construct the arrays
X , filename, cx, cy, r = np.array(all_circles_as_array), np.array(filename), np.array(cx), np.array(cy), np.array(cr)
print(":::::Circles found, input to array:::::")
print(("X array: ") + str(X.shape))

# reshape X array to 2-dimensions
nsamples, nx, ny = X.shape
X = X.reshape((nsamples, nx*ny))
print(":::::Array reshaped for analysis:::::")
print(("X array reshaped: ") + str(X.shape))

###############################################################################
######################## pass circles through ML pipeline #######################
###############################################################################

import pickle

# select model (mlp vs rf) and apply

pipe = pickle.load(open('ML_training/mlp.pkl', 'rb'))
# pipe = pickle.load(open('ML_training/rf.pkl', 'rb'))

y_pred = pipe.predict_proba(X)

########################################################################
######################## output results to csv #########################
########################################################################
# turn arrays into pandas dataframe of results
import pandas as pd
dffile = pd.DataFrame(filename)
dfy_pred = pd.DataFrame(y_pred)
dfcx = pd.DataFrame(cx)
dfcy = pd.DataFrame(cy)
dfcr = pd.DataFrame(cr)
df = pd.concat([dffile, dfy_pred, dfcx, dfcy, dfcr], axis=1)
df.columns = ['file', 'p(neg)', 'p(pos)', 'x', 'y', 'r']

# output csv with predictions
df.to_csv (r'output.csv', index = False)
print(":::::Prediction data (saved to output.csv):::::")
print(df)
########################################################################
################ copy positives to new folder for coding ################
########################################################################

from PIL import Image, ImageOps
dfpos = df.loc[df['p(pos)'] > cutoff]
positives = dfpos['file'].tolist()
def change_to_output(path):
    return os.path.join(os.path.split(os.path.dirname(os.path.dirname(path)))[0], 'output', os.path.basename(path))

for imgname in imgnames:
    if imgname in positives:
        img = cv2.imread(imgname)
        path_out = change_to_output(imgname)
        cv2.imwrite(path_out, img)
        print(("Output file written to: ") + path_out)
