# input to array, skipping extreme values

# import images into X and Y arrays
import cv2
from PIL import Image
import os
import numpy as np
import re
def get_data(path, lab):
    all_images_as_array=[]
    label=[]
    for filename in os.listdir(path):
        if "jpg" in filename:
           #if "_194" not in filename and "_195" not in filename:
                im = cv2.imread(path + filename)
                red = im[:,:,2]
                mean = np.mean(red)
                std = np.std(red)

                # for 2.5/97.5-percentile
                # if mean > 133.5 and mean < 236 and std > 9.6 and std < 34:
                # for 1/99-percentile
                # if mean > 127.57779687499999 and mean <  246.38062499999998 and std > 7.848734708324726 and std < 38.60616608163898: 
                # for 99.9 percentile
                if mean > 120.5 and mean < 249 and std > 5.5 and std < 40:

                    try:
                        img=cv2.imread(path + filename)
                        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                        all_images_as_array.append(gray)
                        label.append(lab)
                    except:
                        continue
    return np.array(all_images_as_array), np.array(label)


path_to_pos = "training_images/" + set + "/YES2/"
path_to_neg = "training_images/" + set + "/NO2/"


# input each class
X1 , y1 = get_data(path_to_pos, 1)
X0, y0 = get_data(path_to_neg, 0)

# combine classes
X = np.concatenate((X1, X0), axis = 0)
y = np.concatenate((y1, y0), axis = 0)


print(X.shape)
print(y.shape)

# reshape X array to 2-dimensions
nsamples, nx, ny = X.shape
X = X.reshape((nsamples, nx*ny))
print(X.shape)

print("Mean of y: ", np.mean(y))
print("Y count_nonzero: ", np.count_nonzero(y))