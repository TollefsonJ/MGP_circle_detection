# import images into X and Y arrays

from PIL import Image
import os
import numpy as np
import re
def get_data(path):
    all_images_as_array=[]
    label=[]
    for filename in os.listdir(path):
        if "jpg" in filename:
            try:
                if re.match(r'YES',filename):
                    label.append(1)
                else:
                    label.append(0)
                img=Image.open(path + filename)
                np_array = np.asarray(img)
                all_images_as_array.append(np_array)
            except:
                continue
    return np.array(all_images_as_array), np.array(label)

path_to_data_set = "training_images/FINAL/"

X, y = get_data(path_to_data_set)


# reshape X array to 2-dimensions

nsamples, nx, ny = X.shape
X = X.reshape((nsamples, nx*ny))


###########################################################################
###########################################################################
###########  NOW: SPLIT, NORMALIZE, RUN MODEL, SAVE  ####################
###########################################################################
###########################################################################


from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# split data into train and test sets

X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# define scaler
scaler = MinMaxScaler(feature_range=(0.0, 1.0))

# fit scaler on the training dataset
scaler.fit(X_train)

# transform the training dataset
X_train_scaled = scaler.transform(X_train)





# define model

from sklearn.neural_network import MLPClassifier

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)


clf.fit(X_train_scaled, y_train)

# test model
test = clf.predict(X_test)


# how'd it do?
from sklearn.metrics import accuracy_score
y_pred = test
y_true = Y_test

# print accuracy
score = accuracy_score(y_true, y_pred, normalize=True)
print(str("Accuracy = ") + str(score))




# ###### save it #######
# from pickle import dump
# # save the model
# dump(clf, open('model.pkl', 'wb'))
#
# # save the scaler
# dump(scaler, open('scaler.pkl', 'wb'))



## sources ##
# https://www.codespeedy.com/prepare-your-own-data-set-for-image-classification-python/
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
