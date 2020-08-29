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
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# split data into train and test sets

X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# define scaler
scaler = preprocessing.StandardScaler().fit(X_train)

# or: use minmax scaler
# scaler = preprocessing.MinMaxScaler()

# transform the training dataset
X_train_scaled = scaler.transform(X_train)





# define model

from sklearn.neural_network import MLPClassifier

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(100,), random_state=1, activation = 'relu', learning_rate='constant')


# train the model
clf.fit(X_train_scaled, y_train)

# test model
X_test_scaled = scaler.transform(X_test)

from sklearn.metrics import accuracy_score
y_true, y_pred = Y_test, clf.predict(X_test_scaled)

# print accuracy
score = accuracy_score(y_true, y_pred, normalize=True)
print(str("Accuracy = ") + str(score))

# get false pos and false neg
diff = np.subtract(y_pred, y_true)
false_pos = np.count_nonzero(diff == 1)
false_neg = np.count_nonzero(diff == -1)

print(str("False positives: ") + str(false_pos))
print(str("False negatives: ") + str(false_neg))


# alternatively: use classification report
# from sklearn.metrics import classification_report
# print('Results on the test set:')
# print(classification_report(y_true, y_pred))




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
