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

# define paramets to search
mlp = MLPClassifier(max_iter=300)
parameter_space = {
    'hidden_layer_sizes': [(5,2), (100,)],
    'activation': ['relu'],
    'solver': ['lbfgs'],
    'alpha': [1e-5],
    'learning_rate': ['constant','adaptive'],
    'random_state':['1']
}

# set up to find optimal model parameters

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)

# run it to find optimal parameters
clf.fit(X_train_scaled, y_train)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))





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
