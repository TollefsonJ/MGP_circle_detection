################## ML predict_proba cutoff parameter ############
# "cutoff" sets the minimum ML probability prediction
# for circles to be returned as positives
cutoff = 0.5


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


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

####### split data into train and test sets

X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from collections import Counter
print("Y counts: ")
print(sorted(Counter(y_train).items()))






####################################
###### scale and classify! ######
####################################

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# define and fit scaler
scaler = preprocessing.StandardScaler().fit(X_train)


######### define model and parameters (mlpclassifier) - use gridsearch
mlp = MLPClassifier(max_iter=500)
mlp_parameter_space = {
    'hidden_layer_sizes': [(100,50), (100,)],
    'activation': ['relu'],
    'solver': ['lbfgs'],
    'alpha': [(1e-5), (0.0001)],
    'learning_rate': ['constant','adaptive'],
    'random_state' : [0]
}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(mlp, mlp_parameter_space, n_jobs=-1)


pipe = Pipeline(steps =[('scaler',scaler) , ('search', grid)])


# run it
pipe.fit(X_train, y_train)

######## Best parameter set (for gridsearch)
print('Best parameters found:\n', grid.best_params_)

########## All gridsearch results
means = grid.cvults_['mean_test_score']
stds = grid.cvults_['std_test_score']
for mean, std, params in zip(means, stds, grid.cvults_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# test model
# set threshold for positive output in the predict_proba line

from sklearn.metrics import recall_score, accuracy_score, precision_score

y_true, y_pred = Y_test, (pipe.predict_proba(X_test)[:,1] >= cutoff).astype(bool)
# y_true, y_pred = Y_test, pipe.predict(X_test_scaled)


# print accuracy
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print(str("Recall score (tp / [tp+fn]) = ") + str(recall))
print(str("Precision score: ") + str(precision))
print(str("Accuracy score: ") + str(accuracy))

# get false pos and false neg
diff = np.subtract(y_pred, y_true)
false_pos = np.count_nonzero(diff == 1)
false_neg = np.count_nonzero(diff == -1)
total_pos = np.count_nonzero(y_pred == 1)
total_neg = np.count_nonzero(y_pred == 0)

print(str("Total positives identified: ") + str(total_pos))
print(str("Total negatives identified: ") + str(total_neg))
print(str("False positives: ") + str(false_pos))
print(str("False negatives: ") + str(false_neg))


# alternatively: use classification report
# from sklearn.metrics import classification_report
# print('Results on the test set:')
# print(classification_report(y_true, y_pred))


###### save it #######
from pickle import dump
# save the model
dump(pipe, open('mlp.pkl', 'wb'))
