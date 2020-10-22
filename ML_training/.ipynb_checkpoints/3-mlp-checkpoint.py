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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


################ resample train array ###########
from collections import Counter
print("Y train array: ")
print(sorted(Counter(y_train).items()))

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

over = SMOTE(sampling_strategy = .5)
under = RandomUnderSampler(sampling_strategy=1)
X_train, y_train = over.fit_resample(X_train, y_train)
X_train, y_train = under.fit_resample(X_train, y_train)
print("Train data resampled.")

from collections import Counter
print("Y train array: ")
print(sorted(Counter(y_train).items()))

print("Y test array: ")
print(sorted(Counter(y_test).items()))


####################################
###### scale and classify! ######
####################################

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# define and fit scaler
scaler = preprocessing.StandardScaler()


################## define and run model itself, after you've found good params ########
mlp = MLPClassifier(hidden_layer_sizes= (100, ),
                    activation = 'relu',
                    solver = 'lbfgs',
                    alpha = 1e-5,
                    learning_rate = 'constant',
                    random_state = 0)

pipe = Pipeline(steps =[('scaler',scaler) , ('MLPClassifier', mlp)])


# run it
pipe.fit(X_train, y_train)

# test model
# set threshold for positive output in the predict_proba line

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

y_true, y_pred = y_test, (pipe.predict_proba(X_test)[:,1] >= cutoff).astype(bool)


############### print accuracy
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(str("Recall score (tp / [tp+fn]) = ") + str(recall))
print(str("Precision score: ") + str(precision))
print(str("Accuracy score: ") + str(accuracy))
print(str("F1 score: ") + str(f1))
################ get false pos and false neg
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
# from pickle import dump
# save the model
# dump(pipe, open('mlp.pkl', 'wb'))
