###############
# this the pipeline example: https://scikit-neuralnetwork.readthedocs.io/en/latest/guide_sklearn.html#example-pipeline
# using the NN setup from the "classification" section of this example:https://scikit-neuralnetwork.readthedocs.io/en/latest/guide_model.html
# this uses binary classification

from sknn.mlp import Classifier, Layer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipeline = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
        ('neural network', Classifier(
            layers=[
                Layer("Maxout", units=100, pieces=2),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=25)])

# run the pipeline
pipeline.fit(X_train, y_train)

# test to see how it works
y_predict = pipeline.predict(X_test)
 print(metrics.accuracy_score(y_test,y_pred))


# save output of trained network as pickle
import pickle
pickle.dump(pipeline, open('FMGP+ind.pkl', 'wb'))
