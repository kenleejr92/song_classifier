"""
AdaBoosClassifer model

- Performs MLP to classify the data

@author - Farzan Memarian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql.cursors
import sys, os
from   tqdm import tqdm
import time
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import PCA


# ----------------
#    models
# ----------------
from sklearn.neural_network import MLPClassifier

DEBUG = 1



# reading the data from sql table using the get_all_data method
var_percentage = 0.99
(X_train, y_train, train_le, X_test, y_test, test_le) = PCA.read_data_perform_pca(var_percentage)
classes = list(test_le.classes_)

# method used
method = "MLPClassifier"

# iterate over classifiers
if DEBUG == 1: print "now performing: ", method
start = time.time()
clf = MLPClassifier(\
	activation='relu', solver='adam', alpha=0.0001, \
	batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, \
	max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,\
	momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, \
	beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# use a full grid search over all parameters
param_grid = {"hidden_layer_sizes": (15,6)}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
grid_search.fit(X_train, y_train)

y_test_predict = grid_search.predict(X_test)
y_train_predict = grid_search.predict(X_train)
test_accuracy = accuracy_score(y_test, y_test_predict)
train_accuracy = accuracy_score(y_train, y_train_predict)

end = time.time()
elasped_time = end - start

file = open("MLPClassifier.txt","w")
file.close()
print "execution time of %s was %s seconds" %(method, elasped_time)
print "train accuracy of method %s is %s" % (method, train_accuracy)
print "test accuracy of method %s is %s" % (method, test_accuracy)
print classification_report(y_test, y_test_predict, target_names=classes)

myFile = open("MLPClassifier.txt","a")
print >> myFile, "execution time of %s was %s seconds\n" %(method, elasped_time)
print >> myFile, "train accuracy of method %s is %s \n" % (method, train_accuracy)
print >> myFile, "test accuracy of method %s is %s \n" % (method, test_accuracy)
print >> myFile, classification_report(y_test, y_test_predict, target_names=classes)
print >> myFile, "\n\n\n"
print >> myFile, "best params: ", grid_search.best_params_
myFile.close()
myFile.close()