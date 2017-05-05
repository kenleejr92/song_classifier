"""
xgboost model

- Performs XGBoost to classify the data

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
from xgboost import XGBClassifier
import PCA
# ----------------
#    models
# ----------------


DEBUG = 1

# method used
method = "XGBClassifier"


# reading the data from sql table using the get_all_data method
var_percentage = 0.99
(X_train, y_train, train_le, X_test, y_test, test_le) = PCA.read_data_perform_pca(var_percentage)
classes = list(test_le.classes_)


# iterate over classifiers
if DEBUG == 1: print "now performing: ", method
start = time.time()

# fit model no training data

clf = xgboost.XGBClassifier(\
	max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', \
	nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, \
	colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)

# use a full grid search over all parameters
param_grid = {"max_depth": [6,15],
	"learning_rate": [100,200]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
grid_search.fit(X_train, y_train)

y_test_predict = grid_search.predict(X_test)
y_train_predict = grid_search.predict(X_train)
test_accuracy = accuracy_score(y_test, y_test_predict)
train_accuracy = accuracy_score(y_train, y_train_predict)

end = time.time()
elasped_time = end - start

file = open("XGBClassifier.txt","w")
file.close()
print "execution time of %s was %s seconds" %(method, elasped_time)
print "train accuracy of method %s is %s" % (method, train_accuracy)
print "test accuracy of method %s is %s" % (method, test_accuracy)
print classification_report(y_test, y_test_predict, target_names=classes)

myFile = open("XGBClassifier.txt","a")
print >> myFile, "execution time of %s was %s seconds\n" %(method, elasped_time)
print >> myFile, "train accuracy of method %s is %s \n" % (method, train_accuracy)
print >> myFile, "test accuracy of method %s is %s \n" % (method, test_accuracy)
print >> myFile, classification_report(y_test, y_test_predict, target_names=classes)
print >> myFile, "\n\n\n"
print >> myFile, "best params: ", grid_search.best_params_
myFile.close()
myFile.close()