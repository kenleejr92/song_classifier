"""
Random Forest

@author - Tim Mahler
"""

#-------------------------
# Libs
#-------------------------

# External
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql.cursors
import sys, os
from tqdm import tqdm
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Internal
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util

#-------------------------
# Globals
#-------------------------

# Get data
(train_X, train_Y, train_le, test_X, test_Y, test_le) = data_accessor_util.get_all_data_sets()

classes = list(test_le.classes_)
print test_le.inverse_transform([0, 1, 2, 3, 4, 5])

print classes

# Convert to numpy
(train_X, train_Y, test_X, test_Y) = data_accessor_util.convert_data_sets_to_numpy(train_X, train_Y, test_X, test_Y)

parameters = {'n_estimators': np.arange(10,210,50),'max_depth': np.arange(3,9,3)}


print train_X

print "SHAPE"
print train_X.shape

print parameters

#-------------------------
# Functions
#-------------------------

# Main func
rf = RandomForestClassifier(max_depth=20, n_estimators=200, max_features=25, n_jobs=12)

print "Running GridSearchCV"
best_fit = GridSearchCV(rf, parameters, cv=3, verbose=10, n_jobs=12)

best_fit.fit(train_X, train_Y)

# Print the estimator
best_estimator = best_fit.best_estimator_
print best_estimator

# Get predictions for model
y_pred_train = best_fit.predict(train_X)
y_pred_test = best_fit.predict(test_X)

print "Got predictions"

accuracy_train = best_fit.score(train_X, train_Y)
accuracy_test = best_fit.score(test_X, test_Y)


print "RESULTS\n*******************"
print "\naccuracy_train = %f"%(accuracy_train)
print "accuracy_test = %f"%(accuracy_test)
print

print(classification_report(test_Y, y_pred_test, target_names=classes))
