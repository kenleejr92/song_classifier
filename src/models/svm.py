"""
LinearSVC model

- Performs linear SVM to classify the data

@author - Farzan Memarian
"""

import sys, os
import time
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from util import run_model_util
from sklearn.svm import LinearSVC


# ----------------
#    models
# ----------------
method = "LinearSVC"
possibles = globals().copy()
possibles.update(locals())
method_func = possibles.get(method)
file_name = method + ".txt"

# setting the model parameters
clf = method_func(\
	penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', \
	fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, \
	random_state=None, max_iter=1000)

# use a full grid search over following parameters
param_grid = {"penalty": []}

run_model_util.run_model(clf, param_grid, method, file_name)







