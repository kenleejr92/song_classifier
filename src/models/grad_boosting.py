"""
GradientBoostingClassifier model

- Performs Gradient Boosting to classify the data

@author - Farzan Memarian
"""

import sys, os
import time
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from util import run_model_util
from sklearn.ensemble import GradientBoostingClassifier




# ----------------
#    model
# ----------------
method = "GradientBoostingClassifier"
# create file_name for saving results
possibles = globals().copy()
possibles.update(locals())
method_func = possibles.get(method)
file_name = method + ".txt"

# setting the model parameters
clf = method_func(\
	loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, \
    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, \
    min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, \
    init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, \
    warm_start=False, presort='auto')

# use a full grid search over following parameters
param_grid = {"max_depth": [3, 10],
              "n_estimators": [50, 200]}

run_model_util.run_model(clf, param_grid, method)


