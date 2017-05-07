"""
XGBClassifier model

- Performs XGBClassifier to classify the data

@author - Farzan Memarian
"""

import sys, os
import time
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from util import run_model_util
from xgboost import XGBClassifier


# ----------------
#    model
# ----------------
method = "XGBClassifier"


# setting the model parameters
clf = method_func(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='multi:softmax', \
nthread=5, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, \
colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)

# use a full grid search over following parameters
param_grid = {"max_depth": [3, 10],\
              "n_estimators": [100, 200]}

run_model_util.run_model(clf, param_grid, method)