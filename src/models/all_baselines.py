"""
baseline models

- Each baseline classification model has been defined as a function
- they call run_models from util module
- run_models in tern calls pca_util to perform pca on data
- this should be run from src folder

@author - Farzan Memarian
"""

import sys, os
import numpy as np
import time
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from util import run_model_util
from xgboost import XGBClassifier


# ----------------
#    models
# ----------------
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier



def xgboost_func():
	method = "XGBClassifier"
	# create file_name for saving results
	possibles = globals().copy()
	possibles.update(locals())
	method_func = possibles.get(method)
	file_name = method + ".txt"

	# setting the model parameters
	clf = method_func(
		max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='multi:softmax', 
		nthread=5, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, 
		colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)

	# use a full grid search over following parameters
	param_grid = {"max_depth": [3, 10],
	              "n_estimators": [100, 200]}

	run_model_util.run_model(clf, param_grid, method, file_name)



def grad_boosting_func():
	method = "GradientBoostingClassifier"
	# create file_name for saving results
	possibles = globals().copy()
	possibles.update(locals())
	method_func = possibles.get(method)
	file_name = method + ".txt"

	# setting the model parameters
	clf = method_func(
		loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
	    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
	    min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, 
	    init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, 
	    warm_start=False, presort='auto')

	# use a full grid search over following parameters
	param_grid = {"max_depth": [3, 10],
	              "n_estimators": [50, 200]}

	run_model_util.run_model(clf, param_grid, method, file_name)


def linear_svm_func():
	method = "LinearSVC"
	# create file_name for saving results
	possibles = globals().copy()
	possibles.update(locals())
	method_func = possibles.get(method)
	file_name = method + ".txt"

	# setting the model parameters
	clf = method_func(
		penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0,
		multi_class='ovr', fit_intercept=True, intercept_scaling=1,
		class_weight=None, verbose=0, random_state=None, max_iter=1000)

	# use a full grid search over following parameters
	param_grid = {"C": [0.5, 1.0, 5.0]}
	run_model_util.run_model(clf, param_grid, method, file_name)


def mlp_func():
	method = "MLPClassifier"
	# create file_name for saving results
	possibles = globals().copy()
	possibles.update(locals())
	method_func = possibles.get(method)
	file_name = method + ".txt"

	# setting the model parameters
	clf = method_func(
		activation='relu', solver='adam', alpha=0.0001, 
		batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, 
		max_iter=400, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
		momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
		beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	# use a full grid search over all parameters
	param_grid = {"hidden_layer_sizes": [(15,6),(40,6)],
	              "activation": ['relu', 'logistic']
	}
	run_model_util.run_model(clf, param_grid, method, file_name)


def logistic_reg_func():
	method = "LogisticRegression"
	# create file_name for saving results
	possibles = globals().copy()
	possibles.update(locals())
	method_func = possibles.get(method)
	file_name = method + ".txt"

	# setting the model parameters
	clf = method_func(
		penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
		intercept_scaling=1, class_weight=None, random_state=None,
		solver='liblinear', max_iter=100, multi_class='ovr',
		verbose=0, warm_start=False, n_jobs=1)

	# use a full grid search over all parameters
	param_grid = {"C": [0.1, 1.0, 10.0],
	              "penalty": ['l2', 'l1']}
	run_model_util.run_model(clf, param_grid, method, file_name)


def random_forest_func():
	method = "RandomForestClassifier"
	# create file_name for saving results
	possibles = globals().copy()
	possibles.update(locals())
	method_func = possibles.get(method)
	file_name = method + ".txt"

	# setting the model parameters
	clf = method_func(
		n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
		min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
		max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, 
		oob_score=False, n_jobs=1, random_state=None, verbose=0, 
		warm_start=False, class_weight=None)

	# use a full grid search over all parameters
	param_grid = {"n_estimators": (10,100), 
	              "max_depth": (3,10),
                  "criterion": ["gini", "entropy"]}
	run_model_util.run_model(clf, param_grid, method, file_name)


def AdaBoost_func():
	method = "AdaBoostClassifier"
	# create file_name for saving results
	possibles = globals().copy()
	possibles.update(locals())
	method_func = possibles.get(method)
	file_name = method + ".txt"

	# setting the model parameters
	clf = method_func(
		base_estimator=None, n_estimators=50, learning_rate=1.0,
		algorithm='SAMME.R', random_state=None)

	# use a full grid search over all parameters
	param_grid = {"n_estimators": np.arange(20,140,40),
	              "learning_rate": [ 0.1, 1]}
	run_model_util.run_model(clf, param_grid, method, file_name)