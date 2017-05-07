"""
run models

- given a model, it uses pca_util to import data and perform PCA on it
- then runs the model and prints the resulst

@author - Farzan Memarian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql.cursors
import sys, os
import time
from tqdm import tqdm
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from time import gmtime, strftime


# ----------------
#    user defined packages
# ----------------
from util import data_accessor_util
from util import PCA



def run_model(clf, param_grid, method, file_name):
	DEBUG = 1

	# reading the data from sql table using the get_all_data method
	var_percentage = 0.95   # percentage of variance retained by PCA
	(X_train, y_train, train_le, X_test, y_test, test_le) = \
							PCA.read_data_perform_pca(var_percentage)
	classes = list(test_le.classes_)

	if DEBUG == 1: print "now performing: ", method
	start = time.time()

	# run grid search
	grid_search = GridSearchCV(clf, param_grid=param_grid)
	# fit model to training data
	grid_search.fit(X_train, y_train)

	y_test_predict = grid_search.predict(X_test)
	y_train_predict = grid_search.predict(X_train)
	test_accuracy = accuracy_score(y_test, y_test_predict)
	train_accuracy = accuracy_score(y_train, y_train_predict)

	end = time.time()
	elasped_time = end - start

	#print on screen
	print "execution time of %s was %s seconds" % (method, elasped_time)
	print "train accuracy of method %s is %s" % (method, train_accuracy)
	print "test accuracy of method %s is %s" % (method, test_accuracy)
	print classification_report(y_test, y_test_predict, target_names=classes)

	# create the file for storing results if don't exist
	# remove existing file and recreate 
	path_to_file = "./results/" + file_name
	file = open(path_to_file, "w")
	file.close()

	myFile = open(path_to_file,"a")
	print >> myFile, "current time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "\n"
	print >> myFile, "model used: ", method
	print >> myFile, "percentage variance retained by PCA: %s \n" % (var_percentage)
	print >> myFile, "execution time of %s was %s seconds\n" % (method, elasped_time)
	print >> myFile, "train accuracy of method %s is %s \n" % (method, train_accuracy)
	print >> myFile, "test accuracy of method %s is %s \n" % (method, test_accuracy)
	print >> myFile, classification_report(y_test, y_test_predict, target_names=classes)
	print >> myFile, "\n\n\n"
	print >> myFile, "best params: ", grid_search.best_params_
	print >> myFile, "\n\n\n"
	print >> myFile, "confusion matrix: \n", confusion_matrix(y_test, y_test_predict), "\n"
	print >> myFile, "\n\n\n"
	print >> myFile, "cv_results: \n", grid_search.cv_results_, "\n"
	print >> myFile, "\n\n\n"
	print >> myFile, "best estimator: \n", grid_search.best_estimator_, "\n"
	print >> myFile, "\n\n\n"
	print >> myFile, "best score: \n", grid_search.best_score_, "\n"
	print >> myFile, "\n\n\n"
	print >> myFile, "number of cross-validation splits: \n", \
						grid_search.n_splits_, "\n"
	myFile.close()
