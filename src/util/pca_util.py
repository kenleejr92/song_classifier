
"""
PCA model

- Performs PCA for feature reduction

@author - Farzan Memarian
"""

import numpy as np
import pandas as pd
import matplotlib
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
from sklearn.decomposition import PCA



#--------------------
#      Functions
#--------------------
def print_pca_variance():
	# reading the data from sql table using the get_all_data method
	X_train, y_train, train_le, X_test, y_test, test_le = data_accessor_util.get_all_data_sets()
	classes = list(test_le.classes_)

	# Converting data to numpy
	(X_train, y_train, X_test, y_test) = data_accessor_util.convert_data_sets_to_numpy(X_train, y_train, X_test, y_test)

	n_features = X_test.shape[1]
	# this block need to be used once to create the plots
	pca = PCA(n_components=n_features)
	pca.fit(X_train)
	var = pca.explained_variance_ratio_
	cumulative_var = np.cumsum(var)

	# To enable creating figures on the server
	matplotlib.use('Agg')
	fig = plt.figure(1)
	plt.plot(var)
	plt.title('individual scree plot')
	plt.xlabel('principal components')
	plt.ylabel('proportion of variance explained')
	plt.savefig("proportion_variance.jpg")
	fig2 = plt.figure(2)
	plt.plot(cumulative_var)
	plt.title('commulative scree plot')
	plt.xlabel('principal components')
	plt.ylabel('commulative proportion of variance explained')    
	plt.savefig("proportion_Variance_com.jpg")



def read_data_perform_pca(var_percentage = 0.95):
	# reading the data from sql table using the get_all_data method
	X_train, y_train, train_le, X_test, y_test, test_le = data_accessor_util.get_all_data_sets()
	classes = list(test_le.classes_)

	# Converting data to numpy
	(X_train, y_train, X_test, y_test) = data_accessor_util.convert_data_sets_to_numpy(X_train, y_train, X_test, y_test)
	n_features = X_test.shape[1]
	# this block need to be used once to create the plots
	pca = PCA(n_components=n_features)
	pca.fit(X_train)
	var = pca.explained_variance_ratio_
	cumulative_var = np.cumsum(var)

	# arg max returns index, add 1 b/c indices start at 0
	percentage_retained = var_percentage
	N_reduced = np.argmax(cumulative_var>percentage_retained)+1;

	pca = PCA(n_components=N_reduced)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	return X_train, y_train, train_le, X_test, y_test, test_le