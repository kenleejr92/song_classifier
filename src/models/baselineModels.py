'''
performs simple classification models on the dataset as baseline
does not use lyrics
   
@author - Farzan Memarian
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymysql.cursors
import sys, os
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# ----------------
#    models
# ----------------
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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


# reading the data from sql table using the get_all_data method
df = data_accessor_util.get_all_data()
X = df.drop(['genre','year'], axis=1)
y = df['genre']
y = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(y))


h = .02  # step size in the mesh
# list of methods used, they correspond to the classifiers list
methods = ["Logistic Regression","KNN", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net, MLP", "AdaBoost",
         "Gaussian Naive Bayes", "QDA"]
# add these two methods as well
# Xgboost
# Gradient boosting


classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

       

# figure = plt.figure(figsize=(27, 9))

# preprocess dataset, split into training and test part
X = preprocessing.StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
import pdb; pdb.set_trace()

accuracy_resuts = []
# iterate over classifiers

for name, clf in zip(methods, classifiers):
    # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    accuracy_resuts.append((name, accuracy))

# plt.tight_layout()
#plt.show()
print accuracy_resuts
file = open("store_accuracy.txt","w")
for method, value in accuracy_resuts:
    entry = "accuracy of " + method + " = " + str(value) + "\n"
    file.write(entry)
file.close()