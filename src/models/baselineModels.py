'''
performs simple classification models on the dataset as baseline
does not use lyrics
   
@author - Farzan Memarian
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import logistic_regression_path
from sklearn.gaussian_process import GaussianProcessClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process.kernels import RBF
import pymysql.cursors
# from sklearn.preprocessing import 


# establish connection to sql server
connection = pymysql.connect(host='localhost', user='root', password='root' ,db='songs')
cursor = connection.cursor()






# **************************
# here features are imported from features table, genre not included 
# **************************
sql = ''' SELECT * FROM `features_subset` '''
cursor.execute(sql)
# read the features from the sql table
the_data = cursor.fetchall()
# put the features in a panda DataFrame
msd_df = pd.DataFrame(list(the_data))
# read the column titles from the sql table
num_fields = len(cursor.description)
feature_names_unencoded = [i[0] for i in cursor.description]
feature_names = [s.encode('utf-8') for s in feature_names_unencoded ]
feature_fields = feature_names[1:-1]  # This line might need to be modified 
                                    # based on the columns of the sql table
msd_df.columns = feature_names

# **************************
# read genres from genre table 
# **************************
sql = ''' SELECT * FROM `genres` '''
cursor.execute(sql)
# read the features from the sql table
the_data = cursor.fetchall()
# put the features in a panda DataFrame
genre_df = pd.DataFrame(list(the_data))
# read the column titles from the sql table
num_fields = len(cursor.description)
genre_names_unencoded = [i[0] for i in cursor.description]
genre_table_names = [s.encode('utf-8') for s in genre_names_unencoded ]
genre_fields = genre_table_names[-1]  # This line might need to be modified 

genre_df.columns = genre_table_names
X = msd_df[feature_fields]    # DataFrame storing the features
y = genre_df[genre_fields]      # DataFrame storing the outputs
import pdb; pdb.set_trace()
cursor.close()
connection.close()
# **************************
# NOW DATA HAS BEEN IMPORTED
# **************************

h = .02  # step size in the mesh

methods = ["Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

# add these two methods as well
# Xgboost
# Gradient boosting


classifiers = [
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

linearly_separable = (X, y)

datasets = [linearly_separable]            

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(methods, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()

