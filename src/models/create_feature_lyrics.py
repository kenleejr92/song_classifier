'''
Developing a deep learning model based on words
note: this must be run form src directory
@ Author: Farzan Memarian
'''

# import packages
import sys, os, pickle
import pandas as pd
import numpy as np
import pdb
import gensim, logging, os, re, glob, sys
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import LSTM
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm


sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util as access_data

# get data
(train_X, train_Y, train_le, test_X, test_Y, test_le) = access_data.get_data_sets_w_lyrics()
size_training = train_X.shape[0]
size_test = test_X.shape[0]
all_data_X = pd.concat((train_X, test_X), axis=0, ignore_index = True)
all_data_Y = pd.concat((train_Y, test_Y), axis=0, ignore_index = True)
# read in stopwords
stoplist = set(line.strip() for line in open('/home/ubuntu/repo/stopwords.txt'))   

#-----------------------------------
#  create features out of genres
#-----------------------------------

# read most popular words
file_name2 = "popular_words_counts.pickle"
path_to_file2 = "/home/ubuntu/repo/src/data/" + file_name2
df_most_common = pd.read_pickle(path_to_file2)
most_common_words = df_most_common['word']
n_common_words = most_common_words.shape[0]
features_df = pd.DataFrame(columns = np.arange(0, n_common_words + 2))

# inputs = np.arange(0, all_data_X.shape[0], 1)
range_limit = all_data_X.shape[0]
inputs = np.arange(0, range_limit, 1)

def create_features(idNum):
	# read songID
	print "loop number %s being executed" %(idNum)
	songID = all_data_X['songID'][idNum]
	with open("/home/ubuntu/lyrics/"+songID+".lyrics") as fidx:
		sentences = [word.lower() for line in fidx for word in line.split()]

	# remove stop words, lowercase terms, remove periods and parentheses
	texts = [re.sub(r'[()]', '', word).rstrip('[.,]')  for word in sentences if word not in stoplist]
	# for lyric in texts: 
	row = [songID]
	for i in range(n_common_words):
		count = [texts.count(most_common_words[i])]
		row.extend(count)
	# here we add the class label
	row.extend([ all_data_Y[0][idNum]] )
	# features_df.loc[i] = row
	return row
	

num_cores = 10 # multiprocessing.cpu_count()

features_list = Parallel(n_jobs=num_cores)(delayed(create_features)(i) for i in inputs)
features_array = np.array(features_list)
features_df = pd.DataFrame(features_array)

pdb.set_trace()
file_name = "lyrics_feature_df.pickle"
path_to_file = "/home/ubuntu/repo/src/data/" + file_name
features_df.to_pickle(path_to_file)
pdb.set_trace()