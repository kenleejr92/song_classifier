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
from tqdm import tqdm


sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util as access_data

# get data
(train_X, train_Y, train_le, test_X, test_Y, test_le) = access_data.get_data_sets_w_lyrics()
size_training = train_X.shape[0]
size_test = test_X.shape[0]
all_data = pd.concat((train_X, test_X), axis=0, ignore_index = True)

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
features_df = pd.DataFrame(columns = np.arange(0,n_common_words+2))

for idNum in tqdm(np.arange(0, all_data.shape[0], 1)):
	# read songID
	songID = all_data['songID'][idNum]
	with open("/home/ubuntu/lyrics/"+songID+".lyrics") as fidx:
		sentences = [word.lower() for line in fidx for word in line.split()]

	# remove stop words, lowercase terms, remove periods and parentheses
	texts = [re.sub(r'[()]', '', word).rstrip('[.,]')  for word in sentences if word not in stoplist]
	# for lyric in texts: 
	row = [songID]
	for i in range(n_common_words):
		count = [texts.count(most_common_words[i])]
		row.extend(count)
	row.extend([ train_Y[0][idNum]] )
	features_df.loc[idNum] = row

file_name = "lyrics_feature_df.pickle"
path_to_file = "/home/ubuntu/repo/src/data/" + file_name
features_df.to_pickle(path_to_file)


pdb.set_trace()