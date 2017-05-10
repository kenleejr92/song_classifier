# -*- coding: utf-8 -*-
# import modules & set up logging
# @ Author Farzan Memarian, using Alan's code
import gensim, logging, os, re, glob, sys
import numpy as np
import pickle
import pdb
import csv
import pandas as pd
from tqdm import tqdm	
from collections import Counter

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )

from util import data_accessor_util as access_data
from util import mysql_util

wvModel = gensim.models.KeyedVectors.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)

# read in stopwords
stoplist = set(line.strip() for line in open('/home/ubuntu/repo/stopwords.txt'))   

(train_X, train_Y, train_le, test_X, test_Y, test_le) = access_data.get_data_sets_w_lyrics()
size_training = train_X.shape[0]
size_test = test_X.shape[0]
all_data = pd.concat((train_X, test_X), axis=0, ignore_index = True)
document = []

for idNum in tqdm(np.arange(0, train_X.shape[0], 1)):
	# check to see if lyrics exist

	songID = train_X['songID'][idNum]

	#print "/home/ubuntu/wordVectors/"+songID+".pkl"
	if os.path.exists("/home/ubuntu/wordVectors/"+songID+".npy"):

		with open("/home/ubuntu/lyrics/"+songID+".lyrics") as fidx:
			sentences = [word.lower() for line in fidx for word in line.split()]

	# remove stop words, lowercase terms, remove periods and parentheses
		texts = [re.sub(r'[()]', '', word).rstrip('[.,]')  for word in sentences if word not in stoplist]
		for lyric in texts: 
				if "'" not in lyric:
					document.extend([lyric])

n_top_words = 1000
top_words_store = np.zeros((n_top_words,2))
c = Counter(document)
most_common = c.most_common(n_top_words)
df_most_common = pd.DataFrame(most_common, columns=["word", "count"])

print "most common words, tuple format:"
print df_most_common, "\n\n"

# save the document
np.save('/home/ubuntu/allWords/document.txt', document, allow_pickle=True)		

file_name1 = "popular_words_counts.csv"
path_to_file1 = "/home/ubuntu/repo/src/data/" + file_name1
file_name2 = "popular_words_counts.pickle"
path_to_file2 = "/home/ubuntu/repo/src/data/" + file_name2
df_most_common.to_csv(path_to_file1, sep='\t')
df_most_common.to_pickle(path_to_file2)

