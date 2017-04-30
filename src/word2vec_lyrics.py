# -*- coding: utf-8 -*-
# import modules & set up logging
import gensim, logging, os, re, glob, sys
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

files = glob.glob('/home/ubuntu/lyrics/*')
#print files

# read in stopwords
stoplist = set(line.strip() for line in open('/home/ubuntu/repo/stopwords.txt'))   
wordvector = ()

wvModel = gensim.models.KeyedVectors.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)

for f in files:
	print f
	with open(f) as fidx:
		sentences = [word.lower() for line in fidx for word in line.split()]

	# remove stop words, lowercase terms, remove periods and parentheses
	texts = [re.sub(r'[()]', '', word).rstrip('[.,]')  for word in sentences if word not in stoplist]
	
	for lyric in texts:                            
		if lyric in wvModel.vocab:
			print lyric
	    	wordvector = np.append(wordvector,  wvModel[lyric], axis = 0)

	print wordvector.shape
	sys.exit()
	