# -*- coding: utf-8 -*-
# import modules & set up logging
import gensim, logging, os, re, glob, sys
import numpy as np
import pickle

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )

from util import data_accessor_util
from util import mysql_util


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
wvModel = gensim.models.KeyedVectors.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)


# read in stopwords
stoplist = set(line.strip() for line in open('/home/ubuntu/repo/stopwords.txt'))   
wordvector = ()

# pull from sql table 
q = """SELECT songs.songID, songs.track_id, songs.has_lyrics, genres.genre
		FROM songs
		LEFT JOIN genres ON genres.songID = songs.track_id
		WHERE genres.genre IS NOT NULL
		AND genres.genre NOT LIKE 'NULL';"""

data = mysql_util.execute_dict_query(q)

for row in data:
	# check to see if lyrics exist
	if row['has_lyrics']:

		with open("/home/ubuntu/lyrics/"+row['songID']+".lyrics") as fidx:
			sentences = [word.lower() for line in fidx for word in line.split()]

		# remove stop words, lowercase terms, remove periods and parentheses
		texts = [re.sub(r'[()]', '', word).rstrip('[.,]')  for word in sentences if word not in stoplist]

		for lyric in texts: 
			if lyric in wvModel.vocab:
				wordvector = np.append(wordvector, wvModel[lyric])

		wordvector = np.reshape(wordvector, (wordvector.shape[0]/300, 300))
		
		# save out one wordvector
		file = open('example_wordVector.pkl', 'w')
		pickle.dump(wordvector, file)
		file.close()

		# quit after one iteration for testing purposes
		sys.exit()
	