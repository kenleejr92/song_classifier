# -*- coding: utf-8 -*-
# import modules & set up logging
import gensim, logging, os, re, glob, sys
import numpy as np
import pickle
<<<<<<< Updated upstream
from tqdm import tqdm	
=======
#from tqdm import tqdm	
>>>>>>> Stashed changes

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )

from util import data_accessor_util as access_data
from util import mysql_util


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
wvModel = gensim.models.KeyedVectors.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)

# read in stopwords
stoplist = set(line.strip() for line in open('/home/ubuntu/repo/stopwords.txt'))   

(train_X, train_Y, train_le, test_X, test_Y, test_le) = access_data.get_data_sets_w_lyrics()

'''
# pull from sql table 
q = """SELECT songs.songID, songs.track_id, songs.has_lyrics, genres.genre
		FROM songs
		LEFT JOIN genres ON genres.songID = songs.track_id
		WHERE genres.genre IS NOT NULL
		AND genres.genre NOT LIKE 'NULL';"""

data = mysql_util.execute_dict_query(q)
'''

<<<<<<< Updated upstream
for idNum in tqdm(np.arange(int(sys.argv[1]),int(sys.argv[2]))):
	# check to see if lyrics exist

	songID = train_X['songID'][idNum]
	songID = 'SOELATB12D021903EE'
	wordvector = ()


	#print "/home/ubuntu/wordVectors/"+songID+".pkl"
	if os.path.exists("/home/ubuntu/wordVectors/"+songID+".npy"):
=======
for idNum in np.arange(int(sys.argv[1]),int(sys.argv[2])):
	# check to see if lyrics exist

	songID = train_X['songID'][idNum]

	#print "/home/ubuntu/wordVectors/"+songID+".pkl"
	if not os.path.exists("/home/ubuntu/wordVectors/"+songID+".pkl"):
>>>>>>> Stashed changes

		with open("/home/ubuntu/lyrics/"+songID+".lyrics") as fidx:
			sentences = [word.lower() for line in fidx for word in line.split()]

	# remove stop words, lowercase terms, remove periods and parentheses
		texts = [re.sub(r'[()]', '', word).rstrip('[.,]')  for word in sentences if word not in stoplist]

<<<<<<< Updated upstream
		print texts
		count = 0
		for lyric in texts: 
			if count > 29:
=======
		count = 0
		for lyric in texts: 
			if count > 24:
>>>>>>> Stashed changes
				break
			if lyric in wvModel.vocab:
				print lyric
				wordvector = np.append(wordvector, wvModel[lyric])
				count += 1
<<<<<<< Updated upstream
		sys.exit()
		if count > 0:
			wordvector = np.reshape(wordvector, (wordvector.shape[0]/300, 300))
			#np.save('/home/ubuntu/wordVectors/'+songID+'.npy', wordvector, allow_pickle=False)

=======

		wordvector = np.reshape(wordvector, (wordvector.shape[0]/300, 300))

>>>>>>> Stashed changes
		# save out one wordvector
		#file = open('/home/ubuntu/wordVectors/'+songID+'.pkl', 'w')
		#pickle.dump(wordvector, file)
		#file.close()

<<<<<<< Updated upstream
=======
		np.save('/home/ubuntu/wordVectors/'+songID+'.npy', wordvector, allow_pickle=False)
>>>>>>> Stashed changes
		
	# quit after one iteration for testing purposes
	# sys.exit()