# -*- coding: utf-8 -*-
# import modules & set up logging
import gensim, logging, os, re, glob, sys
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

# Trim artist and song
def trim_lyrics(lyrics):
	lyrics = re.sub('[^0-9a-zA-Z\w\s]+', '', lyrics)
	return (lyrics)

class read_from_file(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        for line in open(self.fname):
            yield line


allWords = []

files = glob.glob('/Users/frank/Documents/Classwork/DataMining/project/lyrics/*')
#print files

# read in stopwords
stoplist = set(line.strip() for line in open('./stopwords.txt'))   

wvModel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

for f in files:

	with open(f) as fidx:
		sentences = [word.lower() for line in fidx for word in line.split()]
	print sentences

	# remove stop words, lowercase terms, remove periods and parentheses
	texts = [re.sub(r'[()]', '', word).rstrip('[.,]') for word in sentences if word not in stoplist]
	
	for lyric in texts:                            
		if lyric in wvModel.vocab:
			wordvector = wvModel.wv(lyric)
	
	print wordvector
	sys.exit()
	