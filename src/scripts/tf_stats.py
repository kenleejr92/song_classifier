from sklearn.feature_extraction.text import TfidfVectorizer
import os, re, glob, sys
from tqdm import tqdm
import numpy as np

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util as access_data

from collections import Counter
import itertools



# read in stopwords 
stoplist = set(line.strip() for line in open('/home/ubuntu/repo/stopwords.txt'))   

(train_X, train_Y, train_le, test_X, test_Y, test_le) = access_data.get_data_sets_w_lyrics()

print list(test_le.classes_)
sys.exit()
genre_idx = train_Y.iloc[:][0] == int(sys.argv[1])


lyrics = [0]*sum(genre_idx)
count  = 0; words = ()

for idNum, songID in train_X['songID'].iteritems():
	#print songID
	#print genre_idx[idNum];

	if genre_idx[idNum]:
		#  print songID
		with open("/home/ubuntu/lyrics/"+songID+".lyrics") as fidx:
			sentences = [word.lower() for line in fidx for word in line.split()]

		# remove stop words, lowercase terms, remove periods and parentheses
		texts = [re.sub(r'[-()?"\'"]', '', word).rstrip('[.,]')  for word in sentences if word not in stoplist]
		
		words = np.append(words, texts[0:201])
		#lyrics[count] = str(' '.join(texts[0:201]))
		count += 1

	#if count > 100:
		#print lyrics[0:4]
		#sys.exit()
	#	break


		#np.save('/home/ubuntu/wordVectors/'+songID+'.npy', lyrics, allow_pickle=False)

'''
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3))

tfidf_matrix =  tf.fit_transform(lyrics)
feature_names = tf.get_feature_names() 
#print feature_names

dense = tfidf_matrix.todense()
song = dense[0].tolist()[0]
phrase_scores = [pair for pair in zip(range(0, len(song)), song) if pair[1] > 0]
sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
   print('{0: <20} {1}'.format(phrase, score))
'''

counts = Counter(words)
print(counts.most_common(31))
print len(words)