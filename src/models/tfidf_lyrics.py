from sklearn.feature_extraction.text import TfidfVectorizer


# read in stopwords 
stoplist = set(line.strip() for line in open('/home/ubuntu/repo/stopwords.txt'))   

(train_X, train_Y, train_le, test_X, test_Y, test_le) = access_data.get_data_sets_w_lyrics()


for idNum in tqdm(np.arange(int(sys.argv[1]),int(sys.argv[2]))):
	# check to see if lyrics exist

	songID = train_X['songID'][idNum]
	genre_idx = train_X['genre'][idNum]

	lyrics = ()



	with open("/home/ubuntu/lyrics/"+songID+".lyrics") as fidx:
		sentences = [word.lower() for line in fidx for word in line.split()]

	# remove stop words, lowercase terms, remove periods and parentheses
	texts = [re.sub(r'[()]', '', word).rstrip('[.,]')  for word in sentences if word not in stoplist]

	print texts

	lyrics = lyrics.extend(texts[0:51])
	print lyrics
	sys.exit()

	#np.save('/home/ubuntu/wordVectors/'+songID+'.npy', lyrics, allow_pickle=False)

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words = 'english')

tfidf_matrix =  tf.fit_transform(lyrics)
feature_names = tf.get_feature_names() 