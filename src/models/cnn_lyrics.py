'''
Developing a CNN for the wordvectors using a batch training mode
Person to Blame: A. Gee
'''

# import packages
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import LSTM
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing import sequence
import sys, os, pickle
import numpy as np
from keras.utils.np_utils import to_categorical
from tqdm import tqdm

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util as access_data


# define functions
def get_next_batch(songID):
	# testing with one wordvector for now
	try:
		wordVector = np.load('/home/ubuntu/wordVectors/'+songID+'.npy')
		return wordVector
	except:
		return None

# get data
(train_X, train_Y, train_le, test_X, test_Y, test_le) = access_data.get_data_sets_w_lyrics()

size_training = train_X.shape[0]
size_test = test_X.shape[0]


'''
A = train_le.inverse_transform(train_Y)
for ii in range(A.shape[0]):
	print A[ii]
sys.exit()
'''

# set parameters:
filters = 25
kernel_size = 3
hidden_dims = 25
maxlen = 30
num_of_epochs = 3

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model = Sequential()
'''
model.add(Conv1D(filters,
                 kernel_size,
                 strides = 3,
                 activation='relu',
                 input_shape=(30, 300)))
model.add(MaxPooling1D(pool_size=2))
'''
model.add(LSTM(10, input_length=30, input_dim=300))
# We add a vanilla hidden layer:
#model.add(Dense(hidden_dims, activation = 'relu'))
#model.add(Dropout(0.2))
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(4, activation = 'sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model.summary())

for epoch in tqdm(range(num_of_epochs)):

	for ii in tqdm(range(size_training)):
		# get next song to train
		X_batch = get_next_batch(train_X['songID'].iloc[ii])

		if X_batch is None:
			continue

		Y_batch = int(train_Y.iloc[ii][0])

		# encode target value
		encodedOutput = np.zeros((1,4))
		np.put(encodedOutput, [Y_batch], [1])

		#print X_batch.shape

		# pad training and test sets
		X_batch = np.vstack((X_batch, np.zeros((maxlen-X_batch.shape[0],300))))
		X_batch = np.expand_dims(X_batch, axis=0)

		#print "After zero padding size"
		#print X_batch.shape

		# train model
		hist = model.train_on_batch(X_batch, encodedOutput)

print "Training loss of LSTM"
print model.metrics_names
print(hist)

num_scores = 0
sum_score = 0
sumLoss = 0
print "Running Test Data"
for ii in tqdm(range(size_test)):
	X_batch = get_next_batch(test_X['songID'].iloc[ii])

	if X_batch is None:
			continue

	Y_batch = int(test_Y.iloc[ii][0])
	# encode target value
	encodedOutput = np.zeros((1,4))
	np.put(encodedOutput, [Y_batch], [1])
	
	# pad training and test sets
	X_batch = np.vstack((X_batch, np.zeros((maxlen-X_batch.shape[0],300))))
	X_batch = np.expand_dims(X_batch, axis=0)

	# predict class labels
	scores = model.evaluate(X_batch, encodedOutput, verbose=0)
	sumLoss = scores[0] + sumLoss
	sum_score = scores[1] + sum_score
	num_scores += 1

print "Size of Training Set Size: %f" % size_training
print "Size of Test Set Size: %f" % size_test

print sum_score
print num_scores
print("\n Avg Accu: %.2f%%" % (sum_score*100/num_scores))

print("\n Avg Loss: %.2f" % (sumLoss/num_scores))


