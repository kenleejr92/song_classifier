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
import tensorflow as tf

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util as access_data

# define functions
def get_next_training_batch(songID):
	# testing with one wordvector for now
	wordVector = np.load('/home/ubuntu/wordVectors/'+songID+'.npy')
	return wordVector

# get data
(train_X, train_Y, train_le, test_X, test_Y, test_le) = access_data.get_data_sets_w_lyrics()


#size_training = train_X.shape[0]
size_training = 1000

# set parameters:
filters = 25
kernel_size = 3
hidden_dims = 25
maxlen = 25
num_of_epochs = 5

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model = Sequential()
model.add(Conv1D(filters,
                 kernel_size,
                 activation='relu',
                 input_shape=(25, 300)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
# We add a vanilla hidden layer:
#model.add(Dense(hidden_dims, activation = 'relu'))
#model.add(Dropout(0.2))
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation = 'sigmoid'))


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model.summary())

for epoch in range(num_of_epochs):

	for ii in range(size_training):
		# get next song to train
		X_batch = get_next_training_batch(train_X['songID'].iloc[ii])
		Y_batch = np.array([int(train_Y.iloc[ii][0])], ndmin=1)
		print Y_batch.shape

		# pad training and test sets
		X_batch = np.vstack((X_batch, np.zeros((maxlen-X_batch.shape[0],300))))
		#X_batch = np.reshape(X_batch, (X_batch.shape[0], X_batch.shape[1],1))
		X_shape = tf.TensorShape([None]).concatenate(X_batch.shape)
		X_batch = tf.placeholder_with_default(X_batch, shape=X_shape)

		print "After zero padding size"
		print X_batch.shape

		# train model
		model.train_on_batch(X_batch, Y_batch)
		sys.exit()


# predict class labels
predictions = model.predict_classes(x_test)
accuracy = (sum(predictions == y_test)*100/num_test_data)
print accuracy

