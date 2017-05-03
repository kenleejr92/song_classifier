'''
Developing a CNN for the wordvectors using a batch training mode
Person to Blame: A. Gee
'''

# import packages
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing import sequence
import sys, os, pickle
import numpy as np

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util as access_data

# define functions
def get_next_training_batch(songID):
	# testing with one wordvector for now
	with open('example_wordVector.pkl', 'rb') as handle:
		wordVector = pickle.load(handle)
	return (wordVector, 1)

# get data
(train_X, train_Y, train_le, test_X, test_Y, test_le) = access_data.get_data_sets_w_lyrics()
print train_X
sys.exit()
size_training = train_X.shape[0]
print size_training

# set parameters:
filters = 250
kernel_size = 3
hidden_dims = 250
maxlen = 250
num_of_epochs = 5

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model = Sequential()
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 input_shape=(maxlen, 300)))
# we use max pooling:
model.add(MaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims, activation = 'relu'))
model.add(Dropout(0.2))
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation = 'softmax'))


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


for epoch in range(num_of_epochs):

	for ii in range(size_training):
		# get next song to train
		(X_batch, Y_batch) = get_next_training_batch("fakenews")
		
		print X_batch.shape

		# pad training and test sets
		X_batch = np.vstack((X_batch, np.zeros((maxlen-X_batch.shape[0],300))))

		print "After zero padding size"
		print X_batch.shape

		sys.exit()
		# train model
		model.train_on_batch(X_batch, Y_batch)


# predict class labels
predictions = model.predict_classes(x_test)
accuracy = (sum(predictions == y_test)*100/num_test_data)
print accuracy

