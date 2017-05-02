
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import data_accessor_util

import pickle
with open('example_wordVector.pkl', 'rb') as handle:
    wordVector = pickle.load(handle)

def get_next_training_batch(songID)


# get data
(train_X, train_Y, train_le, test_X, test_Y, test_le) = get_data_sets_w_lyrics()
size_training = train_X.shape[0]

# set parameters:
filters = 250
kernel_size = 3
hidden_dims = 250
maxlen = 50
num_of_epochs = 5

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model = Sequential()
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('softmax'))


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


for epoch in range(num_of_epochs):

	for ii in range(size_training):
		# get next song to train
		(X_batch, Y_batch) = get_next_training_batch(songID)
	
		# pad training and test sets
		X_batch = sequence.pad_sequences(X_batch, maxlen=maxlen)
		Y_batch = sequence.pad_sequences(Y_batch, maxlen=maxlen)

		# train model
		model.train_on_batch(X_batch, Y_batch)


# predict class labels
predictions = model.predict_classes(x_test)
accuracy = (sum(predictions == y_test)*100/num_test_data)
print accuracy

