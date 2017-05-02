
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD


import pickle
with open('example_wordVector.pkl', 'rb') as handle:
    wordVector = pickle.load(handle)


# set parameters:
filters = 250
kernel_size = 3
hidden_dims = 250

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
model.add(Activation('sigmoid'))


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# pad training and test sets
#x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# shape needed  (24000, 159, 1) 
model.train_on_batch(X_batch, Y_batch)

# predict class labels
#model.predict_on_batch(self, x)
