import numpy as np

def load_all():	
	X_train = np.load('X_train.npy')
	X_test = np.load('X_test.npy')
	y_train = np.load('y_train.npy')
	y_test = np.load('y_test.npy')
<<<<<<< HEAD
	classes = np.load('classes.npy')
	return X_train, X_test, y_train, y_test, classes
=======
	return X_train, X_test, y_train, y_test
>>>>>>> 9f011a1a6d34a3bc09335bf66cc2082c30795d38
