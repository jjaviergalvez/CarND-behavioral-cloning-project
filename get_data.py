import pickle
import numpy as np
from preprocessing import preprocessing_data

#This function return the data asigned to train and validation set

def get_data_preprocessed():

	with open('data/train/00/data.p', mode='rb') as f:
	    data = pickle.load(f)
	X_train_00, y_train_00 = data['features'], data['labels']
	X_train_00 = preprocessing_data(X_train_00)

	with open('data/train/01/data.p', mode='rb') as f:
	    data = pickle.load(f)
	X_train_01, y_train_01 = data['features'], data['labels']
	X_train_01 = preprocessing_data(X_train_01)

	with open('data/train/02/data.p', mode='rb') as f:
	    data = pickle.load(f)
	X_train_02, y_train_02 = data['features'], data['labels']
	X_train_02 = preprocessing_data(X_train_02)

	with open('data/train/03/data.p', mode='rb') as f:
	    data = pickle.load(f)
	X_train_03, y_train_03 = data['features'], data['labels']
	X_train_03 = preprocessing_data(X_train_03)
	del data

	with open('data/train/04/data.p', mode='rb') as f:
	    data = pickle.load(f)
	X_train_04, y_train_04 = data['features'], data['labels']
	X_train_04 = preprocessing_data(X_train_04)
	del data

	X_train = np.concatenate( (X_train_00, X_train_01, X_train_02, X_train_03, X_train_04) )
	y_train = np.concatenate( (y_train_00, y_train_01, y_train_02, y_train_03, y_train_04) )



	with open('data/cv/00/data.p', mode='rb') as f:
	    data = pickle.load(f)
	X_cv_00, y_cv_00 = data['features'], data['labels']
	X_cv_00 = preprocessing_data(X_cv_00)

	X_cv = X_cv_00
	y_cv = y_cv_00



	return X_train, y_train, X_cv, y_cv