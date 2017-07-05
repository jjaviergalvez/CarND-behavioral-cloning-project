from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pickle
import json
from sklearn.utils import shuffle
import time

#import my coustum function
from get_data import get_data_preprocessed

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


# Function to print the time in HH:MM:SS
def print_time(myTime):
	# credits to http://stackoverflow.com/questions/775049/python-time-seconds-to-hms
	m, s = divmod(myTime, 60)
	h, m = divmod(m, 60)
	print("finished in %d:%02d:%02d" % (h, m, s))

print('reading and preprocessing the data...')
X_train, y_train, X_cv, y_cv = get_data_preprocessed()

# Shuffle the data
X_train, y_train = shuffle(X_train, y_train)

# Model definition
model = Sequential()

model.add(Convolution2D(3, 3, 3, input_shape=(40, 160, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(6, 4, 4))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(12, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(6, 3, 3))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(48))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('tanh'))

# Compile and show the architecure 
model.compile(optimizer='Adam', loss='mse')
model.summary()

# Save the model
print('saving the model...')
json_string = model.to_json()
with open('model.json', 'w') as f:
     json.dump(json_string, f)

# Define checkpoint callback
filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# Defining early stopping callback 
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='min')

callbacks_list = [checkpoint, early_stopping]

# Train the model 
print('training...')
start_time = time.clock()
#history = model.fit(X_train, y_train, batch_size=128, nb_epoch=50, validation_data=(X_cv, y_cv), callbacks=callbacks_list, verbose=2)
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=25, validation_split=0.2, callbacks=callbacks_list, verbose=2)
print_time(time.clock() - start_time)

#save the history to *.p file
print('saving history...')
pickle.dump(history.history, open( "history_train.p", "wb" ))

print()
print('f i n i s h e d!')
print()