import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Dropout

import numpy as np

# 0 means 'cat' and 1 means 'dog'

data = np.load('cat_and_dog.npy', allow_pickle = True)

(x_train, y_train), (x_test, y_test) = data

batch_size = 2
num_classes = 2
epochs = 8

#One Hot Encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Creating Convolutional Neural Network
model = Sequential()
model.add(Conv2D(256, kernel_size = (3, 3), input_shape = (128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(256, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(96, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

#If training param is true, the model will be trained
training = False

if training:
    model.fit(x_train, y_train, batch_size = batch_size,
              epochs = epochs,verbose = 1,
              validation_data = (x_test, y_test))
    model.save('weights.hdf5')
