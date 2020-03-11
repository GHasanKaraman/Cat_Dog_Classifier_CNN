import keras
from keras.models import load_model
import cv2

# 0 means 'cat' and 1 means 'dog'

#Loading the saved CNN model
model = load_model('weights.hdf5')

#Getting the image to predict it. Which image do you wanna predict, write its path instead of cat1.jpg
test_data = cv2.imread('cat1.jpg')
test_data = cv2.resize(test_data, (128, 128))

test_data = test_data.reshape(1, 128, 128, 3)

prediction = model.predict(test_data)

print(prediction)

if (prediction[0,0] > prediction[0,1]):
    print('It\'s a cat')
else:
    print('It\'s a dog')