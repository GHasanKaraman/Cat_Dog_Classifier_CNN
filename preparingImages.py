import numpy as np
import cv2

def getImage(path):
    return cv2.imread(path)

x = np.array([])

#Getting all images then reshaping them to 128x128x3 (3 means RGB)
for i in range(60):
    path = 'datas/{}.jpg'.format(i+1)
    
    image = getImage(path)
    
    image = cv2.resize(image, (128, 128))
    
    x = np.append(x, image)
    
x = x.reshape((-1, 128, 128, 3))

#Seperating datas to train and test datas
x_train = np.append(x[:20,...], x[30:50,...]).reshape((-1,128,128,3))
x_test = np.append(x[20:30,...], x[50:,...]).reshape((-1,128,128,3))

y_train = np.array([])
y_test = np.array([])

#Labelling datas 
for i in range(x_train.shape[0]):
    if i < 20:
        y_train = np.append(y_train, 0)
    else:
        y_train = np.append(y_train, 1)
        
for i in range(x_test.shape[0]):
    if i < 10:
        y_test = np.append(y_test, 0)
    else:
        y_test = np.append(y_test, 1)
        
       
y_train = y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))
        
data = (x_train, y_train), (x_test, y_test)

#Save the labeled datas as npy file
np.save('cat_and_dog', data)
