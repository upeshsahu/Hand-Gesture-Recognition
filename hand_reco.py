import numpy as np
import cv2
import math
import matplotlib.pyplot as plt  
#%matplotlib inline
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
#differnt layers
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split


traindata=[]


for i in range(700):
    img=cv2.imread('data_image/data/okay/okay-'+str(i)+'.jpeg',0)
    traindata.append(img)



for i in range(700):
    img=cv2.imread('data_image/data/peace/peace-'+str(i)+'.jpeg',0)
    traindata.append(img)

for i in range(700):
    img=cv2.imread('data_image/data/ilu/ilu-'+str(i)+'.jpeg',0)
    traindata.append(img)



for i in range(700):
    img=cv2.imread('data_image/data/thumbsup/thumbsup-'+str(i)+'.jpeg',0)
    traindata.append(img)


for i in range(700):
    img=cv2.imread('data_image/data/hi/hi-'+str(i)+'.jpeg',0)
    traindata.append(img)

for i in range(700):
    img=cv2.imread('data_image/data/one/one-'+str(i)+'.jpeg',0)
    traindata.append(img)

for i in range(700):
    img=cv2.imread('data_image/data/swing/swing-'+str(i)+'.jpeg',0)
    traindata.append(img)



y_label=np.ones((4900,),dtype=int)
y_label[0:700]=0
y_label[700:1400]=1
y_label[1400:2100]=2
y_label[2100:2800]=3
y_label[2800:3500]=4
y_label[3500:4200]=5
y_label[4200:4900]=6
x_label=np.array(traindata).reshape(4900,100,100,1).astype("uint8")
y_label=np_utils.to_categorical(y_label,7)
x_label=x_label.astype("float32")
x_label/=255  
x_label,y_label=shuffle(x_label,y_label,random_state=2)
x_train,x_test,y_train,y_test=train_test_split(x_label,y_label,test_size=0.2,random_state=4)

model=Sequential()
model.add(Conv2D(64,(5,5),input_shape=(100,100,1),activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(32,(3,3),input_shape=(100,100,1),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(10,(2,2),activation="relu"))
#model.add(Dropout(0,3))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(7,activation="softmax"))
model.compile(optimizer="adam",metrics=["accuracy"],loss="categorical_crossentropy")

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=3,batch_size=50,verbose=2)

accuracy_scores=model.evaluate(x_test,y_test,verbose=0)

print("Error:%.2f%%" % (100-accuracy_scores[1]*100))
model.summary()


print("done")

capture = cv2.VideoCapture(0)
count=0
while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()
    cv2.rectangle(frame, (200, 250), (400, 450), (0, 255, 0), 0)
    crop_image = frame[250:450, 200:400]
    crop_image = cv2.resize(crop_image, (0,0), fx=0.5, fy=0.5)
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    cv2.imshow("Thresholded", thresh)
    
    thresh=np.array(thresh);
    x_label=thresh.reshape(1,100,100,1).astype("float32")/255
	
    y_label=model.predict(x_label)
    cv2.putText(frame,str(y_label), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)  

    cv2.imshow("Gesture", frame)
    if cv2.waitKey(1) == ord('q'):
        break
   
capture.release()
cv2.destroyAllWindows()
