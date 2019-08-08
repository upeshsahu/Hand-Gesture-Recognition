#https://www.youtube.com/watch?v=O62YO0zXioM
import numpy as np
import cv2
import math
import os
try:
    if not os.path.exists('data_image'):
        os.makedirs('data_image')
except OSError:
    print('Erroe:creating dirsctory of data')        
file1=open('data_image/five.txt','a')
# Open Camera
f=True
capture = cv2.VideoCapture(0)
count=0
while capture.isOpened():

    ret, frame = capture.read()
    cv2.rectangle(frame, (200, 250), (400, 450), (0, 255, 0), 0)  
    crop_image = frame[250:450, 200:400]

    cv2.imshow("Gesture", frame)
    name='./data_image/five.txt'+str(count)+".jpeg";
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
    #all_image = np.hstack((drawing, crop_image))
    

    
    small_image = cv2.resize(thresh, (0,0), fx=0.5, fy=0.5)
    

    cv2.imshow("Thresholded", thresh)
    cv2.imshow("small image",small_image)
    
    #cv2.imshow('Contours', all_image)
    
    #print(np.array(small_image,dtype="float")/255.0)
    #file1.write(np.array(small_image,dtype="float")/255.0)
    
    
    

    count=count+1

capture.release()
cv2.destroyAllWindows()