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

f=[]
capture = cv2.VideoCapture(0)
count=0
while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    # Get hand data from the rectangle sub window
    #                      x1   y1     x2   y2    r   g   b   width
    cv2.rectangle(frame, (200, 250), (400, 450), (0, 255, 0), 0)
    #croping thAT PArticular frame from the big frame
    crop_image = frame[250:450, 200:400]
    #                   y1 y2    x1  x2
    crop_image = cv2.resize(crop_image, (0,0), fx=0.5, fy=0.5)
    name='./data_image/swing/swing-'+str(count)+".jpeg";
    #print(len(crop_image[0])) 
    # Apply Gaussian blur here 3X3 is the kernel used 
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    #within the np.array(we are providing the lower and upper value of the skin colour)
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    #dilation add of pixel and erosion is remove of pixel
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    
    #print('save image')    
    #file1.write(np.array(mask2))c
    #print(len(mask2.flattern()))
    #Show threshold image
#    if f:
#        for i in range(200):
#            for j in range(200):
#                print(str(thresh[i][j])),
#            print()
#        f=False
#        print()        

    #file1.write(np.array(thresh).reshape((1,40000)))
    
    #file1.write("\nfrom here the new array start\n")
    #print("writing new")
    cv2.imshow("Thresholded", thresh)
    
    #cv2.imshow("small image",small_image)
    cv2.imwrite(name,thresh)
    #print(np.array(thresh,dtype="uint8")/255.0)
    print(type(thresh))
    print(thresh.shape)
    #f.append(thresh)
    # if not count==1:
    #     for i in range(thresh.shape[0]):
    #         for j in range(thresh.shape[1]):
    #             print(str(thresh[i][j]/255)+"  "),
    #         print()    
    #print("new array")

    
    #file1.write(np.array(small_image,dtype="float")/255.0)# Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        hull = cv2.convexHull(contour)

        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
        # tips) for all defects
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # Print number of fingers
        if count_defects == 0:
            cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        elif count_defects == 1:
            cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 2:
            cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 3:
            cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        elif count_defects == 4:
            cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        else:
            pass
    except:
        pass

    # Show required images
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
    count=count+1
file1.write(np.array(f,dtype="uint8"))    
capture.release()
cv2.destroyAllWindows()