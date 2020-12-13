import cv2
import numpy as np
import time

# preparation for writing output video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)
time.sleep(3)
background=0

for i in range(30):
    ret,background = cap.read()

background = np.flip(background,axis=1)

while(cap.isOpened()):
    ret, img = cap.read()

    # # Flip the image 
    img = np.flip(img, axis = 1)

    # Convert the image to HSV color space.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (35, 35), 0)

    # Defining lower range for red color detection.
    lower = np.array([0,120,70])
    upper = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower, upper)

    # # Defining upper range for red color detection
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # # Addition of the two masks to generate the final mask.
    mask1 = mask1 + mask2
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    
    out.write(final_output)
    
    cv2.imshow("mask1", final_output)
    k = cv2.waitKey(10)
    if k == 27:
        break
