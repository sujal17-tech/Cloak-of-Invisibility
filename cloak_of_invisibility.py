import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
time.sleep(3)

background = 0
for i in range(150):
    ret, background = cap.read()

background = np.flip(background, axis=1)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = np.flip(img, axis=1)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for blue
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for detecting blue color
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean the mask using morphological operations
    mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations= 10) 
    mask = cv2.morphologyEx(blue_mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations= 10)


    # Invert the mask to segment out non-blue areas
    mask_inverse = cv2.bitwise_not(mask)

    # Replace the blue regions with the background
    res1 = cv2.bitwise_and(background, background, mask=mask)

    # Retain the non-blue regions from the current frame
    res2 = cv2.bitwise_and(img, img, mask=mask_inverse)

    # Combine the two results 
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow('Blue Cloak Effect', final_output)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
