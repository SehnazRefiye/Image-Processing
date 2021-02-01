import numpy as np
import cv2
import pyautogui
import time

time.sleep(1)
img = pyautogui.screenshot('my_screenshot.jpg')

path = r'D:\GIT Projects\my_screenshot.jpg'
RGB = cv2.imread(path)
# cv2.imshow('image', RGB)
"""
# resize image
output = cv2.resize(RGB, (1840, 1000))

cv2.imshow('new', output)
"""
# load the image, clone it for output, and then convert it to grayscale
# RGB = cv2.imread('my_screenshot.jpg')
RGB_copy = RGB.copy()
I_gray = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)

# gray_blurred = cv2.medianBlur(I_gray, 5)
# detect_circles = cv2.HoughCircles(I_gray, cv2.HOUGH_GRADIENT, 1.1, 100, param1=50, param2=30, minRadius=0, maxRadius=0)

# detect circles in the image
detect_circles = cv2.HoughCircles(I_gray, cv2.HOUGH_GRADIENT, 1.1, 100)

# ensure at least some circles were found
if detect_circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(detect_circles[0, :]).astype("int")

    for (x, y, r) in circles:
        # draw the circle in the output image
        cv2.circle(RGB_copy, (x, y), r, (0, 255, 0), 5)
        # cv2.putText(RGB, str(r), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        pyautogui.moveTo(x, y, duration=1)

# show the output image
cv2.imshow("circle", RGB_copy)
cv2.waitKey(0)
