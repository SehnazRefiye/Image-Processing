import cv2
import numpy as np


def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        h = hsv[y, x, 0]
        s = hsv[y, x, 1]
        v = hsv[y, x, 2]
        a = []
        a.append(h)
        a.append(s)
        a.append(v)
        # print(a)
        pixel = 'HSV: ' + str(h) + ' ' + str(s) + ' ' + str(v)
        cv2.putText(img, pixel, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (100, 20, 0), 1)
        cv2.imshow("Image", img)
        print(a)


img_name = input("What's the name of the picture? ")
img = cv2.imread(img_name)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Image", img)
cv2.setMouseCallback('Image', pick_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
