import cv2
import numpy as np
from matplotlib import pyplot as plt

img_name = input("What's the name of the picture? ")
img = cv2.imread(img_name)
tot_pixel = img.size
print("Total pixels: " + str(tot_pixel))
labels = ['RED', 'ORANGE', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE']
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
data = []

# boundaries for the colors: red, orange, yellow, green, blue, purple
boundaries = [([0, 0, 100], [10, 80, 255]),
              ([0, 100, 240], [22, 200, 255]),
              ([0, 160, 235], [35, 250, 255]),
              ([15, 120, 0], [48, 140, 10]),
              ([235, 67, 0], [255, 87, 10]),
              ([125, 0, 100], [155, 80, 255])]


def color(boundaries):
    i = 0
    for(lower, upper) in boundaries:
        # creates numpy array from boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # finds colors in boundaries applies a mask
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('' + str(labels[i]), output)

        color_pixel = np.count_nonzero(output)
        percentage = round(color_pixel * 100 / tot_pixel, 2)

        print(str(labels[i]) + " pixels: " + str(color_pixel))
        print("Percentage of " + str(labels[i]) + " pixels: " + str(percentage) + "%")
        data.append(percentage)
        i = i + 1


color(boundaries)
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=labels, colors=colors)
# show plot
plt.show()
cv2.waitKey(0)
