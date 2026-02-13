import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read image
image = cv2.imread('Coins.jpg')

#Convert into grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Show image
#plt.imshow(gray, cmap='gray')
#plt.show()

#Make the image blur to avoid noises, so it can detect the edges of the coins
blur = cv2.GaussianBlur(gray, (11, 11), 0)
#plt.imshow(blur, cmap='gray')
#plt.show()

#Detect edges using canny
canny = cv2.Canny(blur, 30, 150, 3) #30 and 150 are threshold values that determine the edge of the image.
#plt.imshow(canny, cmap='gray')
#plt.show()

#Making the edges thicker and visible
dilated = cv2.dilate(canny, (1,1), iterations=0)
#plt.imshow(dilated, cmap='gray')
#plt.show()

#Calculate the contour in the image and convert the image into RGB from BGR and then draw the contours
(cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

#Show it
plt.imshow(rgb)
plt.show()

#Result:
print("Coins in the image: ", len(cnt))