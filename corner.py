# -*- coding: utf-8 -*-
"""
Created on Wed Dec 7, 2022

@author: erdem
"""

import cv2
import numpy as np

# Read the image
img = cv2.imread("text.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to a float32 format
gray = np.float32(gray)

# Apply the Shi-Tomasi corner detection method
corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)

# Convert the corner coordinates to integer values
corners = np.intp(corners)

# Draw circles around the detected corners
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

# Display the image with detected corners
cv2.imshow("Corners", img)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
