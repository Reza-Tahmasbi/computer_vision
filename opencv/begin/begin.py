import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../assets/imgs/plane.jpg")
print(type(img))
print(img.shape)
cv.imshow("plane", img)
cv.waitKey()
cv.destroyAllWindows()
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imwrite("../assets/imgs/plane_copy.jpg", img)
print("imagb is saved")