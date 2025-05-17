import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../assets/imgs/plane.jpg")
print(type(img))
print(img.shape)
cv.imshow("plane", img)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite("../assets/imgs/plane_copy.jpg", img)
cv.imwrite("../assets/imgs/plane_copy.png", img)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("rgb_img", rgb_img)
plt.imshow(img)
cv.waitKey()
cv.destroyAllWindows()

print("imagb is saved")