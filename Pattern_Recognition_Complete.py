import cv2
import numpy as np

image = cv2.imread('Mario.png')
image_copy = image.copy()
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template = cv2.imread('Mario_Coin.png', 0)
w, h = template.shape[ : 2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF)
res_norm = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 2596000
threshold_norm = 0.8

loc = np.where(res >= threshold)
loc_norm = np.where(res_norm >= threshold_norm)

for pt in zip(*loc[: : -1]):
    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), 0, 1)

for pt in zip(*loc_norm[: : -1]):
    cv2.rectangle(image_copy, pt, (pt[0] + w, pt[1] + h), 1, 1)

cv2.putText(image, "TM_CCOEFF", (25, 50), 2, 0.67, (255, 0, 0))
cv2.putText(image_copy, "TM_CCOEFF_NORMED", (25, 50), 2, 0.67, (255, 0, 0))
cv2.imshow('Multiple object', np.hstack([image, image_copy]))
cv2.waitKey(0)
cv2.destroyAllWindows()