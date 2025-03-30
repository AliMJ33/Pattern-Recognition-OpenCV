import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Mario.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
template = cv2.imread('Mario_Coin.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
(w, h) = template.shape[: 2]

images = []
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for i, meth in enumerate(methods):
    img = image.copy()
    method = eval(meth)
    res = cv2.matchTemplate(img, template, method)

    # Get the positions of the product of matchTemplate:
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:   # Different methods from the others in methods list
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # After getting the two corners of the matched template, use a mask on those positions from the original image:
    roi = image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    mask = np.zeros(img.shape, dtype= "uint8")

    # Add the shadow effect (reducing brightness) on the image and then reassign the matched part of the image to be bright again and stand out as the recognized part.
    img = cv2.addWeighted(img, 0.25, mask, 0.75, 0)
    img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = roi
    cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 5)
    images.append(img)

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], "gray")
    plt.title(methods[i])
    plt.xticks([]), plt.yticks([])

plt.show()