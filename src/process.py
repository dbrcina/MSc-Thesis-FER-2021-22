import os

import cv2
import numpy as np
import pandas as pd

image_path = r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\baza_slika\210503\P1010003.jpg"
image_annot = os.path.splitext(image_path)[0] + ".csv"

df = pd.read_csv(image_annot, index_col=0)

x = df["x1"][0]
y = df["y1"][0]
w = df["x2"][0] - x
h = df["y2"][0] - y

image = cv2.imread(image_path)
lp = image[y:y + h, x:x + w]

gray = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
hist = cv2.equalizeHist(blur)
thresh = cv2.adaptiveThreshold(hist, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)
otsu = cv2.threshold(hist, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imshow("t", thresh)
cv2.waitKey()
exit()
# output = cv2.connectedComponentsWithStats(otsu, 4, cv2.CV_32S)
# (numLabels, labels, stats, centroids) = output
#
# mask = np.zeros(otsu.shape, dtype=np.uint8)
#
# for i in range(1, numLabels):
#     x = stats[i, cv2.CC_STAT_LEFT]
#     y = stats[i, cv2.CC_STAT_TOP]
#     w = stats[i, cv2.CC_STAT_WIDTH]
#     h = stats[i, cv2.CC_STAT_HEIGHT]
#     area = stats[i, cv2.CC_STAT_AREA]
#     keep_width = 5 < w < 50
#     keep_height = 45 < h < 65
#     keep_area = 100 < area < 500
#
#     if all((keep_width, keep_height, keep_area)):
#         component_mask = (labels == i).astype(np.uint8) * 255
#         mask = cv2.bitwise_or(mask, component_mask)
#
# cv2.imshow("lp", lp)
# cv2.imshow("mask", mask)
# cv2.waitKey()
# cv2.destroyAllWindows()
