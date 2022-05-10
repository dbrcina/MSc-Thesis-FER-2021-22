import random as rng

import cv2
import numpy as np
import pandas as pd

import config
import utils
from src.transform import four_point_transform


def display(title: str, image: np.ndarray) -> None:
    cv2.imshow(title, image)
    cv2.waitKey()


DEBUG = True
image_path = r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\data\baza_slika\170902\P9170046.jpg"
image_annot = utils.replace_file_extension(image_path, config.ANNOTATION_EXT)

df = pd.read_csv(image_annot, index_col=0)
x = df["x1"][0]
y = df["y1"][0]
w = df["x2"][0] - x
h = df["y2"][0] - y

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
lp = image[y:y + h, x:x + w]
display("Original", lp)

gray = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
bilateral = cv2.bilateralFilter(gray,
                                d=15,
                                sigmaColor=50,
                                sigmaSpace=5)
display("bilateral", bilateral)

edged = utils.auto_canny(bilateral)
display("Edged", edged)

exit()


contours = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
drawing = np.zeros((lp.shape[0], lp.shape[1], 3), dtype=np.uint8)
cv2.drawContours(drawing, sorted_contours, 0, (0, 0, 255))
display("tetet", drawing)
exit()
lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 50, minLineLength=w / 4.0, maxLineGap=h / 4.0)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(lp, (x1, y1), (x2, y2), (0, 0, 255))

display("testset", lp)
exit()

contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

c = None
for cnt in sorted_contours:
    approx = cv2.approxPolyDP(cnt, epsilon=cv2.arcLength(cnt, closed=True) * 0.1, closed=True)
    if len(approx) == 4:
        c = approx
        break

cv2.drawContours(lp, [c], 0, (0, 0, 255))
display("lplp", lp)
exit()
rect = cv2.minAreaRect(sorted_contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
transformed = four_point_transform(lp, box)
display("trans", transformed)
exit()

blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
display("Thresh", thresh)

contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
contours_bbs = list(map(cv2.boundingRect, contours))
chars = []
for contour, (x, y, w, h) in sorted(zip(contours, contours_bbs), key=lambda x: (x[1][1], x[1][0])):
    if abs(cv2.contourArea(contour)) < 100 or w > h or h < 10:
        continue
    chars.append(thresh[y:y + h, x:x + w])
    display("test", thresh[y:y + h, x:x + w])
    cv2.destroyAllWindows()

exit()
contours, _ = cv2.findContours(thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
sorted_cnt = sorted(contours, key=cv2.contourArea, reverse=True)

drawing = np.zeros((lp.shape[0], lp.shape[1], 3), dtype=np.uint8)
hull = cv2.convexHull(sorted_cnt[0])
approx = cv2.approxPolyDP(hull, epsilon=cv2.arcLength(hull, closed=True) * 0.02, closed=True)
cv2.drawContours(drawing, [hull], -1, (0, 0, 255))

display("c", drawing)
exit()

hull_list = []
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    hull_list.append(hull)

drawing = np.zeros((lp.shape[0], lp.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.drawContours(drawing, hull_list, i, color)

display("c", drawing)
exit()
cnts = cv2.findContours(dilate, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
c = None
for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
    approx = cv2.approxPolyDP(cnt, epsilon=cv2.arcLength(cnt, closed=True) * 0.02, closed=True)
    if len(approx) == 4:
        c = approx
        break

if c is not None:
    cv2.drawContours(lp, [c], -1, (0, 0, 255))
    display("test", lp)
exit()

if DEBUG:
    display("Original", image)

clahe_image = apply_clahe(image)
if DEBUG:
    display("Original+CLAHE", clahe_image)

lp = clahe_image[y:y + h, x:x + w]
if DEBUG:
    display("OriginalLP", lp)

gray = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
if DEBUG:
    display("LP+Gray", gray)

bilateral = cv2.bilateralFilter(gray,
                                d=config.BILATERAL_D,
                                sigmaColor=config.BILATERAL_SIGMA_COLOR,
                                sigmaSpace=config.BILATERAL_SIGMA_SPACE)
if DEBUG:
    display("LP+Bilateral", bilateral)

th = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
display("LP+Thresh", th)

cnts = cv2.findContours(th, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
c = None
for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
    approx = cv2.approxPolyDP(cnt, epsilon=cv2.arcLength(cnt, closed=True) * 0.02, closed=True)
    if len(approx) == 4:
        c = approx
        break

if c is None:
    exit()

x, y, w, h = cv2.boundingRect(c)
if w >= 50 and h >= 20:
    transformed = four_point_transform(th, c.reshape(4, 2))
    display("LP+Perspective", transformed)
