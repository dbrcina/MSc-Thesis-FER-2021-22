import os

import cv2
import numpy as np
import pandas as pd

import config


def display(title: str, image: np.ndarray) -> None:
    cv2.imshow(title, image)
    cv2.waitKey()


def apply_clahe(image: np.ndarray) -> np.ndarray:
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_image)
    clahe = cv2.createCLAHE(config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID_SIZE)
    clahe_y = clahe.apply(y)
    clahe_img = cv2.merge((clahe_y, cr, cb))
    return cv2.cvtColor(clahe_img, cv2.COLOR_YCrCb2BGR)


DEBUG = True
image_path = r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\baza_slika\141002\Pa140001.jpg"
image_annot = os.path.splitext(image_path)[0] + ".csv"

df = pd.read_csv(image_annot, index_col=0)
x = df["x1"][0]
y = df["y1"][0]
w = df["x2"][0] - x
h = df["y2"][0] - y

image = cv2.imread(image_path, cv2.IMREAD_COLOR)

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

# th = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 4)
th = cv2.threshold(bilateral, 127, 255, cv2.THRESH_BINARY_INV)[1]
display("ther", th)
# canny = cv2.Canny(bilateral, 120, 255)
# if DEBUG:
#     display("LP+Canny", canny)

# cnts = cv2.findContours(canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
# chars = []
# for cnt in sorted(cnts, key=lambda c: cv2.boundingRect(c)[0]):
#     x, y, w, h = cv2.boundingRect(cnt)
#     if h < 10:
#         continue
#     c = gray[y:y+h, x:x+w]
#     display("test", c)
#     cv2.destroyWindow("test")

# cv2.drawContours(image, chars, -1, (0, 0, 255))
# cv2.imshow("TESTE", image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# thresh = cv2.threshold(bilateral, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# if DEBUG:
#     display("LP+Thresh", thresh)

# cnts = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
# c = None
# for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
#     approx = cv2.approxPolyDP(cnt, epsilon=cv2.arcLength(cnt, closed=True) * 0.02, closed=True)
#     if len(approx) == 4:
#         c = approx
#         break
#
# transformed = four_point_transform(thresh, c.reshape(4, 2))
# if DEBUG:
#     display("LP+Perspective", transformed)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# if not DEBUG:
#     display("LP+Morph", morph)
#
# cnts = cv2.findContours(morph, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
# cts = []
# for cnt in sorted(cnts, key=lambda c: cv2.boundingRect(c)[0]):
#     x, y, w, h = cv2.boundingRect(cnt)
#     char_img = morph[y:y + h, x:x + w]
#     char_img = cv2.resize(char_img, dsize=config.OCR_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
#     cts.append(char_img)
#     if not DEBUG:
#         display("TESTEST", char_img)
#         cv2.destroyWindow("TESTEST")
#
# x = torch.stack([VAL_TRANSFORM_OCR(Image.fromarray(s)) for s in cts])
# model = ALPRLightningModule.load_from_checkpoint("pl_ocr/models/epoch=96-val_loss=0.11-val_acc=0.97.ckpt")
# y = model.predict(x)
# CHARACTERS = {i: c for i, c in enumerate(list(string.digits + string.ascii_uppercase))}
# predicted_lp = "".join(itemgetter(*torch.argmax(y, dim=1).tolist())(CHARACTERS))
# print(predicted_lp)
#
# cv2.waitKey()
# cv2.destroyAllWindows()

# output = cv2.connectedComponentsWithStats(canny, 4, cv2.CV_32S)
# (numLabels, labels, stats, centroids) = output
#
# mask = np.zeros(gray.shape, dtype="uint8")
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
#         component_mask = (labels == i).astype("uint8") * 255
#         mask = cv2.bitwise_or(mask, component_mask)
#
# display("MASK", mask)
# # cv2.imshow("lppp", transformed)
# # cv2.imshow("maskkk", mask)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
