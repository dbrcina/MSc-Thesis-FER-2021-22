import os
import string
from operator import itemgetter

import cv2
import pandas as pd
import torch
from PIL import Image

import config
from datasets import VAL_TRANSFORM_OCR
from train import ALPRLightningModule
from transform import four_point_transform

image_path = r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\baza_slika\040603\P1010003.jpg"
image_annot = os.path.splitext(image_path)[0] + ".csv"

df = pd.read_csv(image_annot, index_col=0)

x = df["x1"][0]
y = df["y1"][0]
w = df["x2"][0] - x
h = df["y2"][0] - y

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
lp = image[y:y + h, x:x + w]

ycrcb_image = cv2.cvtColor(lp, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(ycrcb_image)
clahe = cv2.createCLAHE(config.CLAHE_LP_CLIP_LIMIT, config.CLAHE_LP_GRID_SIZE)
clahe_y = clahe.apply(y)
clahe_img = cv2.merge((clahe_y, cr, cb))
lp_updated = cv2.cvtColor(clahe_img, cv2.COLOR_YCrCb2BGR)

gray = cv2.cvtColor(lp_updated, cv2.COLOR_BGR2GRAY)
bilateral = cv2.bilateralFilter(gray,
                                d=config.BILATERAL_D,
                                sigmaColor=config.BILATERAL_SIGMA_COLOR,
                                sigmaSpace=config.BILATERAL_SIGMA_SPACE)

thresh = cv2.threshold(bilateral, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]

c = None

for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
    approx = cv2.approxPolyDP(cnt, epsilon=cv2.arcLength(cnt, closed=True) * 0.02, closed=True)
    if len(approx) == 4:
        c = approx
        break

transformed = four_point_transform(thresh, c.reshape(4, 2))
morph = cv2.erode(transformed, (5, 5), iterations=1)
# cv2.imshow("test", morph)
# cv2.waitKey()
# exit()

cnts = cv2.findContours(morph, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
cts = []
for cnt in sorted(cnts, key=lambda c: cv2.boundingRect(c)[0]):
    x, y, w, h = cv2.boundingRect(cnt)
    if abs(cv2.contourArea(cnt)) < 100 or w > h or h < 10:
        continue
    char_img = morph[y:y + h, x:x + w]
    char_img = cv2.resize(char_img, dsize=config.OCR_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
    cts.append(char_img)
    cv2.imshow("TESTEST", char_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

x = torch.stack([VAL_TRANSFORM_OCR(Image.fromarray(s)) for s in cts])
model = ALPRLightningModule.load_from_checkpoint("pl_ocr/models/epoch=96-val_loss=0.11-val_acc=0.97.ckpt")
y = model.predict(x)
CHARACTERS = {i: c for i, c in enumerate(list(string.digits + string.ascii_uppercase))}
predicted_lp = "".join(itemgetter(*torch.argmax(y, dim=1).tolist())(CHARACTERS))
print(predicted_lp)
cv2.imshow("teste", morph)
cv2.waitKey()

# output = cv2.connectedComponentsWithStats(transformed, 4, cv2.CV_32S)
# (numLabels, labels, stats, centroids) = output
#
# mask = np.zeros(transformed.shape, dtype="uint8")
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
#     # component_mask = (labels == i).astype("uint8") * 255
#     # mask = cv2.bitwise_or(mask, component_mask)
#     cv2.imshow("testse", transformed[y:y+h, x:x+w])
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     # if all((keep_width, keep_height, keep_area)):
#     #     component_mask = (labels == i).astype("uint8") * 255
#     #     mask = cv2.bitwise_or(mask, component_mask)
#
# # cv2.imshow("lppp", transformed)
# # cv2.imshow("maskkk", mask)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
