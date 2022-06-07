# contrast enhancement examples

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import config

FOLDER = "../ce_examples/1"
os.makedirs(FOLDER, exist_ok=True)
SAVE = False


def calc_hist(image: np.ndarray) -> np.ndarray:
    return cv2.calcHist([image], [0], None, [256], [0, 256])


def plot_hist(hist: np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax.plot(hist)


def convert_image(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


image = cv2.imread(r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\ce_examples\2\original.jpg")
cv2.imshow("Original image", image)
if SAVE:
    cv2.imwrite(f"{FOLDER}/original.jpg", image)

ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
cv2.imshow("YCrCb image", ycrcb_image)
if SAVE:
    cv2.imwrite(f"{FOLDER}/ycrcb.jpg", ycrcb_image)

y, cr, cb = cv2.split(ycrcb_image)

hist_original_y = calc_hist(y)
plot_hist(hist_original_y)
if SAVE:
    plt.savefig(f"{FOLDER}/hist_original.jpg")

y_eq = cv2.equalizeHist(y)
eq_image = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)
cv2.imshow("Enhanced image - EQ", eq_image)
if SAVE:
    cv2.imwrite(f"{FOLDER}/eq.jpg", eq_image)

hist_eq_y = calc_hist(y_eq)
plot_hist(hist_eq_y)
if SAVE:
    plt.savefig(f"{FOLDER}/hist_eq.jpg")

y_clahe = cv2.createCLAHE(config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID_SIZE).apply(y)
clahe_image = cv2.cvtColor(cv2.merge([y_clahe, cr, cb]), cv2.COLOR_YCrCb2BGR)
cv2.imshow("Enhanced image - CLAHE", clahe_image)
if SAVE:
    cv2.imwrite(f"{FOLDER}/clahe.jpg", clahe_image)

hist_clahe_y = calc_hist(y_clahe)
plot_hist(hist_clahe_y)
if SAVE:
    plt.savefig(f"{FOLDER}/hist_clahe.jpg")

y_gauss = cv2.GaussianBlur(y_clahe, config.GAUSSIAN_BLUR_KSIZE, config.GAUSSIAN_BLUR_SIGMAX)
gauss_image = cv2.cvtColor(cv2.merge([y_gauss, cr, cb]), cv2.COLOR_YCrCb2BGR)
cv2.imshow("Enhanced image - CLAHE - Gauss", gauss_image)
if SAVE:
    cv2.imwrite(f"{FOLDER}/gauss.jpg", gauss_image)

plt.show()
cv2.destroyAllWindows()
