import argparse
import string
from argparse import Namespace
from operator import itemgetter
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from numpy import ndarray

from data import utils
from models import LeNet5

CHARACTERS = {i: c for i, c in enumerate(list(string.digits + string.ascii_uppercase))}


def _unit_vector(v: ndarray) -> ndarray:
    return v / np.linalg.norm(v)


def _cos_angle(v1: ndarray, v2: ndarray) -> ndarray:
    v1_u = _unit_vector(v1)
    v2_u = _unit_vector(v2)
    return np.dot(v1_u.flatten(), v2_u.flatten())


def _locate_lp(img: ndarray, contours: ndarray) -> Optional[Tuple[ndarray, ndarray]]:
    # https://docs.opencv.org/4.x/db/d00/samples_2cpp_2squares_8cpp-example.html#a23
    lp_cnt = None
    lp_img = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        approx = cv2.approxPolyDP(cnt, epsilon=cv2.arcLength(cnt, closed=True) * 0.02, closed=True)
        if len(approx) == 4:
            max_cosine = 0
            for i in range(4):
                v1 = approx[i - 1] - approx[i]
                v2 = approx[(i + 1) % 4] - approx[i]
                max_cosine = max(max_cosine, abs(_cos_angle(v1, v2)))
            if max_cosine < 0.3:
                x, y, w, h = cv2.boundingRect(cnt)
                lp_cnt = cnt
                lp_img = img[y:y + h, x:x + w]
                break
    return lp_cnt, lp_img


def _segment_characters(lp_th: ndarray, char_contours: ndarray) -> ndarray:
    char_list = []
    for char_cnt in sorted(char_contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(char_cnt)
        if abs(cv2.contourArea(char_cnt)) < 100 or w > h or h < 10:
            continue
        char_img = lp_th[y:y + h, x:x + w]
        char_img = cv2.bitwise_not(char_img)
        # char_img = cv2.copyMakeBorder(char_img, 5, 0, 5, 0, cv2.BORDER_CONSTANT, 0)
        char_img = cv2.resize(char_img, dsize=(28, 28), interpolation=cv2.INTER_NEAREST)
        char_list.append(char_img)
    return np.stack(char_list) if len(char_list) != 0 else None


def _debug_imshow(debug: int, title: str, img: ndarray) -> None:
    if debug:
        cv2.imshow(title, img)


def main(args: Namespace) -> None:
    debug = args.debug

    original = cv2.imread(args.file)
    _debug_imshow(debug=1, title="Original", img=original)

    gray = cv2.cvtColor(original, code=cv2.COLOR_BGR2GRAY)
    _debug_imshow(debug, title="Grayscale", img=gray)

    blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)  # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    _debug_imshow(debug, title="Blur", img=blur)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(21, 21)).apply(blur)
    _debug_imshow(debug, title="CLAHE", img=clahe)

    otsu_th, otsu_img = cv2.threshold(clahe, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _debug_imshow(debug, title="OTSU", img=otsu_img)

    contours = cv2.findContours(otsu_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    lp_cnt, lp_img = _locate_lp(clahe, contours)
    if lp_cnt is None:
        print("License plate has not been detected!")
        cv2.waitKey()
        cv2.destroyAllWindows()
        return
    _debug_imshow(debug, title="LP", img=lp_img)

    lp_th = cv2.threshold(lp_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    _debug_imshow(debug, title="LP_OTSU", img=lp_th)

    char_contours = cv2.findContours(lp_th, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    segmented_chars = _segment_characters(lp_th, char_contours)
    if segmented_chars is None:
        print("No segmented characters")
        cv2.waitKey()
        cv2.destroyAllWindows()
        return

    # for i, c in enumerate(segmented_chars):
    #     cv2.imshow(str(i), c)
    #     cv2.waitKey()

    model = LeNet5.load_from_checkpoint("../ocr.ckpt")
    y = model(utils.transform_data(segmented_chars))
    predicted_lp = "".join(itemgetter(*torch.argmax(y, dim=1).tolist())(CHARACTERS))
    print(predicted_lp)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="path to an image")
    parser.add_argument("--debug", default=0, type=int, help="if non-zero, every pre-processing step is displayed")
    raise SystemExit(main(parser.parse_args()))
