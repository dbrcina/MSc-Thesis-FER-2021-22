import string
import time
from typing import Tuple

import cv2
import numpy as np
import torch

import config
import utils
from datasets import VAL_TRANSFORM_OD, VAL_TRANSFORM_OCR
from train import ALPRLightningModule

CHARACTERS = list(string.digits + string.ascii_uppercase)


def alpr_pipeline(image: np.ndarray,
                  od_model: ALPRLightningModule,
                  ocr_model: ALPRLightningModule,
                  debug: bool = True) -> Tuple[float, Tuple[int, ...], str]:
    start_time = time.time()

    # RGB -> BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # CLAHE
    image = utils.apply_clahe(image)

    # Object detection.
    locate_lp_start_time = time.time()
    lp_prob, lp_bb = locate_lp(image, od_model, debug)
    if debug:
        print(f"Elapsed time 'locate_lp': {time.time() - locate_lp_start_time:.2f}s.")
        print(f"License plate probability: {lp_prob:.3f}.")

    # OCR
    lp_ocr_start_time = time.time()
    lp = lp_ocr(image, lp_bb, ocr_model, debug)
    if debug:
        print(f"Elapsed time 'lp_ocr': {time.time() - lp_ocr_start_time:.2f}s.")

    if debug:
        print(f"Elapsed time 'alpr_pipeline': {time.time() - start_time:.2f}s.")

    return lp_prob, lp_bb, lp


def locate_lp(image: np.ndarray, od_model: ALPRLightningModule, debug: bool) -> Tuple[float, Tuple[int, ...]]:
    # Selective search
    selective_search_start_time = time.time()
    rp_bbs = utils.apply_selective_search(image)[:config.MAX_INFERENCE_SAMPLES]
    if debug:
        print(f"Elapsed time 'locate_lp::selective_search': {time.time() - selective_search_start_time:.2f}s.")

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare input for od.
    prepare_input_start_time = time.time()
    od_input = torch.stack([
        VAL_TRANSFORM_OD(
            cv2.resize(image[y:y + h, x:x + w], config.OD_INPUT_DIM, interpolation=cv2.INTER_CUBIC))
        for x, y, w, h in rp_bbs])
    if debug:
        print(f"Elapsed time 'locate_lp::prepare_od_input': {time.time() - prepare_input_start_time:.2f}s.")

    # Predict license plate.
    predict_start_time = time.time()
    lp_preds = od_model.predict(od_input)
    if debug:
        print(f"Elapsed time 'locate_lp::predict': {time.time() - predict_start_time:.2f}s.")

    index = lp_preds.argmax()
    lp_prob = lp_preds[index].item()
    lp_bb = rp_bbs[index]
    return lp_prob, lp_bb


def lp_ocr(image: np.ndarray, lp_bb: Tuple[int, ...], ocr_model: ALPRLightningModule, debug: bool) -> str:
    x, y, w, h = lp_bb
    image = image[y:y + h, x:x + w]

    # Preprocessing
    preprocess_start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    if debug:
        print(f"Elapsed time 'lp_ocr::preprocess': {time.time() - preprocess_start_time:.2f}s.")

    # Find contours
    contours_start_time = time.time()
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_bbs = list(map(cv2.boundingRect, contours))
    chars = []
    for contour, (x, y, w, h) in sorted(zip(contours, contours_bbs), key=lambda x: (x[1][1], x[1][0])):
        if abs(cv2.contourArea(contour)) < 100 or w > h or h < 10:
            continue
        chars.append(thresh[y:y + h, x:x + w])
    if debug:
        print(f"Elapsed time 'lp_ocr::contours': {time.time() - contours_start_time:.2f}s.")

    if len(chars) == 0:
        print("Didn't manage to find any character contours!")
        return ""

    # Prepare input for ocr.
    prepare_input_start_time = time.time()
    ocr_input = torch.stack([
        VAL_TRANSFORM_OCR(cv2.cvtColor(
            cv2.resize(c, dsize=config.OCR_INPUT_DIM, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB))
        for c in chars])
    if debug:
        print(f"Elapsed time 'lp_ocr::prepare_ocr_input': {time.time() - prepare_input_start_time:.2f}s.")

    # Predict characters.
    predict_start_time = time.time()
    char_preds = ocr_model.predict(ocr_input)
    if debug:
        print(f"Elapsed time 'lp_ocr::predict': {time.time() - predict_start_time:.2f}s.")

    return "".join(list(map(lambda x: CHARACTERS[x], torch.argmax(char_preds, dim=1))))
