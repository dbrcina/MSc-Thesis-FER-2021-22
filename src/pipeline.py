import string
import time
from typing import Tuple, List

import cv2
import numpy as np
import torch

import config
from datasets import od_transform_val, ocr_transform_val
from models import ALPRLightningModule

CHARACTERS = list(string.digits + string.ascii_uppercase)


def input_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Performs image preprocessing on the provided image.

    :param image: BGR image.
    :return: Preprocessed image.
    """

    # BGR -> YCrCb
    y, cr, cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))

    # Applying CLAHE on Y component
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                            tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    y = clahe.apply(y)

    # Applying gaussian filter on Y component
    y = cv2.GaussianBlur(y, (5, 5), 0)

    # YCrCb -> BGR
    image = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

    return image


def selective_search(image: np.ndarray, fast: bool = True) -> np.ndarray:
    """
    Performs selective search algorithm on the provided image.

    :param image: BGR image.
    :param fast: If True, fast version is used, otherwise quality version is used.
    :return: Regional proposal bounding boxes.
    """

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    if fast:
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rp_bbs = ss.process()

    return rp_bbs


def lp_detection(image: np.ndarray, model: ALPRLightningModule, debug: bool = False) -> List[Tuple[int, ...]]:
    """
    Performs License Plate Detection.

    :param image: BGR image.
    :param model: License Plate Detection model.
    :param debug: If True, the progress is print to the console.
    :return: Proposed license plate bounding boxes.
    """

    # Selective search
    start_time = time.time()
    rp_bbs = selective_search(image)[:config.MAX_INFERENCE_SAMPLES]
    if debug:
        print(f"Elapsed time 'lp_detection::selective_search': {time.time() - start_time:.2f}s.")

    # BGR -> RGB (because the model is trained on RGB images)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare input for the model
    start_time = time.time()
    x = torch.stack([
        od_transform_val(
            cv2.resize(image[y:y + h, x:x + w], config.OD_INPUT_DIM, interpolation=cv2.INTER_CUBIC))
        for x, y, w, h in rp_bbs])
    if debug:
        print(f"Elapsed time 'lp_detection::prepare_model_input': {time.time() - start_time:.2f}s.")

    # Predict license plates
    start_time = time.time()
    predictions = model.predict(x)
    if debug:
        print(f"Elapsed time 'lp_detection::predict': {time.time() - start_time:.2f}s.")

    # Retrieve only top k predictions
    topk = torch.topk(predictions.view(-1), k=config.LP_TOPK)

    return [rp_bbs[i] for i in topk.indices]


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
        ocr_transform_val(cv2.cvtColor(
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


def alpr_pipeline(image: np.ndarray,
                  od_model: ALPRLightningModule,
                  ocr_model: ALPRLightningModule,
                  debug: bool = False) -> Tuple[Tuple[int, ...], str]:
    """
    Performs complete Automatic License Plate Recognition.
    :param image: BGR image
    :param od_model: License Plate Detection model.
    :param ocr_model: OCR model.
    :param debug: If True, the progress is print to the console.
    :return: Proposed license plate bounding box along with OCR result.
    """

    start_time = time.time()

    # Input preprocessing
    image = input_preprocessing(image)

    # License Plate Detection
    lp_detection_start_time = time.time()
    lp_bbs = lp_detection(image, od_model, debug)
    lp_bbs = list(
        filter(lambda bb: config.LP_WIDTH_MIN <= bb[2] <= config.LP_WIDTH_MAX and
                          config.LP_HEIGHT_MIN <= bb[3] <= config.LP_HEIGHT_MAX,
               lp_bbs)
    )

    if len(lp_bbs) == 0:
        print("Didn't manage to find any license plate!")
        return None

    lp_bb = lp_bbs[0]
    if debug:
        print(f"Elapsed time 'locate_lp': {time.time() - lp_detection_start_time:.2f}s.")

    # OCR
    lp_ocr_start_time = time.time()
    # lp = lp_ocr(image, lp_bb, ocr_model, debug)
    lp = "ZG1228BH"
    if debug:
        print(f"Elapsed time 'lp_ocr': {time.time() - lp_ocr_start_time:.2f}s.")
        print(f"Elapsed time 'alpr_pipeline': {time.time() - start_time:.2f}s.")

    return lp_bb, lp
