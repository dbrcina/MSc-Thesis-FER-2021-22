import string
import time
from typing import Tuple, Optional

import cv2
import numpy as np
import torch

import config
from datasets import od_transform_val, ocr_transform_val
from models import ALPRLightningModule

CHARACTERS = list(string.digits + string.ascii_uppercase)


def detection_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Performs image preprocessing for License Plate Detection on the provided image.

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


def lp_detection(image: np.ndarray, model: ALPRLightningModule, debug: bool = False) -> Optional[Tuple[int, ...]]:
    """
    Performs License Plate Detection.

    :param image: BGR image.
    :param model: License Plate Detection model.
    :param debug: If True, the progress is print to the console.
    :return: Proposed license plate bounding box or None.
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
    lp_bbs = rp_bbs[topk.indices]

    # Filter based on proportions
    def filter_lp(bb: Tuple[int, ...]) -> bool:
        x, y, w, h = bb
        ratio = w / h
        correct_ratio = config.LP_WIDTH_HEIGHT_RATIO_MIN <= ratio <= config.LP_WIDTH_HEIGHT_RATIO_MAX
        correct_width = config.LP_WIDTH_MIN <= w <= config.LP_WIDTH_MAX
        correct_height = config.LP_HEIGHT_MIN <= h <= config.LP_HEIGHT_MAX
        return all((correct_ratio, correct_width, correct_height))

    lp_bbs = list(filter(filter_lp, lp_bbs))

    # If there are no valid bbs
    if len(lp_bbs) == 0:
        return None

    # TODO: Apply some more filtering on proposed license plates?
    return lp_bbs[0]


def recognition_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Performs image preprocessing for License Plate Recognition on the provided image.

    :param image: BGR image.
    :return: Preprocessed image.
    """

    # BGR > Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying gaussian filter
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Applying adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    return thresh


def lp_recognition(image: np.ndarray, bb: Tuple[int, ...], model: ALPRLightningModule, debug: bool = False) -> str:
    """
    Performs License Plate Recognition.

    :param image: BGR image.
    :param bb: License plate bounding box.
    :param model: OCR model.
    :param debug: If True, the progress is print to the console.
    :return: License plate characters or None.
    """

    bb_x, bb_y, bb_w, bb_h = bb
    image = image[bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]

    # Preprocessing
    start_time = time.time()
    image = recognition_preprocessing(image)
    if debug:
        print(f"Elapsed time 'lp_recognition::preprocessing': {time.time() - start_time:.2f}s.")

    # Find contours
    start_time = time.time()
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_bbs = list(map(cv2.boundingRect, contours))
    heights = list(map(lambda bb: bb[3], contour_bbs))
    counts = np.bincount(heights)
    height = np.argmax(counts)
    chars = []
    for contour, (x, y, w, h) in sorted(zip(contours, contour_bbs), key=lambda x: x[1][0]):
        if (height - 2) <= h <= (height + 2):
            chars.append(image[y:y + h, x:x + w])
    if debug:
        print(f"Elapsed time 'lp_recognition::contours': {time.time() - start_time:.2f}s.")

    # If there are no valid contours
    if len(chars) == 0:
        return ""

    # Prepare input for ocr
    start_time = time.time()
    # Gray -> RGB because of torchvision transforms
    bb_x = torch.stack([
        ocr_transform_val(cv2.cvtColor(
            cv2.resize(c, dsize=config.OCR_INPUT_DIM, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB))
        for c in chars])
    if debug:
        print(f"Elapsed time 'lp_recognition::prepare_model_input': {time.time() - start_time:.2f}s.")

    # Predict characters
    start_time = time.time()
    char_preds = model.predict(bb_x)
    if debug:
        print(f"Elapsed time 'lp_recognition::predict': {time.time() - start_time:.2f}s.")

    return "".join(list(map(lambda x: CHARACTERS[x], torch.argmax(char_preds, dim=1))))


def alpr_pipeline(image: np.ndarray,
                  od_model: ALPRLightningModule,
                  ocr_model: ALPRLightningModule,
                  debug: bool = False) -> Optional[Tuple[Tuple[int, ...], str]]:
    """
    Performs complete Automatic License Plate Recognition.
    :param image: BGR image
    :param od_model: License Plate Detection model.
    :param ocr_model: OCR model.
    :param debug: If True, the progress is print to the console.
    :return: Proposed license plate bounding box along with OCR result or None.
    """

    start_time = time.time()

    # License Plate Detection
    detection_start_time = time.time()
    image = detection_preprocessing(image)
    lp_bb = lp_detection(image, od_model, debug)
    if debug:
        print(f"Elapsed time 'lp_detection': {time.time() - detection_start_time:.2f}s.")
    if lp_bb is None:
        return None

    # License Plate Recognition
    recognition_start_time = time.time()
    lp = lp_recognition(image, lp_bb, ocr_model, debug)
    if debug:
        print(f"Elapsed time 'lp_recognition': {time.time() - recognition_start_time:.2f}s.")

    if debug:
        print(f"Elapsed time 'alpr_pipeline': {time.time() - start_time:.2f}s.")

    return lp_bb, lp
