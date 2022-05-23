import time
from typing import Tuple, Optional

import cv2
import numpy as np
import torch

import config
from datasets import detection_transform_val, recognition_transform_val
from mappings import labels2text
from models import ALPRLightningModule
from predict import predict_binary, predict_ctc


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
        detection_transform_val(
            cv2.resize(image[y:y + h, x:x + w], config.DETECTION_INPUT_DIM, interpolation=cv2.INTER_CUBIC))
        for x, y, w, h in rp_bbs])
    if debug:
        print(f"Elapsed time 'lp_detection::prepare_model_input': {time.time() - start_time:.2f}s.")

    # Predict license plates
    start_time = time.time()
    logits = model(x)
    predictions = predict_binary(logits)
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


def lp_recognition(image: np.ndarray, bb: Tuple[int, ...], model: ALPRLightningModule, debug: bool = False) -> str:
    """
    Performs License Plate Recognition. Expects the input image to be preprocessed with 'detection_preprocessing'.

    :param image: BGR image.
    :param bb: License plate bounding box.
    :param model: License Plate Recognition model.
    :param debug: If True, the progress is print to the console.
    :return: License plate characters.
    """

    bb_x, bb_y, bb_w, bb_h = bb
    lp = image[bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
    if bb_h / bb_w >= 0.6:
        lps = [
            lp[0:bb_h // 2, 0:bb_w],
            lp[bb_h // 2:bb_h, 0:bb_w]
        ]
    else:
        lps = [lp]

    # Prepare input for the model
    start_time = time.time()
    x = torch.stack(
        [recognition_transform_val(cv2.resize(lp, config.RECOGNITION_INPUT_DIM, interpolation=cv2.INTER_CUBIC))
         for lp in lps])
    if debug:
        print(f"Elapsed time 'lp_recognition::prepare_model_input': {time.time() - start_time:.2f}s.")

    # Predict characters
    start_time = time.time()
    logits = model(x)
    logits = logits.permute(1, 0, 2)
    predictions = [predict_ctc(logits[0]), predict_ctc(logits[1])]
    if debug:
        print(f"Elapsed time 'lp_recognition::predict': {time.time() - start_time:.2f}s.")

    return "".join([labels2text(predictions[0]), labels2text(predictions[1])])


def alpr_pipeline(image: np.ndarray,
                  detection_model: ALPRLightningModule,
                  recognition_model: ALPRLightningModule,
                  debug: bool = False) -> Optional[Tuple[Tuple[int, ...], str]]:
    """
    Performs complete Automatic License Plate Recognition.
    :param image: BGR image
    :param detection_model: License Plate Detection model.
    :param recognition_model: License Plate Recognition model.
    :param debug: If True, the progress is print to the console.
    :return: Proposed license plate bounding box along with OCR result or None.
    """

    start_time = time.time()

    # License Plate Detection
    detection_start_time = time.time()
    image = detection_preprocessing(image)
    lp_bb = lp_detection(image, detection_model, debug)
    if debug:
        print(f"Elapsed time 'lp_detection': {time.time() - detection_start_time:.2f}s.")
    if lp_bb is None:
        return None

    # License Plate Recognition
    recognition_start_time = time.time()
    lp = lp_recognition(image, lp_bb, recognition_model, debug)
    if debug:
        print(f"Elapsed time 'lp_recognition': {time.time() - recognition_start_time:.2f}s.")

    if debug:
        print(f"Elapsed time 'alpr_pipeline': {time.time() - start_time:.2f}s.")

    return lp_bb, lp
