import time
from typing import Tuple, Optional

import cv2
import numpy as np
import torch

import config
from ctc_decoder import ctc_decoder
from datasets import detection_transform_val, recognition_transform_val
from models import ALPRLightningModule


def contrast_enhancement(image: np.ndarray) -> np.ndarray:
    # BGR -> YCrCb
    y, cr, cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))

    # Applying CLAHE on Y component
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                            tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    y = clahe.apply(y)

    # Applying gaussian filter on Y component
    y = cv2.GaussianBlur(y, config.GAUSSIAN_BLUR_KSIZE, config.GAUSSIAN_BLUR_SIGMAX)

    # YCrCb -> BGR
    image = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

    return image


def selective_search(image: np.ndarray) -> np.ndarray:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rp_bbs = ss.process()
    return rp_bbs


def lp_detection(image: np.ndarray, model: ALPRLightningModule, debug: bool = False) -> Optional[Tuple[int, ...]]:
    assert image.shape[-1] == 3, "Expecting BGR image"

    # Selective search
    start_time = time.time()
    rp_bbs = selective_search(image)[:config.MAX_INFERENCE_SAMPLES]
    if debug:
        print(f"  selective_search: {time.time() - start_time:.2f}s.")

    # BGR -> Gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Prepare input for the model
    start_time = time.time()

    # Filter based on proportions
    def filter_lp(bb: Tuple[int, ...]) -> bool:
        x, y, w, h = bb
        ratio = w / h
        correct_ratio = config.LP_WIDTH_HEIGHT_RATIO_MIN <= ratio <= config.LP_WIDTH_HEIGHT_RATIO_MAX
        correct_width = config.LP_WIDTH_MIN <= w <= config.LP_WIDTH_MAX
        correct_height = config.LP_HEIGHT_MIN <= h <= config.LP_HEIGHT_MAX
        return all((correct_ratio, correct_width, correct_height))

    rp_bbs = np.array(list(filter(filter_lp, rp_bbs)))
    if len(rp_bbs) == 0:
        return None

    x = torch.stack([
        detection_transform_val(
            cv2.resize(image[y:y + h, x:x + w], config.DETECTION_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
        )
        for x, y, w, h in rp_bbs])
    if debug:
        print(f"  prepare_input: {time.time() - start_time:.2f}s.")

    # Predict license plates
    start_time = time.time()
    logits = model(x)
    predictions = torch.sigmoid(logits)
    if debug:
        print(f"  predict: {time.time() - start_time:.2f}s.")

    # Retrieve only top k predictions
    topk = torch.topk(predictions.view(-1), k=config.LP_TOPK)
    lp_bbs = rp_bbs[topk.indices]

    return lp_bbs[0]


def lp_recognition(image: np.ndarray, model: ALPRLightningModule, debug: bool = False) -> str:
    assert len(image.shape) == 2, "Expecting grayscale image"

    # Prepare input for the model
    start_time = time.time()
    x = recognition_transform_val(cv2.resize(image, config.RECOGNITION_INPUT_DIM, interpolation=cv2.INTER_CUBIC))
    x = x[None, :, :, :]  # Need to add batch size 1
    if debug:
        print(f"  prepare_input: {time.time() - start_time:.2f}s.")

    # Predict characters
    start_time = time.time()
    logits = model(x)
    predictions = ctc_decoder(torch.log_softmax(logits, dim=-1), mode="beam_search")
    if debug:
        print(f"  predict: {time.time() - start_time:.2f}s.")

    return predictions[0]


def alpr_pipeline(image: np.ndarray,
                  detector: ALPRLightningModule,
                  recognizer: ALPRLightningModule,
                  debug: bool = False) -> Optional[Tuple[Tuple[int, ...], str]]:
    start_time = time.time()

    # Contrast enhancement
    image = contrast_enhancement(image)

    # License Plate Detection
    if debug:
        print("License Plate Detection:")
    detection_start_time = time.time()
    bb = lp_detection(image, detector, debug)
    if debug:
        print(f"Elapsed time: {time.time() - detection_start_time:.2f}s.")
    if bb is None:
        return None

    # License Plate Recognition
    if debug:
        print("License Plate Recognition:")
    recognition_start_time = time.time()
    x, y, w, h = bb
    lp = lp_recognition(cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), recognizer, debug)
    if debug:
        print(f"Elapsed time: {time.time() - recognition_start_time:.2f}s.")

    if debug:
        print(f"Elapsed time pipeline: {time.time() - start_time:.2f}s.")

    return bb, lp
