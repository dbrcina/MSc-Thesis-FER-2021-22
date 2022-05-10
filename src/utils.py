import os.path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd

import config


def join_multiple_paths(*paths: str) -> str:
    return os.path.join(*paths)


def replace_file_extension(filename: str, new_ext: str) -> str:
    return os.path.splitext(filename)[0] + new_ext


def read_ground_truth_bb(path: str) -> Tuple[int, ...]:
    df_gt = pd.read_csv(path, index_col=0)
    gt_bb = next(iter(df_gt[["x1", "y1", "x2", "y2"]].itertuples(index=False, name=None)))
    return gt_bb


def calculate_iou(bb1: Tuple[int, ...], bb2: Tuple[int, ...], epsilon: float = 1e-5) -> float:
    # bb: [x1,y1,x2,y2]
    assert len(bb1) == len(bb2)
    assert len(bb1) == 4

    x_inter1 = max(bb1[0], bb2[0])
    y_inter1 = max(bb1[1], bb2[1])
    x_inter2 = min(bb1[2], bb2[2])
    y_inter2 = min(bb1[3], bb2[3])

    width_inter = x_inter2 - x_inter1
    height_inter = y_inter2 - y_inter1
    if width_inter <= 0 or height_inter <= 0:
        return 0.0

    area_inter = width_inter * height_inter
    area_bb1 = abs((bb1[2] - bb1[0]) * (bb1[3] - bb1[1]))
    area_bb2 = abs((bb2[2] - bb2[0]) * (bb2[3] - bb2[1]))

    return float(area_inter) / (area_bb1 + area_bb2 - area_inter + epsilon)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID_SIZE)
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_image)
    clahe_y = clahe.apply(y)
    clahe_image = cv2.merge((clahe_y, cr, cb))
    updated_image = cv2.cvtColor(clahe_image, cv2.COLOR_YCrCb2BGR)
    return updated_image


def apply_selective_search(image: np.ndarray) -> List[Tuple[int, ...]]:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rp_bbs = ss.process()
    return rp_bbs


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged
