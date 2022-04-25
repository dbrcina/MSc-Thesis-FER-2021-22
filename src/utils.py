import os.path

import cv2
import numpy as np


def join_multiple_paths(*paths: str) -> str:
    return os.path.join(*paths)


def replace_file_extension(filename: str, new_ext: str) -> str:
    return os.path.splitext(filename)[0] + new_ext


def selective_search(image: np.ndarray, use_fast: bool = True) -> np.ndarray:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    if use_fast:
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    return ss.process()


def calculate_iou(bb1: tuple[int, ...], bb2: tuple[int, ...], epsilon: float = 1e-5) -> float:
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
