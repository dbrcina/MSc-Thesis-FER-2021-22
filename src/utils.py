import os.path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from models import ALPRLightningModule


def join_multiple_paths(*paths: str) -> str:
    return os.path.join(*paths)


def create_dir(*paths: str) -> str:
    path = join_multiple_paths(*paths)
    os.makedirs(path, exist_ok=True)
    return path


def replace_file_extension(filename: str, new_ext: str) -> str:
    return os.path.splitext(filename)[0] + new_ext


def read_ground_truth(path: str) -> Tuple[Tuple[int, ...], str, bool]:
    df_gt = pd.read_csv(path, index_col=0)
    x1, y1, x2, y2, lp, two_rows = next(iter(df_gt.itertuples(index=False, name=None)))
    return (x1, y1, x2, y2), lp, two_rows


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


def load_model(path: str) -> ALPRLightningModule:
    model = ALPRLightningModule.load_from_checkpoint(path)
    model.eval()
    return model


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged
