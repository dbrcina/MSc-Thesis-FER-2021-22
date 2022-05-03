import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    assert pts.shape == (4, 2)

    # [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# https://theailearner.com/tag/cv2-getperspectivetransform/
def four_point_transform(image: np.ndarray, pts: np.ndarray):
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # compute the width of the new image
    width_top = int(np.linalg.norm(tl - tr))
    width_bottom = int(np.linalg.norm(bl - br))
    width = max(width_top, width_bottom)

    # compute the height of the new image
    height_left = int(np.linalg.norm(tl - bl))
    height_right = int(np.linalg.norm(tr - br))
    height = max(height_left, height_right)

    # construct the set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # construct the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped
