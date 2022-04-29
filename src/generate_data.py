import argparse
import os
import random
import time
from typing import Dict, Tuple, List

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn import model_selection
from torchvision import transforms

import config
import utils


def _save_results(X: np.ndarray,
                  y: np.ndarray,
                  path: Dict[str, str],
                  positives: int,
                  negatives: int) -> Tuple[int, int]:
    for x, label in zip(X, y):
        if label == config.POSITIVE_LABEL:
            positives += 1
            filename_prefix = utils.join_multiple_paths(path["positive"], str(positives))
        else:
            negatives += 1
            filename_prefix = utils.join_multiple_paths(path["negative"], str(negatives))

        filename = filename_prefix + config.DATA_EXT
        cv2.imwrite(filename, x)

    return positives, negatives


def _generate_samples(image: np.ndarray,
                      bbs_ious: List[Tuple[Tuple[int, ...], float]],
                      data: List[np.ndarray],
                      labels: List[int],
                      label: int,
                      transform_list: List[object] = None) -> None:
    # Remove iou
    bbs = list(map(lambda bb_iou: bb_iou[0], bbs_ious))

    for x, y, w, h in bbs:
        roi = image[y:y + h, x:x + w]

        data.append(roi)
        labels.append(label)

        if label == config.POSITIVE_LABEL:
            roi_pil = Image.fromarray(roi)
            for transform in transform_list:
                roi = transform(roi_pil)
                roi = np.array(roi)
                roi = cv2.resize(roi, dsize=config.RCNN_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
                data.append(roi)
                labels.append(label)


def _generate_data_for_image(image: np.ndarray,
                             gt_bb: Tuple[int, ...],
                             transform_list: List[object]) -> Tuple[np.ndarray, ...]:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rp_bbs = ss.process()

    # Map bbs to (bb, iou)
    rp_bbs_ious = list(
        map(lambda bb: (
            (bb[0], bb[1], bb[2], bb[3]),
            utils.calculate_iou(gt_bb, (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
        ),
            rp_bbs)
    )

    # Sort by iou reversed
    rp_bbs_ious_sorted = sorted(rp_bbs_ious, key=lambda bb_iou: bb_iou[1], reverse=True)

    # Retrieve positive and negative examples
    positives = list(
        filter(lambda bb_iou: bb_iou[1] >= config.IOU_POSITIVE, rp_bbs_ious_sorted[:config.MAX_POSITIVE_SAMPLES])
    )
    negatives = list(
        filter(lambda bb_iou: bb_iou[1] <= config.IOU_NEGATIVE, rp_bbs_ious_sorted)
    )
    random.shuffle(negatives)
    negatives = negatives[:config.MAX_NEGATIVE_SAMPLES]

    data = []
    labels = []

    _generate_samples(image, positives, data, labels, config.POSITIVE_LABEL, transform_list)
    _generate_samples(image, negatives, data, labels, config.NEGATIVE_LABEL)

    X_train, X_val, y_train, y_val = model_selection.train_test_split(np.array(data), np.array(labels),
                                                                      test_size=config.TRAIN_VAL_SPLIT,
                                                                      random_state=config.RANDOM_SEED,
                                                                      stratify=labels)

    return X_train, X_val, y_train, y_val


# BGR -> YCrCb -> clahe(Y) -> YCrCb -> BGR
def _preprocess_image(image_path: str, clahe: cv2.CLAHE) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_image)
    clahe_y = clahe.apply(y)
    clahe_img = cv2.merge((clahe_y, cr, cb))
    updated_image = cv2.cvtColor(clahe_img, cv2.COLOR_YCrCb2BGR)
    return updated_image


def _generate_data(base_path: str, train_path: Dict[str, str], val_path: Dict[str, str]) -> Tuple[int, int]:
    total_positives_train = 0
    total_negatives_train = 0
    total_positives_val = 0
    total_negatives_val = 0

    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    transform_list = [
        transforms.RandomRotation(config.RANDOM_ROTATION, interpolation=transforms.InterpolationMode.BILINEAR,
                                  expand=True),
        transforms.RandomPerspective(p=1.0, distortion_scale=0.2)]

    counter = 0
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if not filename.endswith(config.IMG_EXTENSIONS):
                continue

            start_time = time.time()

            image_path = utils.join_multiple_paths(dirpath, filename)
            image_path = r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\baza_slika\070603\P6070075.jpg"
            counter += 1
            print(f"{counter}.) processing '{image_path}' and saving...")
            image = _preprocess_image(image_path, clahe)

            gt_path = utils.replace_file_extension(image_path, config.ANNOTATION_EXT)
            df_gt = pd.read_csv(gt_path, index_col=0)
            gt_bb = next(iter(df_gt[["x1", "y1", "x2", "y2"]].itertuples(index=False, name=None)))

            X_train, X_val, y_train, y_val = _generate_data_for_image(image, gt_bb, transform_list)

            counters = _save_results(X_train, y_train, train_path, total_positives_train, total_negatives_train)
            total_positives_train = counters[0]
            total_negatives_train = counters[1]

            counters = _save_results(X_val, y_val, val_path, total_positives_val, total_negatives_val)
            total_positives_val = counters[0]
            total_negatives_val = counters[1]

            print(f"  Time elapsed: {(time.time() - start_time)}s.")

            if counter == 1:
                break

        if counter == 1:
            break

    total_train = total_positives_train + total_negatives_train
    total_val = total_positives_val + total_negatives_val
    return total_train, total_val


def _create_dir(*paths: str) -> str:
    path = utils.join_multiple_paths(*paths)
    if not os.path.exists(path):
        print(f"Creating directory '{path}'...")
        os.makedirs(path)
        print("  Directory created!")
    return path


def _validate_input_arguments(args: argparse.Namespace) -> Tuple[str, str]:
    base_path = args.base_path
    if not os.path.isdir(base_path):
        print(f"Provided '{base_path}' is not a directory!")
        exit(-1)

    data_path = args.data_path
    if os.path.exists(data_path) and not os.path.isdir(data_path):
        print(f"Provided '{data_path}' is not a directory!")
        exit(-1)

    return base_path, data_path


def main(args: argparse.Namespace) -> None:
    start_time = time.time()

    base_path, data_path = _validate_input_arguments(args)

    train_path = {
        "positive": _create_dir(data_path, config.TRAIN_PATH, str(config.POSITIVE_LABEL)),
        "negative": _create_dir(data_path, config.TRAIN_PATH, str(config.NEGATIVE_LABEL)),
    }
    val_path = {
        "positive": _create_dir(data_path, config.VAL_PATH, str(config.POSITIVE_LABEL)),
        "negative": _create_dir(data_path, config.VAL_PATH, str(config.NEGATIVE_LABEL)),
    }

    cv2.useOptimized()
    cv2.setNumThreads(os.cpu_count() - 1)
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    total_train, total_val = _generate_data(base_path, train_path, val_path)

    print("-----------------------------")
    print(f"TRAIN: {total_train}")
    print(f"VAL: {total_val}")

    print(f"Time elapsed: {(time.time() - start_time) / 60}m.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=str, help="Existing directory for the raw images.")
    parser.add_argument("data_path", type=str, help="(Existing) directory for the training dataset.")
    main(parser.parse_args())
