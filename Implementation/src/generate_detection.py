import argparse
import glob
import time
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import config
import utils
from pipeline import contrast_enhancement, selective_search


def _save_results(data: np.ndarray,
                  labels: np.ndarray,
                  path: Dict[str, str],
                  lp: str,
                  positives: int,
                  negatives: int) -> Tuple[int, int]:
    for x, label in zip(data, labels):
        if label == config.POSITIVE_LABEL:
            positives += 1
            sample_name = f"{positives}-{lp}"
            filename_prefix = utils.join_multiple_paths(path["positive"], sample_name)
        else:
            negatives += 1
            filename_prefix = utils.join_multiple_paths(path["negative"], str(negatives))

        filename = utils.replace_file_extension(filename_prefix, config.DATA_EXT)
        cv2.imwrite(filename, x)

    return positives, negatives


def _generate_data_for_image(image_path: str, gt_bb: Tuple[int, ...]) -> Tuple[np.ndarray, ...]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = contrast_enhancement(image)
    rp_bbs = selective_search(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    positives = 0
    negatives = 0
    data = []
    labels = []

    for x, y, w, h in rp_bbs:
        if positives >= config.MAX_POSITIVE_SAMPLES and negatives >= config.MAX_NEGATIVE_SAMPLES:
            break

        rp_bb = (x, y, x + w, y + h)
        iou = utils.calculate_iou(gt_bb, rp_bb)
        if iou >= config.IOU_POSITIVE and positives < config.MAX_POSITIVE_SAMPLES:
            label = config.POSITIVE_LABEL
            positives += 1
        elif iou <= config.IOU_NEGATIVE and negatives < config.MAX_NEGATIVE_SAMPLES:
            label = config.NEGATIVE_LABEL
            negatives += 1
        else:
            continue

        roi = image[y:y + h, x:x + w]
        roi = cv2.resize(roi, config.DETECTION_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
        data.append(roi)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=config.TRAIN_VAL_SPLIT)

    return X_train, X_val, y_train, y_val


def _generate_data(base_path: str, train_path: Dict[str, str], val_path: Dict[str, str]) -> Tuple[int, int]:
    total_positives_train = 0
    total_negatives_train = 0
    total_positives_val = 0
    total_negatives_val = 0

    for image_path in tqdm(glob.glob(f"{base_path}/**/*.jpg")):
        gt_path = utils.replace_file_extension(image_path, config.ANNOTATION_EXT)
        gt_bb, gt_lp, _ = utils.read_ground_truth(gt_path)

        X_train, X_val, y_train, y_val = _generate_data_for_image(image_path, gt_bb)

        total_positives_train, total_negatives_train = _save_results(X_train, y_train, train_path, gt_lp,
                                                                     total_positives_train, total_negatives_train)

        total_positives_val, total_negatives_val = _save_results(X_val, y_val, val_path, gt_lp,
                                                                 total_positives_val, total_negatives_val)

    total_train = total_positives_train + total_negatives_train
    total_val = total_positives_val + total_negatives_val
    return total_train, total_val


def main(args: Dict[str, Any]) -> None:
    start_time = time.time()

    base_path = args["base_path"]
    data_path = args["data_path"]

    train_path = {
        "positive": utils.create_dir(data_path, config.TRAIN_FOLDER, str(config.POSITIVE_LABEL)),
        "negative": utils.create_dir(data_path, config.TRAIN_FOLDER, str(config.NEGATIVE_LABEL)),
    }
    val_path = {
        "positive": utils.create_dir(data_path, config.VAL_FOLDER, str(config.POSITIVE_LABEL)),
        "negative": utils.create_dir(data_path, config.VAL_FOLDER, str(config.NEGATIVE_LABEL)),
    }

    total_train, total_val = _generate_data(base_path, train_path, val_path)

    print(f"TRAIN: {total_train}")
    print(f"VAL: {total_val}")

    print(f"Time elapsed: {(time.time() - start_time) / 60}m.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License Plate Detection data generator")
    parser.add_argument("base_path", type=str, help="Path to the raw annotated images")
    parser.add_argument("data_path", type=str, help="Path to where the training dataset should be stored")

    main(vars(parser.parse_args()))
