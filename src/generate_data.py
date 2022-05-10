import argparse
import os
import time
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import config
import utils


def _save_results(data: np.ndarray,
                  labels: np.ndarray,
                  path: Dict[str, str],
                  positives: int,
                  negatives: int) -> Tuple[int, int]:
    for x, label in zip(data, labels):
        if label == config.POSITIVE_LABEL:
            positives += 1
            filename_prefix = utils.join_multiple_paths(path["positive"], str(positives))
        else:
            negatives += 1
            filename_prefix = utils.join_multiple_paths(path["negative"], str(negatives))

        filename = utils.replace_file_extension(filename_prefix, config.DATA_EXT)
        cv2.imwrite(filename, x)

    return positives, negatives


def _generate_data_for_image(image_path: str, gt_bb: Tuple[int, ...]) -> Tuple[np.ndarray, ...]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = utils.apply_clahe(image)
    rp_bbs = utils.apply_selective_search(image)

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
        roi = cv2.resize(roi, dsize=config.OD_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
        data.append(roi)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(data, labels,
                                                      test_size=config.TRAIN_VAL_SPLIT,
                                                      random_state=config.RANDOM_SEED)

    return X_train, X_val, y_train, y_val


def _generate_data(base_path: str, train_path: Dict[str, str], val_path: Dict[str, str]) -> Tuple[int, int]:
    total_positives_train = 0
    total_negatives_train = 0
    total_positives_val = 0
    total_negatives_val = 0

    counter = 0
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if not filename.endswith(config.IMG_EXTENSIONS):
                continue

            start_time = time.time()

            counter += 1
            image_path = utils.join_multiple_paths(dirpath, filename)
            print(f"{counter}.) processing '{image_path}' and saving...")

            gt_path = utils.replace_file_extension(image_path, config.ANNOTATION_EXT)
            gt_bb = utils.read_ground_truth_bb(gt_path)

            X_train, X_val, y_train, y_val = _generate_data_for_image(image_path, gt_bb)

            counters = _save_results(X_train, y_train, train_path, total_positives_train, total_negatives_train)
            total_positives_train = counters[0]
            total_negatives_train = counters[1]

            counters = _save_results(X_val, y_val, val_path, total_positives_val, total_negatives_val)
            total_positives_val = counters[0]
            total_negatives_val = counters[1]

            print(f"  Time elapsed: {(time.time() - start_time)}s.")

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


def _validate_input_arguments(args: Dict[str, Any]) -> Tuple[str, str]:
    base_path = args["base_path"]
    if not os.path.isdir(base_path):
        print(f"Provided '{base_path}' is not a directory!")
        exit(-1)

    data_path = args["data_path"]
    if os.path.exists(data_path) and not os.path.isdir(data_path):
        print(f"Provided '{data_path}' is not a directory!")
        exit(-1)

    return base_path, data_path


def main(args: Dict[str, Any]) -> None:
    start_time = time.time()

    base_path, data_path = _validate_input_arguments(args)

    train_path = {
        "positive": _create_dir(data_path, config.TRAIN_FOLDER, str(config.POSITIVE_LABEL)),
        "negative": _create_dir(data_path, config.TRAIN_FOLDER, str(config.NEGATIVE_LABEL)),
    }
    val_path = {
        "positive": _create_dir(data_path, config.VAL_FOLDER, str(config.POSITIVE_LABEL)),
        "negative": _create_dir(data_path, config.VAL_FOLDER, str(config.NEGATIVE_LABEL)),
    }

    cv2.useOptimized()
    cv2.setNumThreads(os.cpu_count() - 1)

    total_train, total_val = _generate_data(base_path, train_path, val_path)

    print("-----------------------------")
    print(f"TRAIN: {total_train}")
    print(f"VAL: {total_val}")

    print(f"Time elapsed: {(time.time() - start_time) / 60}m.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for object detection.")
    parser.add_argument("base_path", type=str, help="Existing directory for the raw images.")
    parser.add_argument("data_path", type=str, help="(Existing) directory for the training dataset.")
    main(vars(parser.parse_args()))
