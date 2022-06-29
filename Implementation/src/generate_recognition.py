import argparse
import glob
import time
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from tqdm.auto import tqdm

import config
import utils
from pipeline import contrast_enhancement, selective_search


def _save_to_disk(data: np.ndarray, path: str, counter: int) -> None:
    for x in data:
        filename_prefix = utils.join_multiple_paths(path, str(counter))
        filename = utils.replace_file_extension(filename_prefix, ".jpg")
        cv2.imwrite(filename, x)
        counter += 1


def _save_data(data: np.ndarray,
               paths: Dict[str, str],
               lp: str,
               lps_data: Dict[str, Any],
               total_train: int,
               total_val: int) -> Tuple[int, int]:
    val_size = max(1, int(len(data) * config.TRAIN_VAL_SPLIT))
    train_size = max(1, len(data) - val_size)

    # Train and Val example will be the same in this case...
    if len(data) == 1:
        data = np.vstack((data, [data[0]]))

    lp_data_dir = lps_data.get(lp)
    if lp_data_dir is None:
        lp_data_dir = {
            lp: {
                "train": {
                    "path": utils.create_dir(paths["train_path"], lp),
                    "counter": 0
                },
                "val": {
                    "path": utils.create_dir(paths["val_path"], lp),
                    "counter": 0
                }
            }
        }
        lps_data.update(lp_data_dir)

    lp_data_train = lps_data[lp]["train"]
    lp_data_val = lps_data[lp]["val"]

    _save_to_disk(data[:train_size], **lp_data_train)
    _save_to_disk(data[train_size:], **lp_data_val)

    lp_data_train["counter"] += train_size
    lp_data_val["counter"] += val_size

    return total_train + train_size, total_val + val_size


def _generate_data_for_image(image_path: str, gt_bb: Tuple[int, ...]) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = contrast_enhancement(image)
    rp_bbs = selective_search(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data = []

    for x, y, w, h in rp_bbs:
        rp_bb = (x, y, x + w, y + h)
        iou = utils.calculate_iou(gt_bb, rp_bb)
        if iou >= config.IOU_POSITIVE:
            roi = image[y:y + h, x:x + w]
            roi = cv2.resize(roi, config.RECOGNITION_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
            data.append(roi)

    return np.array(data)


def _generate_data(base_path: str, paths: Dict[str, str]) -> Tuple[int, int]:
    total_train = 0
    total_val = 0

    lps_data = {}

    for image_path in tqdm(glob.glob(f"{base_path}/**/*.jpg")):
        gt_path = utils.replace_file_extension(image_path, config.ANNOTATION_EXT)
        gt_bb, gt_lp, two_rows = utils.read_ground_truth(gt_path)

        data = _generate_data_for_image(image_path, gt_bb)

        # If IOU is too strict
        if len(data) == 0:
            continue

        total_train, total_val = _save_data(data, paths, gt_lp, lps_data, total_train, total_val)

    return total_train, total_val


def main(args: Dict[str, Any]) -> None:
    start_time = time.time()

    base_path = args["base_path"]
    data_path = args["data_path"]

    paths = {
        "train_path": utils.create_dir(data_path, config.TRAIN_FOLDER),
        "val_path": utils.create_dir(data_path, config.VAL_FOLDER)
    }

    total_train, total_val = _generate_data(base_path, paths)

    print(f"TRAIN: {total_train}")
    print(f"VAL: {total_val}")

    print(f"Time elapsed: {(time.time() - start_time) / 60:.2f}m.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License Plate Recognition data generator")
    parser.add_argument("base_path", type=str, help="Path to the raw annotated images")
    parser.add_argument("data_path", type=str, help="Path to where the training dataset should be stored")

    main(vars(parser.parse_args()))
