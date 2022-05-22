import argparse
import glob
import time
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from tqdm.auto import tqdm

import config
import utils
from pipeline import detection_preprocessing, selective_search


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
    train_size = len(data) - val_size

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


def _generate_data_for_image(image_path: str, gt_bb: Tuple[int, ...], two_rows: bool) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = detection_preprocessing(image)
    rp_bbs = selective_search(image)

    upper_data = []
    lower_data = []

    def resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, dsize=config.RECOGNITION_INPUT_DIM, interpolation=cv2.INTER_CUBIC)

    for x, y, w, h in rp_bbs:
        rp_bb = (x, y, x + w, y + h)
        iou = utils.calculate_iou(gt_bb, rp_bb)
        if iou >= config.IOU_RECOGNITION:
            roi = image[y:y + h, x:x + w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if two_rows:
                upper = roi[0:h // 2, 0:w]
                lower = roi[h // 2:h, 0:w]
                upper_data.append(resize(upper))
                lower_data.append(resize(lower))
            else:
                upper_data.append(resize(roi))

    return np.array(upper_data), np.array(lower_data)


def _generate_data(base_path: str, paths: Dict[str, str]) -> Tuple[int, int]:
    total_train = 0
    total_val = 0

    lps_data = {}

    for image_path in tqdm(glob.glob(f"{base_path}/**/*.jpg")):
        gt_path = utils.replace_file_extension(image_path, config.ANNOTATION_EXT)
        gt_bb, gt_lp, two_rows = utils.read_ground_truth(gt_path)

        upper_data, lower_data = _generate_data_for_image(image_path, gt_bb, two_rows)

        if len(lower_data) > 0:
            # Hardcoded because this one is specific...
            if image_path.endswith("P6040067.jpg"):
                upper_lp = gt_lp[:3]
                lower_lp = gt_lp[3:]
            else:
                upper_lp = gt_lp[:2]
                lower_lp = gt_lp[2:]
            total_train, total_val = _save_data(upper_data, paths, upper_lp, lps_data, total_train, total_val)
            total_train, total_val = _save_data(lower_data, paths, lower_lp, lps_data, total_train, total_val)
        else:
            total_train, total_val = _save_data(upper_data, paths, gt_lp, lps_data, total_train, total_val)

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
    parser = argparse.ArgumentParser(description="Generate data for License Plate Recognition.")
    parser.add_argument("base_path", type=str, help="Path to the raw annotated images.")
    parser.add_argument("data_path", type=str, help="Path to where the training dataset should be stored.")

    main(vars(parser.parse_args()))
