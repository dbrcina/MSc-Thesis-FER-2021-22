import argparse
import os
from typing import Any, Dict

import cv2
from tqdm import tqdm

import config
import utils
from pipeline import alpr_pipeline


def running_mean(mean: float, x: float, n: int) -> float:
    if n == 1:
        return x

    return mean + (x - mean) / n


def main(args: Dict[str, Any]) -> None:
    od_model = utils.load_model(args["od_path"])
    ocr_model = utils.load_model(args["ocr_path"])

    n = 0
    mean_iou = 0
    results = []

    for dirpath, _, filenames in os.walk(args["data_path"]):
        for filename in tqdm(filenames, desc=f"{dirpath}"):
            if not filename.endswith(config.IMG_EXTENSIONS):
                continue

            n += 1

            image_path = utils.join_multiple_paths(dirpath, filename)
            image = cv2.imread(image_path)

            result = alpr_pipeline(image, od_model, ocr_model)
            if result is None:
                mean_iou = running_mean(mean_iou, 0, n)
                continue

            (lp_x, lp_y, lp_w, lp_h), lp = result

            gt_path = utils.replace_file_extension(image_path, config.ANNOTATION_EXT)
            gt_bb = utils.read_ground_truth_bb(gt_path)
            lp_bb = (lp_x, lp_y, lp_x + lp_w, lp_y + lp_h)

            iou = utils.calculate_iou(gt_bb, lp_bb)
            if iou < 0.2:
                results.append((image_path, iou))

            mean_iou = running_mean(mean_iou, iou, n)

    with open(config.EVALUATOR_RESULTS_FILE, mode="w") as file:
        file.write(f"Mean IOU = {mean_iou:.3f}\n\n")
        [file.write(f"{path}: {iou:.3f}\n") for path, iou in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALPR evaluation")
    parser.add_argument("data_path", type=str)
    parser.add_argument("od_path", type=str)
    parser.add_argument("ocr_path", type=str)
    main(vars(parser.parse_args()))
