import argparse
import glob
from typing import Any, Dict

import cv2
from torchmetrics.functional import char_error_rate
from tqdm.auto import tqdm

import config
import utils
from pipeline import alpr_pipeline


def main(args: Dict[str, Any]) -> None:
    data_path = args["data_path"]
    detector = utils.load_model(args["detector_path"])
    recognizer = utils.load_model(args["recognizer_path"])

    iou_sum = 0
    cer_sum = 0
    correct_detections = 0
    correct_recognitions = 0

    filenames = glob.glob(f"{data_path}/**/*.jpg", recursive=True)

    for image_path in tqdm(filenames):
        image = cv2.imread(image_path)

        result = alpr_pipeline(image, detector, recognizer)
        if result is None:
            print(f"'{image_path}'   : didn't manage to find license plate!")
            continue

        (x, y, w, h), lp = result
        lp_bb = (x, y, x + w, y + h)

        gt_path = utils.replace_file_extension(image_path, config.ANNOTATION_EXT)
        gt_bb, gt_lp, _ = utils.read_ground_truth(gt_path)

        iou = utils.calculate_iou(gt_bb, lp_bb)
        iou_sum += iou

        cer_sum += char_error_rate(lp, gt_lp).item()

        if gt_lp != lp or iou <= 0.5:
            print(f"'{image_path}'   : IOU={iou}   Predicted={lp}   Target={gt_lp}")
        elif gt_lp == lp:
            correct_recognitions += 1

        if iou >= 0.7:
            correct_detections += 1

    n = len(filenames)
    print("-------------------------")
    print(f"Detection accuracy   : {correct_detections / n:.5f}")
    print(f"Recognition accuracy : {correct_recognitions / n:.5f}")
    print(f"Mean IOU             : {iou_sum / n:.5f}")
    print(f"Char Error Rate      : {cer_sum / n:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALPR evaluation")
    parser.add_argument("data_path", type=str)
    parser.add_argument("detector_path", type=str)
    parser.add_argument("recognizer_path", type=str)

    main(vars(parser.parse_args()))
