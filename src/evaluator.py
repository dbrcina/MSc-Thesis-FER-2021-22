import argparse
import glob
from typing import Any, Dict

import cv2
from torchmetrics.functional import char_error_rate
from tqdm.auto import tqdm

import config
import utils
from pipeline import lp_detection, lp_recognition, contrast_enhancement

cv2.useOptimized()
cv2.setNumThreads(4)


def main(args: Dict[str, Any]) -> None:
    data_path = args["data_path"]
    detector = utils.load_model(args["detector_path"])
    recognizer = utils.load_model(args["recognizer_path"])

    iou_sum = 0
    cer_sum = 0
    correct_detections = 0
    correct_recognitions = 0
    incorrect_length = 0

    filenames = glob.glob(f"{data_path}/**/*.jpg", recursive=True)

    for image_path in tqdm(filenames):
        gt_path = utils.replace_file_extension(image_path, config.ANNOTATION_EXT)
        gt_bb, gt_lp, _ = utils.read_ground_truth(gt_path)

        image = cv2.imread(image_path)
        image = contrast_enhancement(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        final_iou = None
        final_lp = None
        for _ in range(3):
            bb = lp_detection(image, detector)
            if bb is None:
                continue

            x, y, w, h = bb
            iou = utils.calculate_iou(gt_bb, (x, y, x + w, y + h))
            if final_iou is None or iou > final_iou:
                final_iou = iou
                final_lp = lp_recognition(image_gray[y:y + h, x:x + w], recognizer)

        if final_iou is None:
            print(f"'{image_path}' : didn't manage to find license plate!")
            continue

        iou_sum += final_iou
        cer_sum += char_error_rate(final_lp, gt_lp).item()

        if gt_lp != final_lp or final_iou <= 0.5:
            print(f"'{image_path}' : IOU={final_iou:.3f}   Predicted={final_lp}   Target={gt_lp}")
            if len(gt_lp) != len(final_lp):
                incorrect_length += 1

        if final_iou > 0.5:
            correct_detections += 1
        if gt_lp == final_lp:
            correct_recognitions += 1

    n = len(filenames)
    print("-------------------------")
    print(f"Detection   accuracy : {correct_detections / n:.5f}")
    print(f"Recognition accuracy : {correct_recognitions / n:.5f}")
    print(f"Mean IOU             : {iou_sum / n:.5f}")
    print(f"Char Error Rate      : {cer_sum / n:.5f}")
    print(f"Incorrect length     : {incorrect_length / n:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALPR evaluation")
    parser.add_argument("data_path", type=str)
    parser.add_argument("detector_path", type=str)
    parser.add_argument("recognizer_path", type=str)

    main(vars(parser.parse_args()))
