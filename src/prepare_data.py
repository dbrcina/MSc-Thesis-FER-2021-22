import argparse
import os
import time

import cv2
import pandas as pd

import config
import utils

cv2.useOptimized()
cv2.setNumThreads(os.cpu_count() - 1)


def _validate_input_arguments(args: argparse.Namespace) -> tuple[str, str]:
    base_path = args.base_path
    if not os.path.isdir(base_path):
        print(f"Provided '{base_path}' is not a directory!")
        exit(-1)

    data_path = args.data_path
    if os.path.exists(data_path) and not os.path.isdir(data_path):
        print(f"Provided '{data_path}' is not a directory!")
        exit(-1)

    return base_path, data_path


def _create_dir(*paths: str) -> str:
    path = utils.join_multiple_paths(*paths)
    if not os.path.exists(path):
        print(f"Creating directory '{path}'...")
        os.makedirs(path)
        print(f"'{path}' directory created!")
    return path


def _process_images(base_path: str, positive_data_path: str, negative_data_path: str) -> tuple[int, int]:
    total_positives = 0
    total_negatives = 0

    counter = 0
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if not filename.endswith(config.IMG_EXTENSIONS):
                continue

            start_time = time.time()

            image_fp = utils.join_multiple_paths(dirpath, filename)
            counter += 1
            print(f"{counter}) Processing '{image_fp}'...")

            gt_fp = utils.replace_file_extension(image_fp, config.ANNOTATION_EXT)

            # Load ground truth and make a list of bounding boxes:
            df_gt = pd.read_csv(gt_fp, index_col=0)
            gt_bbs = [bb for bb in df_gt[["x1", "y1", "x2", "y2"]].itertuples(index=False, name=None)]

            image = cv2.imread(image_fp)
            rp_bbs = utils.selective_search(image, use_fast=False)

            positives = 0
            negatives = 0

            for x, y, w, h in rp_bbs:
                rp_bb = (x, y, x + w, y + h)
                roi_path = None
                roi_name = None
                for gt_bb in gt_bbs:
                    iou = utils.calculate_iou(rp_bb, gt_bb)

                    if iou >= config.IOU_POSITIVE and positives <= config.MAX_POSITIVE_SAMPLES:
                        roi_path = positive_data_path
                        roi_name = str(total_positives)
                        positives += 1
                        total_positives += 1
                    else:
                        # determine if the proposed bb falls within the ground truth bb
                        full_overlap = utils.is_full_overlap(rp_bb, gt_bb)
                        if not full_overlap and iou <= config.IOU_NEGATIVE and negatives <= config.MAX_NEGATIVE_SAMPLES:
                            roi_path = negative_data_path
                            roi_name = str(total_negatives)
                            negatives += 1
                            total_negatives += 1

                    if roi_path is not None and roi_name is not None:
                        roi = image[y:y + h, x:x + w]
                        roi = cv2.resize(roi, config.RCNN_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(utils.join_multiple_paths(roi_path, roi_name) + config.DATA_EXT, roi)

                if positives == config.MAX_POSITIVE_SAMPLES and negatives == config.MAX_NEGATIVE_SAMPLES:
                    break

            print(f"Time elapsed: {(time.time() - start_time)}s.")

    return total_positives, total_negatives


def main(args: argparse.Namespace) -> None:
    start_time = time.time()

    base_path, data_path = _validate_input_arguments(args)
    positive_data_path = _create_dir(data_path, config.POSITIVE_DATA_PATH)
    negative_data_path = _create_dir(data_path, config.NEGATIVE_DATA_PATH)

    total_positives, total_negatives = _process_images(base_path, positive_data_path, negative_data_path)

    print(f"Total positives: {total_positives}")
    print(f"Total negatives: {total_negatives}")
    print(f"Time elapsed: {(time.time() - start_time) / 60}m.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=str, help="Existing directory for the raw images.")
    parser.add_argument("data_path", type=str, help="(Existing) directory for the training dataset.")
    main(parser.parse_args())
