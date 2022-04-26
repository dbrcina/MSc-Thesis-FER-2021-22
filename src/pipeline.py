import argparse

import cv2
import torch.nn

import config
from datasets import VAL_TRANSFORM_OD
from models import ALPRLightningModule


def main(args: argparse.Namespace) -> None:
    od_model = ALPRLightningModule.load_from_checkpoint(args.od_model_path)

    image = cv2.imread(r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\data_od\train\1\88.jpg")
    image = VAL_TRANSFORM_OD(image)
    image = image[None, :, :, :]
    out = od_model(image)
    print(torch.sigmoid(out*(-1)))
    return

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rp_bbs = ss.process()

    lp_roi = None
    lp_prob = 0

    for i, (x, y, w, h) in enumerate(rp_bbs):
        if i >= config.MAX_INFERENCE_SAMPLES:
            break

        roi = image[y:y + h, x:x + w]
        roi = cv2.resize(roi, config.RCNN_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
        roi = VAL_TRANSFORM_OD(roi)
        roi = roi[None, :, :, :]
        prob = od_model.predict(roi)
        if prob >= 0.5:
            if prob > lp_prob:
                lp_prob = prob
                lp_roi = (x, y, w, h)

    if lp_roi is None:
        print("Didn't manage to find license plate.")
        return

    x, y, w, h = lp_roi
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0))
    cv2.imshow("Test", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("od_model_path", type=str, help="Path to object detection model.")
    parser.add_argument("ocr_model_path", type=str, help="Path to OCR model.")
    parser.add_argument("image_path", type=str, help="Path to image.")
    main(parser.parse_args())
