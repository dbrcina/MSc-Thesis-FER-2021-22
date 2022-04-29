import argparse
import string

import cv2
import torch

import config
from datasets import VAL_TRANSFORM_OD, VAL_TRANSFORM_OCR
from models import ALPRLightningModule
from PIL import Image
import time

def main(args: argparse.Namespace) -> None:
    start_time = time.time()
    od_model = ALPRLightningModule.load_from_checkpoint(args.od_model_path)
    ocr_model = ALPRLightningModule.load_from_checkpoint(args.ocr_model_path)

    chars = list(string.digits + string.ascii_uppercase)

    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    # x = VAL_TRANSFORM_OCR(Image.fromarray(image))
    # x: torch.Tensor = x[None, :, :, :]
    # prob = ocr_model.predict(x)
    # i = torch.argmax(prob, dim=1)
    # print(chars[i])
    # return

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rp_bbs = ss.process()

    lp_roi = None
    lp_prob = 0

    for x, y, w, h in rp_bbs[:config.MAX_INFERENCE_SAMPLES]:
        roi = image[y:y + h, x:x + w]
        roi = cv2.resize(roi, config.RCNN_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
        roi = VAL_TRANSFORM_OD(roi)
        roi = roi[None, :, :, :]
        prob = od_model.predict(roi)
        if prob > 0.5:
            if prob > lp_prob:
                lp_prob = prob
                lp_roi = (x, y, w, h)

    if lp_roi is None:
        print("Didn't manage to find license plate.")
        return

    x, y, w, h = lp_roi
    print(lp_prob)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print(time.time()-start_time)
    cv2.imshow("Test", image)
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("od_model_path", type=str, help="Path to object detection model.")
    parser.add_argument("ocr_model_path", type=str, help="Path to OCR model.")
    parser.add_argument("image_path", type=str, help="Path to test image.")
    main(parser.parse_args())
