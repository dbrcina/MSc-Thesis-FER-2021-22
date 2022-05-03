import argparse
import string
import time
from operator import itemgetter
from typing import Dict, Any

import cv2
import torch
from PIL import Image

import config
from datasets import VAL_TRANSFORM_OD, VAL_TRANSFORM_OCR
from train import ALPRLightningModule
from transform import four_point_transform


def main(args: Dict[str, Any]) -> None:
    start_time = time.time()

    od_model = ALPRLightningModule.load_from_checkpoint(args["od_model_path"])
    ocr_model = ALPRLightningModule.load_from_checkpoint(args["ocr_model_path"])
    image_path = args["image_path"]

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image.shape[:2] != (config.IMG_HEIGHT, config.IMG_WIDTH):
        image = cv2.resize(image, dsize=(config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_image)
    clahe = cv2.createCLAHE(config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID_SIZE)
    clahe_y = clahe.apply(y)
    clahe_img = cv2.merge((clahe_y, cr, cb))
    updated_image = cv2.cvtColor(clahe_img, cv2.COLOR_YCrCb2BGR)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(updated_image)
    ss.switchToSelectiveSearchFast()
    rp_bbs = ss.process()

    rgb_image = cv2.cvtColor(updated_image, cv2.COLOR_BGR2RGB)
    lp_roi = None
    lp_prob = 0

    resized = list(
        map(lambda bb: cv2.resize(rgb_image[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]], config.RCNN_INPUT_DIM,
                                  interpolation=cv2.INTER_CUBIC),
            rp_bbs[:config.MAX_INFERENCE_SAMPLES])
    )
    x = torch.stack([VAL_TRANSFORM_OD(r) for r in resized])
    preds = od_model.predict(x)

    index = preds.argmax()
    lp_prob = preds[index]
    if lp_prob > 0.5:
        lp_roi = rp_bbs[index]

    if lp_roi is None:
        print("Didn't manage to find license plate.")
        return

    x, y, w, h = lp_roi
    lp = image[y:y + h, x:x + w]

    gray = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray,
                                    d=config.BILATERAL_D,
                                    sigmaColor=config.BILATERAL_SIGMA_COLOR,
                                    sigmaSpace=config.BILATERAL_SIGMA_SPACE)

    thresh = cv2.threshold(bilateral, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    erode = cv2.erode(thresh, (5,5))
    cv2.imshow("test", erode)
    cv2.waitKey()
    return

    cnts = cv2.findContours(thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]

    c = None

    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        approx = cv2.approxPolyDP(cnt, epsilon=cv2.arcLength(cnt, closed=True) * 0.02, closed=True)
        if len(approx) == 4:
            c = approx
            break

    transformed = four_point_transform(thresh, c.reshape(4, 2))
    morph = cv2.erode(transformed, (5, 5), iterations=1)

    cv2.imshow("test", morph)
    cv2.waitKey()
    return

    cnts = cv2.findContours(morph, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    cts = []
    for cnt in sorted(cnts, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        if abs(cv2.contourArea(cnt)) < 100 or w > h or h < 10:
            continue
        char_img = morph[y:y + h, x:x + w]
        char_img = cv2.resize(char_img, dsize=config.OCR_INPUT_DIM, interpolation=cv2.INTER_CUBIC)
        cts.append(char_img)

    x = torch.stack([VAL_TRANSFORM_OCR(Image.fromarray(s)) for s in cts])
    y = ocr_model.predict(x)
    CHARACTERS = {i: c for i, c in enumerate(list(string.digits + string.ascii_uppercase))}
    predicted_lp = "".join(itemgetter(*torch.argmax(y, dim=1).tolist())(CHARACTERS))

    x, y, w, h = lp_roi
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, predicted_lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    print(lp_prob)
    cv2.imshow("Test", image)
    print(f"Time elapsed: {time.time() - start_time}s")
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("od_model_path", type=str, help="Path to object detection model.")
    parser.add_argument("ocr_model_path", type=str, help="Path to OCR model.")
    parser.add_argument("image_path", type=str, help="Path to test image.")
    main(vars(parser.parse_args()))
