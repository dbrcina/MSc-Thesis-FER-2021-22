import argparse
import time
from typing import Dict, Any

import cv2

import config
from datasets import VAL_TRANSFORM_OD
from train import ALPRLightningModule


def main(args: Dict[str, Any]) -> None:
    start_time = time.time()

    od_model = ALPRLightningModule.load_from_checkpoint(args["od_model_path"])
    ocr_model = ALPRLightningModule.load_from_checkpoint(args["ocr_model_path"])
    image_path = args["image_path"]
    #
    # chars = [c for c in list(string.digits + string.ascii_uppercase)]
    #
    # im = Image.open(r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\data_ocr\train\class_8\class_8_2.jpg").convert(
    #     "RGB")
    # t = VAL_TRANSFORM_OCR(im)
    # t = t[None, :, :, :]
    # preds = ocr_model.predict(t)
    # i = preds.argmax(dim=1)
    # print(preds[0][i])
    # print(chars[i])
    # return

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image.shape[:2] != (config.IMG_HEIGHT, config.IMG_WIDTH):
        image = cv2.resize(image, dsize=(config.IMG_HEIGHT, config.IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

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

    for x, y, w, h in rp_bbs[:config.MAX_INFERENCE_SAMPLES]:
        roi = rgb_image[y:y + h, x:x + w]
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
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
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
