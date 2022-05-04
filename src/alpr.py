import argparse
import string
import time
import tkinter as tk
from operator import itemgetter
from tkinter import ttk, filedialog
from typing import List

import cv2
import numpy as np
import torch
from PIL import ImageTk, Image

import config
from datasets import VAL_TRANSFORM_OD, VAL_TRANSFORM_OCR
from train import ALPRLightningModule


class ControlFrame(ttk.Frame):
    def __init__(self, container, image_canvas: tk.Canvas, od_path: str, ocr_path: str) -> None:
        super().__init__(container)

        self.image_canvas = image_canvas
        self.od_model = ALPRLightningModule.load_from_checkpoint(od_path)
        self.ocr_model = ALPRLightningModule.load_from_checkpoint(ocr_path)
        self.characters = {i: c for i, c in enumerate(list(string.digits + string.ascii_uppercase))}

        self.od_model.eval()
        self.ocr_model.eval()

        self._create_widgets()

    def _create_widgets(self) -> None:
        ttk.Button(self, text="Select image", command=self._select_image).grid(row=0, column=0, padx=5, pady=5)
        self.btn_start = ttk.Button(self, text="Start", command=self._start, state="disabled")
        self.btn_start.grid(row=0, column=1, padx=5, pady=5)

    def _select_image(self) -> None:
        image_path = filedialog.askopenfile(mode="r", filetypes=[("jpeg files", "*.jpg")])
        if image_path is None:
            return

        image = cv2.imread(image_path.name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (config.IMG_HEIGHT, config.IMG_WIDTH):
            image = cv2.resize(image, dsize=(config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

        self.image = image
        self._display_image(image)

        self.btn_start["state"] = "enabled"

    def _display_image(self, image: np.ndarray) -> None:
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(image))
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def _start(self) -> None:
        print("Starting...")
        start_time = time.time()

        image_cpy = self.image.copy()

        clahe_image = self._apply_clahe(image_cpy)

        ss_time = time.time()
        rp_bbs = self._apply_selective_search(clahe_image)
        print(f"SS: {time.time() - ss_time}s")

        # Prepare inputs for OD model
        prepare_od_time = time.time()
        od_input = torch.stack([
            VAL_TRANSFORM_OD(
                cv2.resize(clahe_image[y:y + h, x:x + w], config.RCNN_INPUT_DIM, interpolation=cv2.INTER_CUBIC))
            for x, y, w, h in rp_bbs])
        print(f"OD PREPR: {time.time() - prepare_od_time}s")

        # Predictions for license plate
        od_model_time = time.time()
        lp_preds = self.od_model.predict(od_input)
        lp_index = lp_preds.argmax()
        lp_x, lp_y, lp_w, lp_h = rp_bbs[lp_index]
        lp = clahe_image[lp_y:lp_y + lp_h, lp_x:lp_x + lp_w]
        print(f"OD: {time.time() - od_model_time}")

        find_cnts_time = time.time()
        lp_gray = cv2.cvtColor(lp, cv2.COLOR_RGBA2GRAY)
        lp_bilateral = cv2.bilateralFilter(lp_gray,
                                           config.BILATERAL_D,
                                           config.BILATERAL_SIGMA_COLOR,
                                           config.BILATERAL_SIGMA_SPACE)
        lp_thresh = cv2.threshold(lp_bilateral, 127, 255, cv2.THRESH_BINARY_INV)[1]
        lp_cnts = cv2.findContours(lp_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        chars = []
        for lp_cnt in sorted(lp_cnts, key=lambda c: cv2.boundingRect(c)[0]):
            x, y, w, h = cv2.boundingRect(lp_cnt)
            if abs(cv2.contourArea(lp_cnt)) < 100 or w > h or h < 10:
                continue
            chars.append(lp_thresh[y:y + h, x:x + w])
        print(f"FIND CNTS: {time.time() - find_cnts_time}s")

        # Prepare inputs for OCR model
        prepare_ocr_time = time.time()
        ocr_input = torch.stack([
            VAL_TRANSFORM_OCR(cv2.cvtColor(
                cv2.resize(c, dsize=config.OCR_INPUT_DIM, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB))
            for c in chars])
        print(f"OCR PREP: {time.time() - prepare_ocr_time}s")

        # Predictions for characters
        ocr_model_time = time.time()
        char_preds = self.ocr_model.predict(ocr_input)
        print(f"OCR: {time.time() - ocr_model_time}s")

        print(f"TOTAL ELAPSED: {time.time() - start_time}s")

        predicted_lp = "".join(itemgetter(*torch.argmax(char_preds, dim=1).tolist())(self.characters))

        cv2.rectangle(image_cpy, (lp_x, lp_y), (lp_x + lp_w, lp_y + lp_h), (255, 0, 0))
        (w, h), _ = cv2.getTextSize(predicted_lp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image_cpy, (lp_x, lp_y - 20), (lp_x + w, lp_y), (255, 0, 0), -1)
        cv2.putText(image_cpy, predicted_lp, (lp_x, lp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        self._display_image(image_cpy)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb_image)
        clahe = cv2.createCLAHE(config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID_SIZE)
        return cv2.cvtColor(cv2.merge((clahe.apply(y), cr, cb)), cv2.COLOR_YCrCb2RGB)

    def _apply_selective_search(self, image: np.ndarray) -> List[np.ndarray]:
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        return ss.process()[:config.MAX_INFERENCE_SAMPLES]


class ALPRApp(tk.Tk):
    def __init__(self, od_path: str, ocr_path: str) -> None:
        super().__init__()

        self.od_path = od_path
        self.ocr_path = ocr_path
        self.title("Automatic License Plate Recognizer")
        self.resizable(False, False)

        self._create_widgets()

    def _create_widgets(self) -> None:
        image_canvas = tk.Canvas(self, width=config.IMG_WIDTH, height=config.IMG_HEIGHT)
        image_canvas.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill="x")

        control_frame = ControlFrame(self, image_canvas, self.od_path, self.ocr_path)
        control_frame.pack()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("od_path", type=str, help="Path to object detection model.")
    parser.add_argument("ocr_path", type=str, help="Path to ocr model.")
    args = vars(parser.parse_args())

    ALPRApp(args["od_path"], args["ocr_path"]).mainloop()
