import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import ImageTk, Image

import config
import utils
from pipeline import alpr_pipeline


class ControlFrame(ttk.Frame):
    def __init__(self, container, image_canvas: tk.Canvas, detector_path: str, recognizer_path: str) -> None:
        super().__init__(container)

        self.image_canvas = image_canvas
        self.detector = utils.load_model(detector_path)
        self.recognizer = utils.load_model(recognizer_path)

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
        if image.shape[:2] != (config.IMG_HEIGHT, config.IMG_WIDTH):
            image = cv2.resize(image, dsize=(config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image = image
        self._display_image(image)

        self.btn_start["state"] = "enabled"

    def _display_image(self, image: np.ndarray) -> None:
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(image))
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def _start(self) -> None:
        image_cpy = self.image.copy()

        result = alpr_pipeline(cv2.cvtColor(image_cpy, cv2.COLOR_RGB2BGR), self.detector, self.recognizer, True)
        if result is None:
            messagebox.showinfo("Information", "Didn't manage to find any license plate!")
            return

        (lp_x, lp_y, lp_w, lp_h), lp = result

        cv2.rectangle(image_cpy, (lp_x, lp_y), (lp_x + lp_w, lp_y + lp_h), (255, 0, 0), 2)
        (w, h), _ = cv2.getTextSize(lp, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(image_cpy, (lp_x, lp_y - 30), (lp_x + w, lp_y), (255, 0, 0), -1)
        cv2.putText(image_cpy, lp, (lp_x, lp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self._display_image(image_cpy)


class ALPRApp(tk.Tk):
    def __init__(self, detector_path: str, recognizer_path: str) -> None:
        super().__init__()

        self.detector_path = detector_path
        self.recognizer_path = recognizer_path
        self.title("Automatic License Plate Recognizer")
        self.resizable(False, False)

        self._create_widgets()

    def _create_widgets(self) -> None:
        image_canvas = tk.Canvas(self, width=config.IMG_WIDTH, height=config.IMG_HEIGHT)
        image_canvas.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill="x")

        control_frame = ControlFrame(self, image_canvas, self.detector_path, self.recognizer_path)
        control_frame.pack()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic License Plate Recognizer")
    parser.add_argument("detector_path", type=str)
    parser.add_argument("recognizer_path", type=str)
    args = vars(parser.parse_args())

    ALPRApp(args["detector_path"], args["recognizer_path"]).mainloop()
