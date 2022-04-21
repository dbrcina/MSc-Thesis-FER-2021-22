import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from PIL import ImageTk, Image

import config
import utils


class ImageCanvas(tk.Canvas):
    def __init__(self, container) -> None:
        super().__init__(container, width=config.IMG_WIDTH, height=config.IMG_HEIGHT)

        self._reset_params()
        self._register_events()

    def _reset_params(self) -> None:
        self.img = None
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.rect_id = 0

    def _register_events(self) -> None:
        self.bind("<Button-1>", self._handle_mouse_pressed)
        self.bind("<Button1-Motion>", self._handle_mouse_motion)

    def _handle_mouse_pressed(self, event: tk.Event) -> None:
        if self.img is not None:
            self.x1, self.y1 = event.x, event.y

    def _handle_mouse_motion(self, event: tk.Event) -> None:
        if self.img is not None:
            self.x2, self.y2 = event.x, event.y
            # Update coordinates of rectangle based on rect_id
            self.coords(self.rect_id, self.x1, self.y1, self.x2, self.y2)

    def load_img(self, file: str) -> None:
        self._reset_params()
        self.img = ImageTk.PhotoImage(Image.open(file))
        self.create_image(0, 0, anchor=tk.NW, image=self.img)
        # Initially, no rectangle is created because coordinates are set to 0
        self.rect_id = self.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline="red")

    def get_bbox(self) -> tuple[int, int, int, int]:
        topx, botx = sorted((self.x1, self.x2))
        topy, boty = sorted((self.y1, self.y2))
        return topx, topy, botx, boty


class ControlFrame(ttk.Frame):
    def __init__(self, container, image_canvas: ImageCanvas) -> None:
        super().__init__(container)
        self.image_canvas = image_canvas
        self.directory = None
        self.filenames = []
        self.current_index = 0
        self.lp = tk.StringVar()

        self._create_widgets()

    def _create_widgets(self) -> None:
        ttk.Button(self, text="Select directory", command=self._select_dir).grid(column=0, row=0, columnspan=2)

        # add for space...
        ttk.Label(self).grid(column=0, row=1)

        ttk.Label(self, text="Directory:", font="Helvetica 10 bold").grid(column=0, row=2, sticky=tk.E)
        self.lbl_dir = ttk.Label(self, text="No selected directory", font="Helvetica 8")
        self.lbl_dir.grid(column=1, row=2, padx=5, pady=5)

        ttk.Label(self).grid(column=0, row=3)

        ttk.Label(self, text="License Plate:", font="Helvetica 10 bold").grid(column=0, row=4, sticky=tk.E)
        self.entry_lp = ttk.Entry(self, textvariable=self.lp, font="Helvetica 8", state=tk.DISABLED)
        self.entry_lp.grid(column=1, row=4, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(self).grid(column=0, row=5)

        btn_frame = ttk.Frame(self)
        btn_frame.grid(column=0, row=6, columnspan=2)
        self.btn_prev = ttk.Button(btn_frame, text="Previous", command=lambda: self._new_img("Left"), state=tk.DISABLED)
        self.btn_prev.grid(column=0, row=0, sticky=tk.E)
        self.btn_next = ttk.Button(btn_frame, text="Next", command=lambda: self._new_img("Right"), state=tk.DISABLED)
        self.btn_next.grid(column=1, row=0, sticky=tk.W)

        ttk.Label(self).grid(column=0, row=7)

        self.btn_save = ttk.Button(self, text="Save", command=self._save_annot, state=tk.DISABLED)
        self.btn_save.grid(column=0, row=8, columnspan=2)

        self.lp.trace("w", lambda name, index, mode, lp=self.lp: self.btn_save.config(
            state=tk.NORMAL) if lp.get() != "" else self.btn_save.config(state=tk.DISABLED))

    def _select_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select directory")
        if directory == "":
            return

        self.directory = directory
        self.image_canvas.delete("all")
        self.lbl_dir["text"] = directory
        self.entry_lp.delete(0, tk.END)
        self.current_index = 0

        self.btn_prev["state"] = tk.DISABLED
        self.btn_save["state"] = tk.DISABLED

        self._get_image_filenames()
        if len(self.filenames) == 0:
            self.btn_next["state"] = tk.DISABLED
            self.entry_lp["state"] = tk.DISABLED
            return

        self.btn_next["state"] = tk.NORMAL
        self.entry_lp["state"] = tk.NORMAL

        self._display_current_img()

    def _get_image_filenames(self) -> None:
        if self.directory is not None:
            self.filenames = [f for f in os.listdir(self.directory) if f.lower().endswith(config.IMG_EXTENSIONS)]

    def _display_current_img(self) -> None:
        assert self.current_index < len(self.filenames)

        self.entry_lp.delete(0, tk.END)
        self.image_canvas.load_img(self._get_current_img_full_filename())

    def _get_current_img_full_filename(self) -> str:
        assert self.current_index < len(self.filenames)

        return utils.join_multiple_paths(self.directory, self.filenames[self.current_index])

    def _new_img(self, keysym: str) -> None:
        assert len(self.filenames) != 0, "This shouldn't happen because buttons needs to be disabled if len is 0"

        if keysym == "Right":
            self.current_index = min(self.current_index + 1, len(self.filenames) - 1)
            self.btn_prev["state"] = tk.NORMAL
            if self.current_index == len(self.filenames) - 1:
                self.btn_next["state"] = tk.DISABLED
        elif keysym == "Left":
            self.current_index = max(self.current_index - 1, 0)
            self.btn_next["state"] = tk.NORMAL
            if self.current_index == 0:
                self.btn_prev["state"] = tk.DISABLED
        else:
            return

        self._display_current_img()

    def _save_annot(self) -> None:
        lp = self.lp.get()
        if lp == "":
            return

        bbox = self.image_canvas.get_bbox()
        if bbox == (0, 0, 0, 0):
            messagebox.showinfo("Information", "Missing bounding box")
            return

        image_filename = self._get_current_img_full_filename()
        annot_filename = utils.replace_file_extension(image_filename, config.ANNOTATION_EXT)
        annot = {
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
            "lp": lp
        }
        pd.DataFrame([annot]).to_csv(annot_filename)

        messagebox.showinfo("Information", f"Saved successfully to '{annot_filename}'")

    def handle_control_key(self, keysym: str) -> None:
        if keysym == "s":
            self._save_annot()


class AnnotatorApp(tk.Tk):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title(title)
        self.resizable(False, False)

        self._create_widgets()
        self._register_events()

    def _create_widgets(self) -> None:
        self.image_canvas = ImageCanvas(self)
        self.image_canvas.grid(column=0, row=0)

        ttk.Separator(self, orient=tk.VERTICAL).grid(column=1, row=0, sticky=tk.NS)

        self.control_frame = ControlFrame(self, self.image_canvas)
        self.control_frame.grid(column=2, row=0, padx=10)

    def _register_events(self) -> None:
        self.bind("<Control-s>", lambda event: self.control_frame.handle_control_key(event.keysym))


if __name__ == "__main__":
    AnnotatorApp("License Plate Annotator").mainloop()
