import argparse
import glob
import os

import numpy as np
from PIL import Image

import config


def main(args: argparse.Namespace) -> None:
    path = args.path
    if not os.path.isdir(path):
        print(f"Provided '{path}' is not a directory!")
        exit(-1)

    print(f"Reading images from '{path}' into memory and calculating mean and std:")

    filenames = glob.glob(f"{path}/*/*{config.DATA_EXT}")
    data = np.array([np.array(Image.open(filename)) for filename in filenames])
    print(f"MEAN: {data.mean(axis=(0, 1, 2))}")
    print(f"STD: {data.std(axis=(0, 1, 2))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to train data.")
    main(parser.parse_args())
