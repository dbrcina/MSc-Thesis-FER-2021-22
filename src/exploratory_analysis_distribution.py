import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

import config


def main(args: argparse.Namespace) -> None:
    path = args.path
    if not os.path.isdir(path):
        print(f"Provided '{path}' is not a directory!")
        exit(-1)

    print(f"Reading images from '{path}' into memory and calculating mean and std:")

    mean = torch.zeros(3)
    var = torch.zeros(3)
    n = 0
    for i, filename in enumerate(glob.glob(f"{path}/*/*{config.DATA_EXT}")):
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}):...")
        img = to_tensor(Image.open(filename))
        mean += img.mean()
        var += img.var(9)
        n += img.shape[1] * img.shape[2]

    return

    filenames = glob.glob(f"{path}/*/*{config.DATA_EXT}")
    data = np.array([np.array(Image.open(filename), dtype=float) for filename in filenames])

    # Normalize to [0,1]
    data /= 255.0

    # Calculate over the last dimension, which represents color channel...
    print(f"MEAN: {data.mean(axis=(0, 1, 2))}")
    print(f"STD: {data.std(axis=(0, 1, 2))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to data.")
    main(parser.parse_args())
