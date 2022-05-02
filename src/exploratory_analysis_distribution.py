import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


# https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
def main(args: argparse.Namespace) -> None:
    path = args.path

    print(f"Reading images from '{path}' into memory and calculating mean and std:")

    dataset = ImageFolder(path, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=1024)

    channels_mean_sum = 0
    channels_mean_sq_sum = 0
    n = 0

    for x, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_mean_sum += torch.mean(x, dim=(0, 2, 3))
        channels_mean_sq_sum += torch.mean(x ** 2, dim=(0, 2, 3))
        n += 1

    mean = channels_mean_sum / n
    std = (channels_mean_sq_sum / n - mean ** 2) ** 0.5

    print(f"MEAN: {mean}")
    print(f"STD: {std}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to data.")
    main(parser.parse_args())
