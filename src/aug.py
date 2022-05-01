import argparse

from PIL import Image
from torchvision.transforms import RandomRotation, InterpolationMode

import config


def main(args: argparse.Namespace) -> None:
    image_path = args.image_path

    image = Image.open(image_path).convert("RGB")
    transform = RandomRotation(degrees=config.ROTATION_ANGLE, interpolation=InterpolationMode.BILINEAR, expand=True)
    image_aug = transform(image)
    image_aug.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to a image.")
    main(parser.parse_args())
