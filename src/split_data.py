import argparse
import os
import random

import config
import utils


def _split(data_path: str, path: str) -> None:
    print(f"Splitting '{path}'...")
    filenames = next(iter(os.walk(utils.join_multiple_paths(data_path, path))))[2]
    filenames = list(filter(lambda x: x.endswith(config.IMG_EXTENSIONS), filenames))

    train = set(random.choices(filenames, k=int(0.8 * len(filenames))))
    val = set(filenames) - train

    print(f"TRAIN: {len(train)}")
    train_path = utils.join_multiple_paths(data_path, config.TRAIN_DATA_PATH, path)
    print(f"Saving to '{train_path}'...")
    os.makedirs(train_path)

    print(f"VAL: {len(val)}")
    val_path = utils.join_multiple_paths(data_path, config.VAL_DATA_PATH, path)
    print(f"Saving to '{val_path}'...")
    os.makedirs(val_path)


def main(args: argparse.Namespace) -> None:
    data_path = args.data_path
    if not os.path.isdir(args.data_path):
        print(f"Provided '{data_path}' is not a directory!")
        return

    for path in os.listdir(data_path):
        if path not in (config.POSITIVE_DATA_PATH, config.NEGATIVE_DATA_PATH):
            continue
        _split(data_path, path)

    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Directory for the training dataset.")
    main(parser.parse_args())
