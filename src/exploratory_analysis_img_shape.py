import argparse
import os

import pandas as pd

import config
import utils


def main(args: argparse.Namespace) -> None:
    path = args.path
    if not os.path.isdir(path):
        print(f"Provided '{path}' is not a directory!")
        exit(-1)

    df_all = None
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if not filename.endswith(config.ANNOTATION_EXT):
                continue

            df_current = pd.read_csv(utils.join_multiple_paths(dirpath, filename), index_col=0)
            if df_all is None:
                df_all = df_current
            else:
                df_all = pd.concat((df_all, df_current), ignore_index=True)

    df_all["width"] = df_all.x2 - df_all.x1
    df_all["height"] = df_all.y2 - df_all.y1

    df_width_height = df_all[["width", "height"]]

    print("MIN:")
    print(df_width_height.min())

    print("-------------")

    print("MAX:")
    print(df_width_height.max())

    print("-------------")

    print("MEAN:")
    print(df_width_height.mean())

    print("-------------")

    print("MEDIAN")
    print(df_width_height.median())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to annotated images.")
    main(parser.parse_args())
