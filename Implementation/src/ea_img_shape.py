import argparse
import glob
from typing import Dict, Any

import pandas as pd
from tqdm.auto import tqdm


def main(args: Dict[str, Any]) -> None:
    path = args["path"]

    df_all = None
    for annot_path in tqdm(glob.glob(f"{path}/**/*.csv")):
        df_current = pd.read_csv(annot_path, index_col=0)
        if df_all is None:
            df_all = df_current
        else:
            df_all = pd.concat((df_all, df_current), ignore_index=True)

    df_all["width"] = df_all.x2 - df_all.x1
    df_all["height"] = df_all.y2 - df_all.y1

    df_width_height = df_all[["width", "height"]]
    df_ratio = df_width_height["width"] / df_width_height["height"]

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

    print("-------------")

    print("RATIO MIN:")
    print(df_ratio.min())

    print("-------------")

    print("RATIO MAX:")
    print(df_ratio.max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the raw annotated images")

    main(vars(parser.parse_args()))
