import argparse

import cv2

import config


def main(args: argparse.Namespace) -> None:
    image_path = args.image_path

    input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ycrcb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_image)

    equ = cv2.equalizeHist(y)
    hist_eq_input_image = cv2.cvtColor(cv2.merge((equ, cr, cb)), cv2.COLOR_YCrCb2BGR)

    clahe = cv2.createCLAHE(config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID_SIZE)
    clahe_img = clahe.apply(y)
    clahe_input_image = cv2.cvtColor(cv2.merge((clahe_img, cr, cb)), cv2.COLOR_YCrCb2BGR)

    cv2.imshow("input", input_image)
    cv2.imshow("equ", hist_eq_input_image)
    cv2.imshow("clahe", clahe_input_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to a image.")
    main(parser.parse_args())
