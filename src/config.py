IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

POSITIVE_DATA_PATH = "license_plate"
NEGATIVE_DATA_PATH = "no_license_plate"

ANNOTATION_EXT = ".csv"
DATA_EXT = ".jpg"

IOU_POSITIVE = 0.7
IOU_NEGATIVE = 0.1

MAX_POSITIVE_SAMPLES = 30
MAX_NEGATIVE_SAMPLES = 20

RCNN_INPUT_DIM = (100, 100)

TRAIN_DATA_PATH = "train"
VAL_DATA_PATH = "val"

RANDOM_SEED = 42
