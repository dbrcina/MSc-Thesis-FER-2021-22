IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

TRAIN_PATH = "train"
VAL_PATH = "val"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

ANNOTATION_EXT = ".csv"
DATA_EXT = ".jpg"

IOU_POSITIVE = 0.7
IOU_NEGATIVE = 0.2

MAX_POSITIVE_SAMPLES = 30
MAX_NEGATIVE_SAMPLES = 30

RCNN_INPUT_DIM = (227, 227)
OCR_INPUT_DIM = (28, 28)

RANDOM_SEED = 42

TRAIN_VAL_SPLIT = 0.2

MAX_INFERENCE_SAMPLES = 2000

CLAHE_CLIP_LIMIT = 5
CLAHE_TILE_GRID_SIZE = (50, 20)

BILATERAL_D = 10
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 4
