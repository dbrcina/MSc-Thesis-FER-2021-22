IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_EXTENSIONS = (".jpg",)

TRAIN_FOLDER = "train"
VAL_FOLDER = "val"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

ANNOTATION_EXT = ".csv"
DATA_EXT = ".jpg"

IOU_POSITIVE = 0.70
IOU_NEGATIVE = 0.05

MAX_POSITIVE_SAMPLES = 30
MAX_NEGATIVE_SAMPLES = 30

OD_INPUT_DIM = (227, 227)
OCR_INPUT_DIM = (28, 28)

RANDOM_SEED = 42

TRAIN_VAL_SPLIT = 0.2

MAX_INFERENCE_SAMPLES = 2000

CLAHE_CLIP_LIMIT = 3
CLAHE_TILE_GRID_SIZE = (15, 15)

BILATERAL_D = 25
BILATERAL_SIGMA_COLOR = 20
BILATERAL_SIGMA_SPACE = 7

LP_TOPK = 10

LP_WIDTH_MIN = 60
LP_WIDTH_MAX = 285
LP_HEIGHT_MIN = 20
LP_HEIGHT_MAX = 135

EVALUATOR_RESULTS_FILE = "evaluator_results.txt"
