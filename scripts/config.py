import colorlog
from colorlog import ColoredFormatter

CORE_COUNT = 32
TEST_CASE_COUNT = 5000
SAVE_EVERY = 1
SEED = 1234
LAYER_SIZE_UB = 2**5
IFMAP_LOWER = 1
IFMAP_UPPER = 64
LOG2_FILTER_LOWER = 0
LOG2_FILTER_UPPER = 10
LOG2_CHANNEL_LOWER = 0
LOG2_CHANNEL_UPPER = 10
DIRECTLY_SUPPORTED_KERNELS = [(1, 1), (3, 3)]
SUBPROCESS_OUTPUT_DIR = "../data/subprocess_output"

ARCH_CONFIG_DICT = {
    "small": {"filter_count": 9, "channel_count": 9},
    "medium": {"filter_count": 18, "channel_count": 18},
    "large": {"filter_count": 32, "channel_count": 18},
}


formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(log_color)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG": "green",
        "INFO": "yellow",
        "WARNING": "orange",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)

logger = colorlog.getLogger()

logger.addHandler(handler)

logger.setLevel("DEBUG")