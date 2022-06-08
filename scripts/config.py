import colorlog
from colorlog import ColoredFormatter
from math import sqrt

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

energy_model = {
    # hack to overestimate MACS
    'mac': lambda mac_count: mac_count * (5 * 10**-12) ** 2,
    'sram': lambda access_count, sram_size, precision_bits: ((50+0.022*sqrt(precision_bits * sram_size)) * 10**-15) * access_count,
    'dram': lambda access_count, precision_bits: (20*10**-12)*(precision_bits)*access_count
}

arch_config = {
    "filter_count": 32,
    "channel_count": 18,
    "directly_supported_kernels": [(1, 1), (3, 3)],
    "ifmap_mem_ub": 2**20 // 18 * 18,
    "allow_ifmap_distribution": True,
    "ofmap_mem_ub": 2**20,
    "allow_ofmap_distribution": True,
    "reuse_chain_bank_size": 512,
    "weight_bank_size": 16
}


formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(log_color)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG": "green",
        "INFO": "yellow",
        "WARNING": "red",
        "ERROR": "red,bg_white",
        "CRITICAL": "black,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)

logger = colorlog.getLogger()

logger.addHandler(handler)

logger.setLevel("DEBUG")