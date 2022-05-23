from frontend import *

arch_config = {
    "filter_count": 32,
    "channel_count": 18,
    "directly_supported_kernels": [(1, 1), (3, 3)],
    "ifmap_mem_ub": 2**20 // 18 * 18,
    "allow_ifmap_distribution": True,
    "ofmap_mem_ub": 2**20,
    "allow_ofmap_distribution": True,
}

RESULTS_CSV_PATH = "../data/timm_0_5314.csv"

with open("../data/timm_lib_testcases.pickle", "rb") as file:
    layer_name_tracker = pickle.load(file)

test_case_list = list(layer_name_tracker.keys())
test_case_list = sorted(test_case_list)
result_df = launch_workers_with_test_cases(test_case_list, layer_name_tracker)
