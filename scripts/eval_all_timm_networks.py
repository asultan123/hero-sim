from frontend import *
import config
import frontend as front
from timeit import default_timer as timer

arch_config = {
    "filter_count": 32,
    "channel_count": 18,
    "directly_supported_kernels": [(1, 1), (3, 3)],
    "ifmap_mem_ub": 2**20 // 18 * 18,
    "allow_ifmap_distribution": True,
    "ofmap_mem_ub": 2**20,
    "allow_ofmap_distribution": True,
}

front.config.RESULTS_CSV_PATH = "../data/timm.csv"

with open("../data/timm_lib_testcases.pickle", "rb") as file:
    layer_name_tracker = pickle.load(file)

test_case_list = list(layer_name_tracker.keys())
test_case_list = sorted(test_case_list)

print(len(test_case_list))
# start = timer()
# result_df = launch_workers_with_test_cases(test_case_list, layer_name_tracker)
# end = timer()
# print(end - start)
