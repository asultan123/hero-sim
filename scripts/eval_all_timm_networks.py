from frontend import *
import config
import frontend as front
from timeit import default_timer as timer

front.config.RESULTS_CSV_PATH = "../data/timm_1mb.csv"

if __name__ == "__main__":
    
    with open("../data/timm_oifmap_1mb.pickle", "rb") as file:
        layer_name_tracker = pickle.load(file)

    test_case_list = list(layer_name_tracker.keys())
    test_case_list = sorted(test_case_list)

    print(len(test_case_list))
    start = timer()
    result_df = launch_workers_with_test_cases(test_case_list, layer_name_tracker)
    end = timer()
    print(f"Processed all timm networks in {end - start :.2f} seconds")
    result_df.to_csv(front.config.RESULTS_CSV_PATH, index=False)
    