from frontend import *
import frontend as front
from timeit import default_timer as timer
from config import arch_config

front.config.RESULTS_CSV_PATH = "../data/timm_1mb.csv"

def layer_dims_generator():
    processed_models_files = os.listdir("../data/processed_models")
    processed_models_names = [
        "".join(filename.split(".")[:-2]) for filename in processed_models_files
    ]
    processed_models_filepaths = [
        os.path.join("../data/processed_models", filename)
        for filename in os.listdir("../data/processed_models")
    ]

    ignore_list = pickle.load(open("../data/frontend_ignore_list.pickle", "rb"))

    for model_name, path in tqdm(
        list(zip(processed_models_names, processed_models_filepaths))
    ):
        if model_name in ignore_list:
            continue

        with open(path, "rb") as file:
            layer_dims = pickle.load(file)
        yield model_name, layer_dims

try:
    print('Trying to load cached layers')
    with open("../data/timm_oifmap_1mb.pickle", "rb") as file:
        layer_name_tracker = pickle.load(file)
        
except Exception as e:
    print(e)
    layer_name_tracker = {}
    for model_name, layer_dims in tqdm(layer_dims_generator()):
        (
            test_case_list,
            layer_name_tracker,
        ) = convert_collected_model_layers_to_testcases(
            layer_dims, arch_config, model_name, layer_name_tracker
        )
        
    with open(f"../data/timm_oifmap_1mb.pickle", "wb") as file:
        pickle.dump(layer_name_tracker, file)


test_case_list = list(layer_name_tracker.keys())
test_case_list = sorted(test_case_list)

print(len(test_case_list))
start = timer()
result_df = launch_workers_with_test_cases(test_case_list, layer_name_tracker)
end = timer()
print(f"Processed all timm networks in {end - start :.2f} seconds")
result_df.to_csv(front.config.RESULTS_CSV_PATH, index=False)
