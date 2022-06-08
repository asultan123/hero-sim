from frontend import *
from config import arch_config

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
