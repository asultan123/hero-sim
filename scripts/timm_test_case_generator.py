from frontend import *

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
        
arch_config = {
    "filter_count": 32,
    "channel_count": 18,
    "directly_supported_kernels": [(1, 1), (3, 3)],
    "ifmap_mem_ub": 2**20 // 18 * 18,
    "allow_ifmap_distribution": True,
    "ofmap_mem_ub": 2**20,
    "allow_ofmap_distribution": True,
}

layer_name_tracker = {}
for model_name, layer_dims in tqdm(layer_dims_generator()):
    (
        test_case_list,
        layer_name_tracker,
    ) = convert_collected_model_layers_to_testcases(
        layer_dims, arch_config, model_name, layer_name_tracker
    )
    
    
with open("../data/timm_lib_testcases_1.pickle", "wb") as file:
    pickle.dump(layer_name_tracker, file)