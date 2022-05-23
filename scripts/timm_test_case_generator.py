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

layer_name_tracker = {}
for model_name, layer_dims in tqdm(layer_dims_generator()):
    (
        test_case_list,
        layer_name_tracker,
    ) = convert_collected_model_layers_to_testcases(
        layer_dims, arch_config, model_name, layer_name_tracker
    )
    
    
with open("../data/timm_lib_testcases.pickle", "wb") as file:
    pickle.dump(layer_name_tracker, file)