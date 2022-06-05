from ModelAnalysis import *


if __name__ == "__main__":
    model_name_list = timm.list_models(
        exclude_filters=["*_iabn", "swin_*", "tnt_*", "tresnet_*"], pretrained=False
    )

    processed_models_list = os.listdir("../data/processed_models")
    processed_models_list = [
        filename.split(".")[-3] for filename in processed_models_list
    ]

    for model_name in tqdm(model_name_list):
        if model_name in processed_models_list:
            continue
        model = load_model_from_timm(model_name)
        input = load_default_input_tensor_for_model(model)
        model_stats = ModelDimCollector.collect_layer_dims_from_model(model, input)
        with open(
            f"../data/processed_models/{model_name}.model.pickle", "wb"
        ) as model_file:
            pickle.dump(model_stats, model_file)
