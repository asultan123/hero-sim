import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import os
import multiprocessing


def get_layer_names(model):
    model_layer_names_without_decomposition_indiciators = set()
    for name in model["layer_name"].to_list():
        name = r".".join(name.split(".")[:-2])
        model_layer_names_without_decomposition_indiciators.add(name)
    return list(model_layer_names_without_decomposition_indiciators)


def escape_periods(name: str):
    return name.replace(".", r"\.")


def aggregate_layers(model):
    aggregated_layers = {}
    layer_names = get_layer_names(model)
    for name in layer_names:
        sub_layers_df = model[
            model["layer_name"].str.match(f"{escape_periods(name)}.+")
        ]
        sub_layers_df = sub_layers_df.set_index("layer_name")
        avg_utils = sub_layers_df.loc[:, "pe_util"].mean(axis=0)
        metrics_dict = (
            sub_layers_df.loc[
                :,
                [
                    "dram_load",
                    "dram_store",
                    "ifmap",
                    "weight",
                    "psum",
                    "reuse_chain",
                    "latency",
                ],
            ]
            .sum(axis=0)
            .to_dict()
        )
        metrics_dict["pe_util"] = avg_utils
        aggregated_layers[name] = metrics_dict
    return aggregated_layers


def collect_model_results(model_name):
    result_df = pd.read_csv("../data/timm.csv")
    model = result_df.loc[result_df["model_name"] == model_name]
    aggregated_layers = aggregate_layers(model)
    csv_path = os.path.join(arch_results_basepath, f"{model_name}.csv")
    pd.DataFrame.from_dict(aggregated_layers, orient="index").to_csv(csv_path)


if __name__ == "__main__":

    arch_results_basepath = Path("../data/arch_results")

    if arch_results_basepath.exists() and arch_results_basepath.is_dir():
        shutil.rmtree(arch_results_basepath, ignore_errors=True)

    os.makedirs(arch_results_basepath)

    result_df = pd.read_csv("../data/timm.csv")
    model_name_list = result_df["model_name"].unique()
    with multiprocessing.Pool(len(os.sched_getaffinity(0))) as pool:
        for _ in tqdm(
            pool.imap_unordered(collect_model_results, model_name_list),
            total=len(model_name_list),
        ):
            pass
