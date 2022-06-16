# import pandas as pd
# import seaborn as se
from audioop import avg
from lib2to3.pytree import convert
import pickle
from unittest import skip

# from collections import Counter
import numpy as np
import os
from pandas import DataFrame
import torch
import pickle
import timm
from ModelAnalysis import load_model_from_timm, load_default_input_tensor_for_model
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.autograd.profiler_util import FunctionEventAvg
from tqdm.autonotebook import tqdm as notebook_tqdm
from timeit import timeit
from tqdm import tqdm
from ModelAnalysis import ModelProfiler
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
import pandas as pd

# import torchvision
import json
from timeit import default_timer as timer
from pathlib import Path
import shutil


def profile_model_forward(model, default_input, skip_first=5, repeat=20):
    runtimes_in_usec = []
    model.eval()

    with torch.no_grad():
        for _ in range(skip_first + repeat):
            start = timer()
            model(default_input)
            end = timer()
            runtimes_in_usec.append((end - start) * 10**6)
    return np.average(runtimes_in_usec[skip_first:])


def profile_model_operations(
    model,
    default_input,
    skip_first=5,
    wait=2,
    warmup=2,
    active=1,
    repeat=20,
    ops_instance_map={"aten::conv2d": Conv2d, "aten::linear": Linear},
    event_trace_basepath="./result",
    collect_output_dims=True,
):
    ops_names_to_log = list(ops_instance_map.keys())

    event_trace_path = Path(event_trace_basepath)

    if event_trace_path.exists() and event_trace_path.is_dir():
        shutil.rmtree(event_trace_path, ignore_errors=True)

    os.makedirs(event_trace_path)

    def trace_handler(p):
        event_trace_file_path = os.path.join(
            event_trace_path, f"event_trace_{p.step_num}.json"
        )
        p.export_chrome_trace(event_trace_file_path)

    model = model.eval()

    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU],
            on_trace_ready=trace_handler,
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./result/tensorboard_trace.json'),
            record_shapes=True,
            with_modules=True,
            with_stack=True,
            schedule=torch.profiler.schedule(
                skip_first=skip_first,
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat,
            ),
        ) as p:
            for _ in range(skip_first + (wait + warmup + active) * repeat):
                model(default_input)
                p.step()

    supported_ops_metrics = parse_trace_files(
        event_trace_path, repeat, ops_names_to_log
    )
    ops_metrics = aggregate_ops_metrics(
        supported_ops_metrics, ops_instance_map, collect_output_dims
    )
    return ops_metrics


def parse_trace_files(event_trace_path, repeat, ops_names_to_log):
    event_trace_file_names = os.listdir(event_trace_path)
    assert len(event_trace_file_names) == repeat
    event_trace_filepaths = [
        os.path.join(event_trace_path, filename) for filename in event_trace_file_names
    ]
    supported_ops_metrics = [list() for _ in range(len(event_trace_filepaths))]
    for file_idx, filepath in enumerate(event_trace_filepaths):
        event_trace_dict = json.load(open(filepath))
        for event in event_trace_dict["traceEvents"]:
            if event["name"] in ops_names_to_log:
                supported_ops_metrics[file_idx].append(
                    {
                        "dur": event["dur"],
                        "input_dims": event["args"]["Input Dims"],
                        "instance": event["name"],
                    }
                )

    event_list_lengths = [len(event_list) for event_list in supported_ops_metrics]
    assert len(set(event_list_lengths)) == 1

    transposed_supported_ops_metrics = [list(i) for i in zip(*supported_ops_metrics)]

    return transposed_supported_ops_metrics


def aggregate_ops_metrics(supported_ops_metrics, ops_instance_map, collect_output_dims):
    ops_metrics = []
    for op_events in supported_ops_metrics:

        for first, second in zip(op_events[:-1], op_events[1:]):
            assert first["input_dims"] == second["input_dims"]

        ops_metrics.append(
            {
                "input_dims": first["input_dims"][: (2 if collect_output_dims else 1)],
                "instance": ops_instance_map[first["instance"]],
                "dur": np.average([op["dur"] for op in op_events]),
            }
        )
    return ops_metrics


def get_model_name_from_filename(filename: str):
    return "".join(filename.split(".")[:-2])


def convert_fmap_dim_to_list(fmap_dim):
    return [fmap_dim.channels, fmap_dim.height, fmap_dim.width]


def append_layer_duration_to_processed_model(
    avg_forward_pass_duration,
    ops_metrics,
    processed_model,
):
    assert len(processed_model) == len(ops_metrics)
    model_layer_durations = {}
    model_layer_durations["forward"] = avg_forward_pass_duration
    modified_processed_model = {}
    for (layer_name, (input_dims, layer)), op in zip(
        processed_model.items(), ops_metrics
    ):
        if isinstance(layer, op["instance"]):
            modified_processed_model[layer_name] = (input_dims, layer, op["dur"])
        else:
            raise Exception("Mismatch between ops and processed model layers")

    return modified_processed_model


def aggregate_profiling_info(cpu_profiling_basepath):
    aggregate_profiling_results = {}
    for model_name in tqdm(os.listdir(cpu_profiling_basepath)):
        profile_path = os.path.join(cpu_profiling_basepath, model_name)
        profile_results = (
            pd.read_csv(profile_path, index_col=False)
            .iloc[:, 1:]
            .set_index("Layer Name")
            .to_dict(orient="index")
        )
        supported_layers_forward_duration = 0
        sanitized_model_name = ".".join(model_name.split(".")[:-1])
        for layer_name, result in profile_results.items():
            if layer_name != "forward":
                supported_layers_forward_duration += result["Duration"]
            aggregate_profiling_results[(sanitized_model_name, layer_name)] = result
        aggregate_profiling_results[(sanitized_model_name, "supported_forward")] = {
            "Duration": supported_layers_forward_duration
        }
    pd.DataFrame.from_dict(aggregate_profiling_results, orient="index").to_csv(
        "../data/cpu_model_profiling.csv"
    )
    return aggregate_profiling_results


if __name__ == "__main__":

    torch.set_num_threads(16)

    processed_model_basepath = "../data/processed_models"
    processed_model_list = os.listdir(processed_model_basepath)
    ignore_list = pickle.load(open("../data/frontend_ignore_list.pickle", "rb"))

    profiling_results_basepath = Path("../data/profiling")

    if profiling_results_basepath.exists() and profiling_results_basepath.is_dir():
        shutil.rmtree(profiling_results_basepath, ignore_errors=True)

    os.makedirs(profiling_results_basepath)

    repeat = 20

    for model_name in tqdm(processed_model_list):

        model_name = get_model_name_from_filename(model_name)

        if model_name in ignore_list:
            continue

        model = load_model_from_timm(model_name)

        default_input = load_default_input_tensor_for_model(model)

        print(f"Profiling {model_name} ")

        res = ModelProfiler.profile_layers(
            model, default_input, wait=0, warmup=0, repeat=repeat
        )

        all_layers = sum(
            [
                duration
                for layer_durations in res.values()
                for duration in layer_durations.values()
            ]
        )
        supported = sum([duration for duration in res["supported"].values()])

        res["supported"]["forward"] = all_layers
        res = {
            "Layer Name": list(res["supported"].keys()),
            "Duration": list(res["supported"].values()),
        }

        print(f"Percent of Compute {supported / all_layers *100 :.2f}")
        DataFrame.from_dict(res).to_csv(
            os.path.join(profiling_results_basepath, f"{model_name}.csv")
        )

    cpu_profiling_basepath = Path("../data/profiling")
    aggregate_profiling_results = aggregate_profiling_info(cpu_profiling_basepath)
