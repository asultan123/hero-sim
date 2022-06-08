# import pandas as pd
# import seaborn as se
from lib2to3.pytree import convert
import pickle
from unittest import skip

# from collections import Counter
import numpy as np
import os
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
from ModelAnalysis import ModelDimCollector

# import torchvision
import json
from timeit import default_timer as timer
from pathlib import Path
import shutil


def profile_model_forward(model, default_input, skip_first=5, run_count=20):
    runtimes_in_usec = []
    model.eval()
    
    print('\n')
    with torch.no_grad():
        for _ in tqdm(range(skip_first + run_count)):
            start = timer()
            model(default_input)
            end = timer()
            runtimes_in_usec.append((end - start) * 10**6)
    return runtimes_in_usec[skip_first:]


def profile_model_operations(
    model,
    default_input,
    skip_first=5,
    wait=2,
    warmup=2,
    active=1,
    repeat=20,
    trace_ops=["aten::conv2d", "aten::linear"],
    event_trace_basepath="./result",
):
    event_trace_path = Path(event_trace_basepath)

    if event_trace_path.exists() and event_trace_path.is_dir():
        shutil.rmtree(event_trace_path, ignore_errors=True)

    os.makedirs(event_trace_path)

    def trace_handler(p):
        event_trace_file_path = os.path.join(
            event_trace_path, f"event_trace_{p.step_num}.json"
        )
        p.export_chrome_trace(event_trace_file_path)

    print('\n')

    with profile(
        activities=[ProfilerActivity.CPU],
        on_trace_ready=trace_handler,
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
        for _ in tqdm(range(skip_first + (wait + warmup + active) * repeat)):
            model(default_input)
            p.step()

    event_trace_file_names = os.listdir(event_trace_path)
    assert len(event_trace_file_names) == repeat
    event_trace_filepaths = [
        os.path.join(event_trace_path, filename) for filename in event_trace_file_names
    ]
    supported_ops_metrics = [list() for _ in range(len(event_trace_filepaths))]
    for file_idx, filepath in enumerate(event_trace_filepaths):
        event_trace_dict = json.load(open(filepath))
        for event in event_trace_dict["traceEvents"]:
            if event["name"] in trace_ops:
                supported_ops_metrics[file_idx].append(
                    {"dur": event["dur"], "input_dims": event["args"]["Input Dims"]}
                )

    event_list_lengths = [len(event_list) for event_list in supported_ops_metrics]
    assert len(set(event_list_lengths)) == 1

    transposed_supported_ops_metrics = [list(i) for i in zip(*supported_ops_metrics)]
    return transposed_supported_ops_metrics


def get_model_name_from_filename(filename: str):
    return "".join(filename.split(".")[:-2])


def convert_fmap_dim_to_list(fmap_dim):
    return [fmap_dim.channels, fmap_dim.height, fmap_dim.width]


def aggregate_op_metrics_into_dict(
    forward_pass_durations, supported_ops_metrics, processed_model
):
    avg_forward_pass_duration = np.average(forward_pass_durations)
    assert len(processed_model) == len(supported_ops_metrics)
    model_layer_durations = {}
    model_layer_durations["forward"] = avg_forward_pass_duration
    for (layer_name, (ifmap_dim, _)), op_metrics in zip(
        processed_model.items(), supported_ops_metrics
    ):
        layer_ifmap_dim = convert_ifmap_dim_to_list(ifmap_dim)
        for metric in op_metrics:
            input_dims = metric["input_dims"][0][1:]  # remove batch dim
            assert input_dims == layer_ifmap_dim
        model_layer_durations[layer_name] = np.average(
            [metric["dur"] for metric in op_metrics]
        )
    return model_layer_durations
    print(".")


if __name__ == "__main__":
    processed_model_basepath = "../data/processed_models"
    processed_model_list = os.listdir(processed_model_basepath)
    ignore_list = pickle.load(open("../data/frontend_ignore_list.pickle", "rb"))
    
    for model_name in tqdm(processed_model_list[1:2]):
        
        if model_name in ignore_list:
            continue 
        
        model = load_model_from_timm(get_model_name_from_filename(model_name))
        
        default_input = load_default_input_tensor_for_model(model)
        processed_model = ModelDimCollector.collect_layer_dims_from_model(model, default_input)
        
        print(f'Profiling {model_name} forward pass duration')
        
        forward_pass_durations = profile_model_forward(model, default_input)
        
        print(f'Profiling {model_name} operation metrics')
        
        supported_ops_metrics = profile_model_operations(model, default_input)
        # aggregate_op_metrics_into_dict(
        #     forward_pass_durations, supported_ops_metrics, processed_model
        # )
