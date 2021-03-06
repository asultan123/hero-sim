from argparse import ArgumentError
import multiprocessing
from unittest import result
from click import Argument
from regex import E
import torch
from PIL import Image
from torchvision import transforms, models
from sys import version
import timm
from tqdm import tqdm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from functools import reduce
from math import sqrt, floor, ceil
from math import inf
import pandas as pd
import pickle
import seaborn as sns
from functools import reduce
from math import sqrt, floor, ceil
from sklearn.preprocessing import MinMaxScaler
from copy import copy, deepcopy
from functools import partial
from timeit import default_timer as timer
import os
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
from schema import LayerDimensions, IfmapLayerDimensions, TestCase
from typing import Dict, Tuple, List, Optional
from collections import deque

# get_ipython().run_line_magic('matplotlib', 'inline')
version


def factors(n):
    step = 2 if n % 2 else 1
    return set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(sqrt(n)) + 1, step) if n % i == 0),
        )
    )


def eval_util(
    effective_channel_tiling,
    effective_filter_tiling,
    channels,
    filters,
    current_k,
    adjusted_pe_count,
):
    util_baseline = (
        effective_channel_tiling
        * current_k
        * effective_filter_tiling
        / adjusted_pe_count
    )
    util_baseline_occurence = (floor(filters / effective_filter_tiling)) * (
        floor(channels / effective_channel_tiling)
    )

    if channels % effective_channel_tiling == 0:
        util_in_last_channel_iter = 1
        util_drop_due_to_channel_tiling_occurrence = 0
        # util_in_last_iter_of_both = 1
    else:
        util_in_last_channel_iter = (
            (channels % effective_channel_tiling)
            * current_k
            * effective_filter_tiling
            / adjusted_pe_count
        )
        util_drop_due_to_channel_tiling_occurrence = floor(
            filters / effective_filter_tiling
        )

        # util_in_last_iter_of_both = (channels%effective_channel_tiling)

    if filters % effective_filter_tiling == 0:
        util_in_last_filter_iter = 1
        util_drop_due_to_filter_tiling_occurrence = 0
        # util_in_last_iter_of_both *= 1
    else:
        util_in_last_filter_iter = (
            (filters % effective_filter_tiling)
            * effective_channel_tiling
            * current_k
            / adjusted_pe_count
        )
        util_drop_due_to_filter_tiling_occurrence = floor(
            channels / effective_channel_tiling
        )

        # util_in_last_iter_of_both *= ((filters%effective_filter_tiling))

    if channels % effective_channel_tiling > 0 and filters % effective_filter_tiling:
        util_in_last_iter_of_both = (
            (channels % effective_channel_tiling)
            * (filters % effective_filter_tiling)
            * current_k
            / adjusted_pe_count
        )
        util_drop_due_to_both_occurence = 1

    else:
        util_in_last_iter_of_both = 1
        util_drop_due_to_both_occurence = 0

    total_tiles = (
        util_drop_due_to_channel_tiling_occurrence
        + util_drop_due_to_filter_tiling_occurrence
        + util_drop_due_to_both_occurence
        + util_baseline_occurence
    )
    weighted_avg_util = (
        util_drop_due_to_channel_tiling_occurrence * util_in_last_channel_iter
        + util_drop_due_to_filter_tiling_occurrence * util_in_last_filter_iter
        + util_drop_due_to_both_occurence * util_in_last_iter_of_both
        + util_baseline_occurence * util_baseline
    ) / total_tiles
    return weighted_avg_util, total_tiles


def eval_latency(kernel_sizes_supported_in_direct_mode, layer, total_tiles):
    current_k = layer.kernel_size[0]
    if current_k in kernel_sizes_supported_in_direct_mode:
        ifmap_size = layer.input_size[2] * layer.input_size[3]
        latency_lowering_lifting = 0
    else:
        # convert to 1x1 conv with balanced lowering ifmap dims
        latency_lowering_lifting = (
            current_k * layer.output_size[1] * layer.output_size[2]
        )
        ifmap_size = layer.input_size[2] * layer.output_size[1]

    latency_compute = total_tiles * ifmap_size

    return latency_compute + 2 * latency_lowering_lifting


def eval_access_counts(
    ifmap_size,
    filters,
    channels,
    effective_filter_tiling,
    effective_channel_tiling,
    channel_tiling,
    filter_tiling,
):
    ifmap_access_count = (
        ifmap_size
        * (
            (channels // effective_channel_tiling) * effective_channel_tiling
            + channels % effective_channel_tiling
        )
        * ceil(filters / effective_filter_tiling)
    )

    ofmap_access_count = (
        2
        * ifmap_size
        * (
            (filters // effective_filter_tiling) * effective_filter_tiling
            + filters % effective_filter_tiling
        )
        * ceil(channels / effective_channel_tiling)
    )

    weight_access_count = ifmap_size * (
        (channel_tiling * filter_tiling)
        * (
            (channels // effective_channel_tiling)
            * (filters // effective_filter_tiling)
        )
        + (channel_tiling * effective_filter_tiling)
        * ((channels // effective_channel_tiling) * (filters % effective_filter_tiling))
        + (effective_channel_tiling * filter_tiling)
        * ((channels % effective_channel_tiling) * (filters // effective_filter_tiling))
        + (effective_channel_tiling * effective_filter_tiling)
        * ((channels % effective_channel_tiling) * (filters % effective_filter_tiling))
    )

    return ifmap_access_count, ofmap_access_count, weight_access_count


def eval_obj(layer_util, layer_latency, layer_access_counts, weights, obj_fn):
    (
        obj,
        util_score,
        latency_score,
        ifmap_access_counts_score,
        ofmap_access_counts_score,
    ) = obj_fn(layer_util, layer_latency, layer_access_counts, weights)

    avg_util = np.average(layer_util)
    avg_latency = np.average(layer_latency)
    avg_ifmap_access_counts = np.average(
        np.array([accesses[0] for accesses in layer_access_counts])
    )
    avg_ofmap_access_counts = np.average(
        np.array([accesses[1] for accesses in layer_access_counts])
    )
    avg_weight_access_counts = np.average(
        np.array([accesses[2] for accesses in layer_access_counts])
    )

    return (
        obj,
        avg_util,
        avg_latency,
        avg_ifmap_access_counts,
        avg_ofmap_access_counts,
        avg_weight_access_counts,
        util_score,
        latency_score,
        ifmap_access_counts_score,
        ofmap_access_counts_score,
    )


def layer_conversion(layer, kernel_sizes_supported_in_direct_mode):

    current_k = layer.kernel_size[0]
    if current_k in kernel_sizes_supported_in_direct_mode:
        current_k = current_k**2
        channels = layer.input_size[1]
        filters = layer.output_size[0]
        ifmap_size = layer.input_size[2] * layer.input_size[3]
    else:
        # convert to 1x1 conv with balanced lowering ifmap dims
        channels = current_k * layer.input_size[1]
        filters = current_k * layer.output_size[0]
        ifmap_size = layer.input_size[2] * layer.output_size[1]
        current_k = 1
    return current_k, channels, filters, ifmap_size


def get_effective_tiling_factors(channel_tiling, filter_tiling, current_k, orientation):
    if orientation == "vertical":
        effective_channel_tiling = channel_tiling
        effective_filter_tiling = filter_tiling // current_k

    elif orientation == "horizontal":
        effective_filter_tiling = filter_tiling
        effective_channel_tiling = channel_tiling // current_k
    else:
        raise Exception("WTF")
    return effective_channel_tiling, effective_filter_tiling


def validate_layer_config(layer: LayerDimensions):
    if layer.stride != (1, 1):
        raise ArgumentError(f"Unsupported layer stride size")


def get_layer_stats_for_arch(
    kernel_sizes_supported_in_direct_mode,
    channel_tiling,
    filter_tiling,
    orientation,
    adjusted_pe_count,
    include_group_conv,
    pad_input,
    layer,
):
    if pad_input and layer.padding != "same":
        layer.input_size[-2] += layer.padding[0]
        layer.input_size[-1] += layer.padding[1]

    if include_group_conv:
        in_channels = int(layer.input_size[-3] / layer.groups)
        out_channels = int(layer.output_size[0] / layer.groups)
        old_in_channels = layer.input_size[-3]
        old_out_channels = layer.output_size[0]

        layer.input_size[-3] = in_channels
        layer.output_size[-3] = out_channels

    current_k, channels, filters, ifmap_size = layer_conversion(
        layer, kernel_sizes_supported_in_direct_mode
    )

    (effective_channel_tiling, effective_filter_tiling,) = get_effective_tiling_factors(
        channel_tiling, filter_tiling, current_k, orientation
    )

    weighted_avg_util, total_tiles = eval_util(
        effective_channel_tiling,
        effective_filter_tiling,
        channels,
        filters,
        current_k,
        adjusted_pe_count,
    )

    layer_util = [weighted_avg_util]

    layer_latency = [
        eval_latency(kernel_sizes_supported_in_direct_mode, layer, total_tiles)
        * layer.groups
    ]

    layer_access_counts = [
        tuple(
            access * layer.groups
            for access in eval_access_counts(
                ifmap_size,
                filters,
                channels,
                effective_filter_tiling,
                effective_channel_tiling,
                channel_tiling,
                filter_tiling,
            )
        )
    ]

    if include_group_conv:
        layer.input_size[-3] = old_in_channels
        layer.output_size[-3] = old_out_channels

    result_dict = {
        "layer_util": layer_util,
        "layer_latency": layer_latency,
        "layer_access_counts": layer_access_counts,
    }

    return result_dict


def eval_arch(
    stats_dict,
    filter_tiling,
    channel_tiling,
    orientation,
    kernel_sizes_supported_in_direct_mode,
    include_group_conv=False,
    pad_input=False,
):

    adjusted_pe_count = filter_tiling * channel_tiling
    aggregate_layer_util = []
    aggregate_layer_latency = []
    aggregate_layer_access_counts = []

    specialized_layer_stats_evaluator = partial(
        get_layer_stats_for_arch,
        kernel_sizes_supported_in_direct_mode,
        channel_tiling,
        filter_tiling,
        orientation,
        adjusted_pe_count,
        include_group_conv,
        pad_input,
    )

    def layer_generator():
        for _, stats in stats_dict.items():
            for _, layer in enumerate(stats["raw_stats"].values()):
                if layer.kernel_size[0] != layer.kernel_size[1]:
                    continue
                yield layer

    result_dict_list = [
        specialized_layer_stats_evaluator(layer) for layer in layer_generator()
    ]

    aggregate_layer_util.extend(
        [val for res in result_dict_list for val in res["layer_util"]]
    )

    aggregate_layer_latency.extend(
        [val for res in result_dict_list for val in res["layer_latency"]]
    )

    aggregate_layer_access_counts.extend(
        [val for res in result_dict_list for val in res["layer_access_counts"]]
    )

    return (
        adjusted_pe_count,
        aggregate_layer_util,
        aggregate_layer_latency,
        aggregate_layer_access_counts,
    )


def get_arch_score(
    stats_dict,
    kernel_sizes_supported_in_direct_mode,
    include_group_conv,
    weights,
    pad_layers,
    obj_fn,
    arch_config,
):
    filter_tiling, channel_tiling, orientation = arch_config
    (adjusted_pe_count, layer_util, layer_latency, layer_access_counts,) = eval_arch(
        stats_dict,
        filter_tiling,
        channel_tiling,
        orientation,
        kernel_sizes_supported_in_direct_mode,
        include_group_conv,
        pad_layers,
    )

    (
        arch_score,
        avg_util,
        avg_latency,
        avg_ifmap_access_counts,
        avg_ofmap_access_counts,
        avg_weight_access_counts,
        util_score,
        latency_score,
        ifmap_access_counts_score,
        ofmap_access_counts_score,
    ) = eval_obj(layer_util, layer_latency, layer_access_counts, weights, obj_fn)

    arch_metrics = {
        "avg_util": avg_util,
        "avg_latency": avg_latency,
        "avg_ifmap_access_counts": avg_ifmap_access_counts / channel_tiling,
        "avg_ofmap_access_counts": avg_ofmap_access_counts / filter_tiling,
        "avg_weight_access_counts": avg_weight_access_counts
        / (channel_tiling * filter_tiling),
    }
    result_dict = {
        "arch_config": arch_config,
        "adjusted_pe_count": adjusted_pe_count,
        "arch_score": arch_score,
        "arch_metrics": arch_metrics,
    }

    return result_dict


def obj_fn(layer_util, layer_latency, layer_access_counts, weights):
    util_scaler = MinMaxScaler()
    latency_scaler = MinMaxScaler()
    ifmap_access_counts_scaler = MinMaxScaler()
    ofmap_access_counts_scaler = MinMaxScaler()

    util_score = np.average(
        util_scaler.fit_transform(np.array(layer_util).reshape(-1, 1))
    )
    latency_score = np.average(
        latency_scaler.fit_transform(
            np.array([1 / l for l in layer_latency]).reshape(-1, 1)
        )
    )
    ifmap_access_counts_score = np.average(
        ifmap_access_counts_scaler.fit_transform(
            np.array([1 / accesses[0] for accesses in layer_access_counts]).reshape(
                -1, 1
            )
        )
    )
    ofmap_access_counts_score = np.average(
        ofmap_access_counts_scaler.fit_transform(
            np.array([1 / accesses[1] for accesses in layer_access_counts]).reshape(
                -1, 1
            )
        )
    )

    res = weights[0] * util_score - (
        weights[1] * latency_score
        + weights[2] * ifmap_access_counts_score
        + weights[3] * ofmap_access_counts_score
    )
    # res =  weights[3]*ofmap_access_counts_score

    return (
        res,
        util_score,
        latency_score,
        ifmap_access_counts_score,
        ofmap_access_counts_score,
    )


def optimize(
    obj_fn=obj_fn,
    pe_count=9 * 3,
    kernel_sizes_supported_in_direct_mode=[1, 3],
    weights=[1, 0, 0, 0],
    allowed_orientations=["vertical", "horizontal"],
    include_group_conv=False,
    stats_dict=None,
    pad_layers=False,
    tiling_is_multiple_of_kernel_size=False,
):
    max_supported_k = max([k**2 for k in kernel_sizes_supported_in_direct_mode])
    max_arch_score = -inf
    opt_arch_metrics = {}
    opt_arch_config = {}

    if stats_dict is None:
        with open("../data/stats_dict.backup", "rb") as backup:
            stats_dict = pickle.load(backup)

    def arch_gen():
        for filter_tiling in factors(pe_count):
            for orientation in allowed_orientations:
                channel_tiling = pe_count // filter_tiling

                if filter_tiling < max_supported_k and orientation == "vertical":
                    continue
                if channel_tiling < max_supported_k and orientation == "horizontal":
                    continue

                if orientation == "horizontal" and tiling_is_multiple_of_kernel_size:
                    if channel_tiling % max_supported_k != 0:
                        continue

                if orientation == "vertical" and tiling_is_multiple_of_kernel_size:
                    if filter_tiling % max_supported_k != 0:
                        continue

                yield filter_tiling, channel_tiling, orientation

    with multiprocessing.Pool(len(os.sched_getaffinity(0))) as multiprocessing_pool:
        result_dict_list = list(
            multiprocessing_pool.map(
                partial(
                    get_arch_score,
                    stats_dict,
                    kernel_sizes_supported_in_direct_mode,
                    include_group_conv,
                    weights,
                    pad_layers,
                    obj_fn,
                ),
                arch_gen(),
            )
        )

    # fun = partial(
    #     get_arch_score,
    #     stats_dict,
    #     kernel_sizes_supported_in_direct_mode,
    #     include_group_conv,
    #     weights,
    #     pad_layers,
    #     obj_fn,
    # )
    # for i in arch_gen():
    #     fun(i)

    for result in result_dict_list:
        arch_score = result["arch_score"]
        arch_metrics = result["arch_metrics"]
        adjusted_pe_count = result["adjusted_pe_count"]
        filter_tiling, channel_tiling, orientation = result["arch_config"]

        if arch_score > max_arch_score:
            max_arch_score = arch_score
            opt_arch_config = {
                "filter_count": filter_tiling,
                "channel_count": channel_tiling,
                "orientation": orientation,
                "adjusted_pe_count": adjusted_pe_count,
                "max_score": max_arch_score,
            }
            opt_arch_metrics = arch_metrics

    return opt_arch_config, opt_arch_metrics


def convert_frontend_testcase_to_tempo_layer(frontend_layer: TestCase):
    return LayerDimensions(
        kernel_size=(frontend_layer.kernel, frontend_layer.kernel),
        stride=(1, 1),
        padding=(0, 0),
        groups=1,
        input_size=[
            1,
            frontend_layer.c_in,
            frontend_layer.ifmap_h,
            frontend_layer.ifmap_w,
        ],
        output_size=[
            frontend_layer.f_out,
            frontend_layer.ifmap_h - frontend_layer.kernel + 1,
            frontend_layer.ifmap_w - frontend_layer.kernel + 1,
        ],
    )


def find_optimal_pe_allocation(model_layers: deque, pe_budget: int):
    stats_dict = {"model": {"raw_stats": {}}}
    for idx, test_case in enumerate(model_layers):
        stats_dict["model"]["raw_stats"][
            idx
        ] = convert_frontend_testcase_to_tempo_layer(test_case)
    return optimize(
        stats_dict=stats_dict,
        pe_count=pe_budget,
        allowed_orientations=["horizontal"],
        include_group_conv=True,
        tiling_is_multiple_of_kernel_size=True,
    )


if __name__ == "__main__":
    r = [324]
    # r = [27, 32, 64, 81, 144, 128, 256, 324, 512, 576]
    # r.extend(list(i**2 for i in range(4, 10)))
    r.sort()
    res_list = []
    for pe in tqdm(r):
        res = optimize(
            pe_count=pe,
            kernel_sizes_supported_in_direct_mode=[1, 3],
            include_group_conv=True,
            pad_layers=True,
        )

        print(res)
        res_list.append(res)
