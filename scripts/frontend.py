from argparse import ArgumentError
from concurrent.futures import thread
import subprocess
from enum import Enum
from random import randint, choice, seed, choices
import threading, queue
from dataclasses import asdict
from time import sleep
from typing import Dict, Tuple, List, Optional
from matplotlib.style import available
from pandas import DataFrame, concat
from pathlib import Path
import os
from timeit import default_timer as timer
import math
import result_pb2
import ModelAnalysis
from ModelAnalysis import (
    ModelDimCollector,
    load_default_input_tensor_for_model,
)
from tqdm import tqdm
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
from TEMPO import find_optimal_pe_allocation
from schema import IfmapLayerDimensions, SimResult, TestCase
import config
from typing import Union
from copy import deepcopy
import pickle
from multiprocessing import Pool
import psutil


os.environ[
    "SC_COPYRIGHT_MESSAGE"
] = "DISABLE"  # Disable system-c copyright message over stdout
Path(config.SUBPROCESS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

seed(config.SEED)


class OperationMode(Enum):
    linear = 0
    conv = 1


class LoweringMethod(Enum):
    balanced = 0
    expensive = 1  # im2col


class VerifyMode(Enum):
    random_test_cases = 0
    network = 1


VERIFY_MODE = VerifyMode.network


def generate_test_cases_queue(count: int):
    test_cases_queue = queue.Queue(0)
    expected_f_in = [2**i for i in range(config.LOG2_FILTER_LOWER, config.LOG2_FILTER_UPPER)]
    expected_c_out = [3] + [
        2**i for i in range(config.LOG2_CHANNEL_LOWER, config.LOG2_CHANNEL_UPPER)
    ]

    for r in range(count):
        op_mode = OperationMode(choices([0, 1], weights=[1, 3], k=1)[0])
        ifmap_size = ofmap_size = math.inf
        while ifmap_size > config.LAYER_SIZE_UB or ofmap_size > config.LAYER_SIZE_UB:
            if op_mode == OperationMode.linear:
                ifmap_w = 1
                ifmap_h = randint(config.IFMAP_LOWER, config.IFMAP_UPPER) ** 2
                kernel = 1
            elif op_mode == OperationMode.conv:
                ifmap_h = ifmap_w = randint(config.IFMAP_LOWER, config.IFMAP_UPPER)
                kernel = choices([1, 3], weights=[1, 3], k=1)[0]
            f_out, c_in = choice(expected_f_in), choice(expected_c_out)
            ifmap_size = ifmap_h * ifmap_w * c_in
            ofmap_size = (ifmap_w - kernel + 1) * (ifmap_h - kernel + 1) * f_out
        arch_filter_counts, arch_channel_counts = choice(
            list(config.ARCH_CONFIG_DICT.values())
        ).values()
        test_cases_queue.put(
            TestCase(
                ifmap_h,
                ifmap_w,
                kernel,
                c_in,
                f_out,
                arch_filter_counts,
                arch_channel_counts,
                groups=1,
            )
        )

    return test_cases_queue


def spawn_simulation_process(worker_id: int, test_case: TestCase):
    args = (
        "../build/hero_sim_backend",
        "--ifmap_h",
        f"{test_case.ifmap_h}",
        "--ifmap_w",
        f"{test_case.ifmap_w}",
        "--k",
        f"{test_case.kernel}",
        "--c_in",
        f"{test_case.c_in}",
        "--f_out",
        f"{test_case.f_out}",
        "--filter_count",
        f"{test_case.arch_filter_count}",
        "--channel_count",
        f"{test_case.arch_channel_count}",
        "--result_as_protobuf",
        "--sim_bias" if test_case.bias is True else "",
    )
    layer_name = test_case.layer_name if test_case.layer_name is not None else ""
    stderr_file_path = os.path.join(
        config.SUBPROCESS_OUTPUT_DIR, f"output_{worker_id}_{layer_name}_stderr.temp"
    )
    stdout_file_path = os.path.join(
        config.SUBPROCESS_OUTPUT_DIR, f"output_{worker_id}_{layer_name}_stdout.temp"
    )

    with open(stderr_file_path, "wb") as stderr_file, open(
        stdout_file_path, "w"
    ) as stdout_file:
        popen = subprocess.Popen(args, stdout=stdout_file, stderr=stderr_file)
        popen.wait()
    with open(stderr_file_path, "rb") as stderr_file, open(
        stdout_file_path, "r"
    ) as stdout_file:
        res = result_pb2.Result()
        res.ParseFromString(stderr_file.read())

    ifmap_size = test_case.ifmap_h * test_case.ifmap_w
    true_ifmap_size = (test_case.ifmap_h - test_case.arch_padding[0]) * (
        test_case.ifmap_w - test_case.arch_padding[1]
    )
    ifmap_reduction_factor = true_ifmap_size / ifmap_size

    ofmap_h = test_case.ifmap_h - test_case.kernel + 1  # assuming stride 1
    ofmap_w = test_case.ifmap_w - test_case.kernel + 1  # assuming stride 1
    ofmap_size = ofmap_h * ofmap_w

    true_ofmap_h = (
        test_case.ifmap_h - test_case.arch_padding[0] - test_case.kernel + 1
    )  # assuming stride 1
    true_ofmap_w = (
        test_case.ifmap_w - test_case.arch_padding[1] - test_case.kernel + 1
    )  # assuming stride 1
    true_ofmap_size = true_ofmap_h * true_ofmap_w
    ofmap_reduction_factor = true_ofmap_size / ofmap_size

    return SimResult(
        valid=res.valid,
        sim_time=res.sim_time,
        dram_load=res.dram_load_access * ifmap_reduction_factor * test_case.groups,
        dram_store=res.dram_store_access * ofmap_reduction_factor * test_case.groups,
        weight=res.weight_access * test_case.groups,
        ifmap=res.ifmap_access * ifmap_reduction_factor * test_case.groups,
        psum=res.psum_access * ofmap_reduction_factor * test_case.groups,
        pe_util=res.avg_util * ifmap_reduction_factor,  # prolly bs
        latency=res.latency
        * test_case.groups,  # unaffected by ifmap reduction (delays would still be required)
        macs=res.macs * test_case.groups,
        reuse_chain=res.reuse_chain_accesses
        * test_case.groups
        * ifmap_reduction_factor,
        max_psum_program = res.max_psum_program,
        max_ifmap_program = res.max_ifmap_program,
        max_ifmap_reuse_chain_program = res.max_ifmap_reuse_chain_program,
        max_pe_program = res.max_pe_program,
    )


def create_new_sim_result_rows(test_case, result, layer_name_tracker):
    rows = []
    for layer_name, model_name in layer_name_tracker[test_case]:
        test_case_with_name = TestCase(
            ifmap_h=test_case.ifmap_h,
            ifmap_w=test_case.ifmap_w,
            kernel=test_case.kernel,
            c_in=test_case.c_in,
            f_out=test_case.f_out,
            groups=test_case.groups,
            arch_padding=test_case.arch_padding,
            arch_filter_count=test_case.arch_filter_count,
            arch_channel_count=test_case.arch_channel_count,
            layer_name=layer_name,
            lowering_ops=test_case.lowering_ops,
            lifting_ops=test_case.lifting_ops,
            bias=test_case.bias,
            model_name=model_name,
        )
        combined_dict = {}
        combined_dict.update(asdict(test_case_with_name))
        combined_dict.update(asdict(result))
        rows.append(DataFrame([combined_dict]))
    return rows


def test_case_worker(
    worker_id,
    test_cases_queue: queue.Queue,
    results_queue: queue.Queue,
):
    while True:
        test_case: TestCase = test_cases_queue.get()
        config.logger.debug(f"worker {worker_id} spawning process with test case\n{test_case}")
        sim_result = spawn_simulation_process(worker_id, test_case)
        results_queue.put((test_case, sim_result))
        test_cases_queue.task_done()


def results_collection_worker(
    worker_id: int,
    test_case_count: int,
    results_queue: queue.Queue,
    done_queue: queue.Queue,
    layer_name_tracker: Dict[TestCase, str] = None,
):
    collection_counter = 0
    results_dataframe = DataFrame()
    aggregate_dataframe = DataFrame()

    while True:
        test_case, result = results_queue.get()

        if layer_name_tracker is None:
            layer_name_tracker = {}
            layer_name_tracker[test_case] = [
                test_case.layer_name if test_case.layer_name is not None else "RANDOM"
            ]

            new_rows = create_new_sim_result_rows(test_case, result, layer_name_tracker)
            layer_name_tracker = None

        else:
            new_rows = create_new_sim_result_rows(test_case, result, layer_name_tracker)

        aggregate_dataframe = concat([aggregate_dataframe, *new_rows])

        if (collection_counter + 1) % config.SAVE_EVERY == 0:
            results_dataframe = concat([results_dataframe, aggregate_dataframe])
            aggregate_dataframe = DataFrame()
            # results_dataframe.to_csv(config.RESULTS_CSV_PATH, index=False)
            percent_complete = int(collection_counter / test_case_count * 100)
            config.logger.info(
                f"Worker {worker_id} processed %{percent_complete} of test cases",
            )

        done_queue.put(results_dataframe)
        collection_counter += 1
        results_queue.task_done()


def pad_ifmap_dims(ifmap_dims: IfmapLayerDimensions, padding: Tuple[int, int]):
    return IfmapLayerDimensions(
        height=ifmap_dims.height + padding[0],
        width=ifmap_dims.width + padding[1],
        channels=ifmap_dims.channels,
    )


def lower_ifmap_and_convert_to_conv(
    ifmap_dims: IfmapLayerDimensions,
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    bias: bool,
    method: LoweringMethod = LoweringMethod.balanced,
) -> Tuple[IfmapLayerDimensions, Conv2d]:

    if ifmap_dims.height != ifmap_dims.width:
        raise NotImplementedError("Asymmetric input feature map cannot be lowered")
    if kernel_size[0] != kernel_size[1]:
        raise NotImplementedError("Asymmetric kernel sizes cannot be lowered")

    # invalid pads due to non (1, 1) strides filtered out during lifting
    if method is LoweringMethod.balanced:
        ofmap_h = ifmap_dims.height - kernel_size[0] + 1
        ifmap_w = 1
        ifmap_h = ifmap_dims.height * ofmap_h
        in_channels = int(in_channels * kernel_size[0])
        out_channels = int(in_channels * kernel_size[0])
        new_dims = IfmapLayerDimensions(ifmap_w, ifmap_h, in_channels)
        layer = Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), bias=bias is not None
        )
        lowering = lifting = (ofmap_h**2) * kernel_size[0]
        return (new_dims, layer, lowering, lifting)
    elif method is LoweringMethod.expensive:
        raise NotImplementedError(
            "Expensive lowering dimension calculation unavailable"
        )
    else:
        raise NotImplementedError("Invalid lowering method requested")


def find_minimal_fmap_padding(
    ifmap_dim: IfmapLayerDimensions, arch_config: Dict[str, int]
):
    arch_channel_count = arch_config["channel_count"] + 2
    min_pad = (0, 0)
    min_ifmap_total_size = math.inf
    for wpad in range(arch_channel_count):
        hpad = (
            arch_channel_count
            - ifmap_dim.height * wpad
            - ifmap_dim.height * ifmap_dim.width
        ) / (ifmap_dim.width + wpad)
        new_ifmap_total_size = (ifmap_dim.height + hpad) * (ifmap_dim.width + wpad)
        if hpad == math.floor(hpad) and new_ifmap_total_size >= arch_channel_count:
            if min_ifmap_total_size > new_ifmap_total_size:
                min_ifmap_total_size = new_ifmap_total_size
                min_pad = (int(hpad), int(wpad))
    return min_pad


def get_layer_equivalents(
    layer_dims,
    directly_supported_kernels: List[Tuple[int, int]],
):
    new_layer_dims = {}
    for layer_name, (ifmap_dims, layer) in layer_dims.items():
        if isinstance(layer, Linear):
            layer_out_channels = layer.out_features
            lowering_ops = lifting_ops = 0
            new_layer_dims[layer_name] = {
                "groups": 1,
                "dims": IfmapLayerDimensions(1, 1, layer_out_channels),
                "conv_layer": Conv2d(
                    new_dims.channels,
                    layer_out_channels,
                    kernel_size=(1, 1),
                    bias=layer.bias is not None,
                ),
                "lowering_ops": lowering_ops,
                "lifting_ops": lifting_ops,
            }
        elif isinstance(layer, Conv2d):
            

            if layer.dilation != (1, 1):
                layer.kernel_size = tuple((layer.kernel_size[i]-1)*layer.dilation[i] + 1 for i in range(2))
            
            if layer.kernel_size[0] != layer.kernel_size[1]:
                raise Exception("Asymmetric Kernels Unsupported")
            
            new_dims = pad_ifmap_dims(ifmap_dims, layer.padding)
            in_channels = int(layer.in_channels / layer.groups)
            new_dims.channels = in_channels
            out_channels = int(layer.out_channels / layer.groups)

            if (
                layer.stride == (1, 1)
                and layer.kernel_size in directly_supported_kernels
            ):
                conv_layer = Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=layer.kernel_size,
                    bias=layer.bias is not None,
                )

                lowering_ops = lifting_ops = 0
            else:
                (
                    new_dims,
                    conv_layer,
                    lowering_ops,
                    lifting_ops,
                ) = lower_ifmap_and_convert_to_conv(
                    new_dims,
                    in_channels,
                    out_channels,
                    bias=layer.bias,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                )

            new_layer_dims[f"{layer_name}"] = {
                "groups": layer.groups,
                "dims": new_dims,
                "conv_layer": conv_layer,
                "lowering_ops": lowering_ops,
                "lifting_ops": lifting_ops,
            }
        else:
            raise TypeError(f"Invalid layer type {type(layer)}")
    return new_layer_dims


def convert_layer_dims_to_test_cases(
    layer_dims: Dict[str, Tuple[IfmapLayerDimensions, Conv2d]],
    arch_config: Dict[str, int],
    model_name: str = None,
):
    test_cases_list = []
    for layer_name, layer_data in layer_dims.items():
        groups = layer_data["groups"]
        ifmap_dims = layer_data["dims"]
        layer = layer_data["conv_layer"]
        lowering_ops = layer_data["lowering_ops"]
        lifting_ops = layer_data["lifting_ops"]
        arch_padding = layer_data["arch_padding"]
        test_cases_list.append(
            TestCase(
                ifmap_h=ifmap_dims.height,
                ifmap_w=ifmap_dims.width,
                c_in=layer.in_channels,
                f_out=layer.out_channels,
                kernel=layer.kernel_size[0],
                groups=groups,
                arch_padding=arch_padding,
                arch_channel_count=arch_config["channel_count"],
                arch_filter_count=arch_config["filter_count"],
                layer_name=layer_name,
                lowering_ops=lowering_ops,
                lifting_ops=lifting_ops,
                bias=layer.bias is not None,
                model_name=model_name,
            )
        )
    return test_cases_list


def launch_workers_with_test_cases(
    test_cases_list: List[TestCase],
    layer_name_tracker: Dict[TestCase, str] = None,
    core_count=config.CORE_COUNT,
):
    test_cases_queue = queue.Queue(0)
    for case in test_cases_list:
        test_cases_queue.put(case)

    queue_size = test_cases_queue.qsize()
    results_queue = queue.Queue(0)
    done_queue = queue.Queue(0)
    for worker_id in range(config.CORE_COUNT):
        threading.Thread(
            target=test_case_worker,
            daemon=True,
            args=[worker_id, test_cases_queue, results_queue],
        ).start()
    threading.Thread(
        target=results_collection_worker,
        daemon=True,
        args=[config.CORE_COUNT, queue_size, results_queue, done_queue, layer_name_tracker],
    ).start()

    for _ in range(queue_size):
        result_df = done_queue.get()

    return result_df


def remove_duplicate_test_cases(
    test_cases_list: List[TestCase], layer_name_tracker=None
):
    if layer_name_tracker is None:
        layer_name_tracker = {}
    for case in test_cases_list:
        layer_name = case.layer_name
        model_name = case.model_name
        case_without_layer_name = TestCase(
            ifmap_h=case.ifmap_h,
            ifmap_w=case.ifmap_w,
            kernel=case.kernel,
            c_in=case.c_in,
            f_out=case.f_out,
            groups=case.groups,
            arch_padding=case.arch_padding,
            arch_filter_count=case.arch_filter_count,
            arch_channel_count=case.arch_channel_count,
            lowering_ops=case.lowering_ops,
            lifting_ops=case.lifting_ops,
            bias=case.bias,
        )
        try:
            layer_name_tracker[case_without_layer_name].append((layer_name, model_name))
        except KeyError:
            layer_name_tracker[case_without_layer_name] = [(layer_name, model_name)]
        except:
            raise
    test_cases_list = []
    for case in layer_name_tracker.keys():
        test_cases_list.append(case)
    return test_cases_list, layer_name_tracker


def pad_layer_dims_based_on_arch_config(layer_dims, arch_config=None):
    if arch_config is not None:
        for layer_name, layer_data in layer_dims.items():
            ifmap_dims = layer_data["dims"]

            if ifmap_dims.height * ifmap_dims.width < arch_config["channel_count"]:
                padding = find_minimal_fmap_padding(ifmap_dims, arch_config)

                ifmap_dims = pad_ifmap_dims(ifmap_dims, padding)

                layer_dims[layer_name]["dims"] = ifmap_dims
                layer_dims[layer_name]["arch_padding"] = padding
            else:
                layer_dims[layer_name]["arch_padding"] = (0, 0)

    return layer_dims


def find_opt_arch(
    models, pe_budget: int, directly_supported_kernels=config.DIRECTLY_SUPPORTED_KERNELS
):
    aggregate_test_case_list = []
    for model in models:
        input = load_default_input_tensor_for_model(model)
        layer_dims = ModelDimCollector.collect_layer_dims_from_model(model, input)
        layer_dims = get_layer_equivalents(layer_dims, directly_supported_kernels)

        test_cases = convert_layer_dims_to_test_cases(layer_dims)
        test_cases, _ = remove_duplicate_test_cases(test_cases)
        aggregate_test_case_list.extend(test_cases)

    arch_config, metrics = find_optimal_pe_allocation(
        aggregate_test_case_list, pe_budget
    )
    arch_config["directly_supported_kernels"] = directly_supported_kernels
    return arch_config, metrics


def create_sub_layers_from_layer_with_large_ifmap(
    layer_data,
    ifmap_ub_per_bank,
    ifmap_bank_count,
    distribute_ifmaps_across_banks=False,
):
    ifmap_dims: IfmapLayerDimensions = layer_data["dims"]
    ifmap_single_size = ifmap_dims.width * ifmap_dims.height

    if ifmap_single_size <= ifmap_ub_per_bank:
        channels_per_bank = ifmap_ub_per_bank // ifmap_single_size
        channels_per_sub_layer = channels_per_bank * ifmap_bank_count
    else:
        if distribute_ifmaps_across_banks is False:
            raise Exception(
                "Input feature map for layer requested is too large to decompose \
                    without enabling bank distribution"
            )

        banks_per_ifmap_single = math.ceil(ifmap_single_size / ifmap_ub_per_bank)
        channels_per_sub_layer = ifmap_bank_count // banks_per_ifmap_single

    sub_layers = ifmap_dims.channels // channels_per_sub_layer

    new_layer_dim_list = []

    if sub_layers == 0:
        new_layer_dim_list.append(layer_data)
        return new_layer_dim_list

    for sub_layer in range(sub_layers):
        new_layer_dim = deepcopy(layer_data)
        new_layer_dim["dims"].channels = channels_per_sub_layer
        conv_op: Conv2d = new_layer_dim["conv_layer"]
        if sub_layer == 0:
            new_layer_dim["conv_layer"] = Conv2d(
                in_channels=channels_per_sub_layer,
                out_channels=conv_op.out_channels,
                kernel_size=conv_op.kernel_size,
                bias=conv_op.bias is not None,
            )
        else:
            new_layer_dim["conv_layer"] = Conv2d(
                in_channels=channels_per_sub_layer,
                out_channels=conv_op.out_channels,
                kernel_size=conv_op.kernel_size,
                bias=True,  # bias required for all sub layers after the first one
            )
            new_layer_dim["lowering_ops"] = 0
            new_layer_dim["lifting_ops"] = 0
        new_layer_dim_list.append(new_layer_dim)

    # do I need one last "extra" sub layer for remaining channels?
    remaining_channels = ifmap_dims.channels % channels_per_sub_layer
    if remaining_channels > 0:
        new_layer_dim = deepcopy(layer_data)
        conv_op: Conv2d = new_layer_dim["conv_layer"]
        new_layer_dim["dims"].channels = remaining_channels
        new_layer_dim["lowering_ops"] = 0
        new_layer_dim["lifting_ops"] = 0
        new_layer_dim["conv_layer"] = Conv2d(
            in_channels=remaining_channels,
            out_channels=conv_op.out_channels,
            kernel_size=conv_op.kernel_size,
            bias=True,  # bias required for all sub layers after the first one
        )
        new_layer_dim_list.append(new_layer_dim)
    return new_layer_dim_list


# need to decompose based on ofmaps too
def decompose_layers_with_large_ifmaps(layer_dims, arch_config: Dict[str, int]):

    ifmap_bank_count = arch_config["channel_count"]
    distribute_ifmaps_across_banks: bool = arch_config["allow_ifmap_distribution"]

    try:
        ifmap_ub = arch_config["ifmap_mem_ub"]
    except KeyError:
        return layer_dims

    ifmap_ub_per_bank = ifmap_ub // ifmap_bank_count
    bank_adjusted_ifmap_ub = ifmap_ub_per_bank * ifmap_bank_count

    if bank_adjusted_ifmap_ub != ifmap_ub:
        bank_adjusted_ifmap_ub_in_kb = bank_adjusted_ifmap_ub / 2**10
        ifmap_ub_in_kb = ifmap_ub / 2**10
        config.logger.warning(
            f"Upper bound for IFmap memory supplied is not a multiple of arch ifmap channel count (Available banks for IFmap), \
                will adjust ub to {bank_adjusted_ifmap_ub_in_kb :.5f} KB Instead of {ifmap_ub_in_kb :.5f} KB"
        )
    new_layer_dims = {}
    for layer_name, layer_data in layer_dims.items():
        ifmap_dims: IfmapLayerDimensions = layer_data["dims"]
        ifmap_single_size = ifmap_dims.width * ifmap_dims.height
        if ifmap_single_size > bank_adjusted_ifmap_ub:
            raise Exception(
                "Input feature map for layer requested is too large to decompose"
            )

        if (
            ifmap_single_size > ifmap_ub_per_bank
            and distribute_ifmaps_across_banks is False
        ):
            raise Exception(
                "Input feature map for layer requested is too large to decompose without enabling bank distribution"
            )

        sub_layers = create_sub_layers_from_layer_with_large_ifmap(
            layer_data,
            ifmap_ub_per_bank,
            ifmap_bank_count,
            distribute_ifmaps_across_banks,
        )
        for layer_idx, layer in enumerate(sub_layers):
            sub_layer_name = f"{layer_name}.c{layer_idx}"
            new_layer_dims[sub_layer_name] = layer

    return new_layer_dims


def create_sub_layers_from_layer_with_large_ofmap(
    layer_data,
    ofmap_ub_per_bank,
    ofmap_bank_count,
    distribute_ofmaps_across_banks=False,
    arch_allows_invalid_windows=True,
):
    ifmap_dims: IfmapLayerDimensions = layer_data["dims"]

    conv_layer: Conv2d = layer_data["conv_layer"]
    layer_filters = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size

    ofmap_height = ifmap_dims.height - kernel_size[0] + 1
    ofmap_width = ifmap_dims.width - kernel_size[1] + 1

    if kernel_size != (1, 1) and arch_allows_invalid_windows is False:
        ofmap_single_size = ifmap_dims.width * ofmap_height
    else:
        ofmap_single_size = ofmap_height * ofmap_width

    if ofmap_single_size <= ofmap_ub_per_bank:
        filters_per_bank = ofmap_ub_per_bank // ofmap_single_size
        filters_per_sub_layer = filters_per_bank * ofmap_bank_count
    else:
        if distribute_ofmaps_across_banks is False:
            raise Exception(
                "Output feature map for layer requested is too large to decompose \
                    without enabling bank distribution"
            )

        banks_per_ofmap_single = math.ceil(ofmap_single_size / ofmap_ub_per_bank)
        filters_per_sub_layer = ofmap_bank_count // banks_per_ofmap_single

    sub_layers = layer_filters // filters_per_sub_layer

    new_layer_dim_list = []

    if sub_layers == 0:
        new_layer_dim_list.append(layer_data)
        return new_layer_dim_list

    for sub_layer in range(sub_layers):
        new_layer_dim = deepcopy(layer_data)
        conv_op: Conv2d = new_layer_dim["conv_layer"]
        new_layer_dim["conv_layer"] = Conv2d(
            in_channels=conv_op.in_channels,
            out_channels=filters_per_sub_layer,
            kernel_size=conv_op.kernel_size,
            bias=conv_op.bias is not None,
        )
        if sub_layer > 0:
            new_layer_dim["lowering_ops"] = 0
            new_layer_dim["lifting_ops"] = 0
        new_layer_dim_list.append(new_layer_dim)

    # do I need one last "extra" sub layer for remaining filters?
    remaining_filters = layer_filters % filters_per_sub_layer
    if remaining_filters > 0:
        new_layer_dim = deepcopy(layer_data)
        conv_op: Conv2d = new_layer_dim["conv_layer"]
        new_layer_dim["conv_layer"] = Conv2d(
            in_channels=conv_op.in_channels,
            out_channels=remaining_filters,
            kernel_size=conv_op.kernel_size,
            bias=conv_op.bias is not None,
        )
        new_layer_dim["lowering_ops"] = 0
        new_layer_dim["lifting_ops"] = 0
        new_layer_dim_list.append(new_layer_dim)
    return new_layer_dim_list


def decompose_layers_with_large_ofmaps(
    layer_dims, arch_config: Dict[str, int], arch_allows_invalid_windows=True
):

    ofmap_bank_count = arch_config["filter_count"]
    distribute_ofmaps_across_banks: bool = arch_config["allow_ofmap_distribution"]

    try:
        ofmap_ub = arch_config["ofmap_mem_ub"]
    except KeyError:
        return layer_dims

    ofmap_ub_per_bank = ofmap_ub // ofmap_bank_count
    bank_adjusted_ofmap_ub = ofmap_ub_per_bank * ofmap_bank_count

    if bank_adjusted_ofmap_ub != ofmap_ub:
        bank_adjusted_ofmap_ub_in_kb = bank_adjusted_ofmap_ub / 2**10
        ofmap_ub_in_kb = ofmap_ub / 2**10
        config.logger.warning(
            f"Upper bound for IFmap memory supplied is not a multiple of arch ifmap channel count (Available banks for IFmap), \
                will adjust ub to {bank_adjusted_ofmap_ub_in_kb :.5f} KB Instead of {ofmap_ub_in_kb :.5f} KB"
        )
    new_layer_dims = {}
    for layer_name, layer_data in layer_dims.items():
        ifmap_dims: IfmapLayerDimensions = layer_data["dims"]
        conv_layer: Conv2d = layer_data["conv_layer"]
        kernel_size = conv_layer.kernel_size
        if kernel_size[0] != kernel_size[1]:
            raise Exception(
                "Asymmetric kernels unsupported when decomposing layers with large ofmaps"
            )

        ofmap_height = ifmap_dims.height - kernel_size[0] + 1
        ofmap_width = ifmap_dims.width - kernel_size[1] + 1

        if kernel_size != (1, 1) and arch_allows_invalid_windows is False:
            ofmap_single_size = ifmap_dims.width * ofmap_height
        else:
            ofmap_single_size = ofmap_height * ofmap_width

        if ofmap_single_size > bank_adjusted_ofmap_ub:
            raise Exception(
                "Output feature map for layer requested is too large to decompose"
            )

        if (
            ofmap_single_size > ofmap_ub_per_bank
            and distribute_ofmaps_across_banks is False
        ):
            raise Exception(
                "Output feature map for layer requested is too large to decompose without enabling bank distribution"
            )

        sub_layers = create_sub_layers_from_layer_with_large_ofmap(
            layer_data,
            ofmap_ub_per_bank,
            ofmap_bank_count,
            distribute_ofmaps_across_banks,
            arch_allows_invalid_windows,
        )
        for layer_idx, layer in enumerate(sub_layers):
            sub_layer_name = f"{layer_name}.f{layer_idx}"
            new_layer_dims[sub_layer_name] = layer

    return new_layer_dims


def convert_collected_model_layers_to_testcases(
    layer_dims, arch_config, model_name=None, layer_name_tracker=None
):
    layer_dims = get_layer_equivalents(
        layer_dims, arch_config["directly_supported_kernels"]
    )
    layer_dims = pad_layer_dims_based_on_arch_config(layer_dims, arch_config)

    layer_dims = decompose_layers_with_large_ofmaps(layer_dims, arch_config)
    layer_dims = decompose_layers_with_large_ifmaps(layer_dims, arch_config)
    test_cases_list = convert_layer_dims_to_test_cases(
        layer_dims, arch_config, model_name
    )
    test_cases_list, layer_name_tracker = remove_duplicate_test_cases(
        test_cases_list, layer_name_tracker
    )

    return test_cases_list, layer_name_tracker


def eval_collected_model_layers(layer_dims, arch_config, model_name):
    test_cases_list, layer_name_tracker = convert_collected_model_layers_to_testcases(
        layer_dims, arch_config, model_name
    )
    result_df = launch_workers_with_test_cases(test_cases_list, layer_name_tracker)
    return result_df


def eval_network(model, arch_config, model_name=None, pre_processed_network=False):
    Path(config.SUBPROCESS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    os.environ["SC_COPYRIGHT_MESSAGE"] = "DISABLE"

    if not pre_processed_network:
        input = load_default_input_tensor_for_model(model)
        layer_dims = ModelDimCollector.collect_layer_dims_from_model(model, input)
    else:
        layer_dims = model
    result_df = eval_collected_model_layers(layer_dims, arch_config, model_name)
    return result_df


if __name__ == "__main__":
    models = []
    models.append(
        (
            "mobilentv3_rw",
            pickle.load(open("../data/processed_models/mobilenetv3_rw.model.pickle", "rb")),
        )
    )

    for model_name, model in models:
        config.RESULTS_CSV_PATH = f"../data/{model_name}.csv"
        res_df = eval_network(model, config.arch_config, model_name=model_name, pre_processed_network=True)
