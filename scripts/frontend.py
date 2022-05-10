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
from config import *
from typing import Union
from copy import deepcopy
import pickle
from multiprocessing import Pool
import psutil

RESULTS_CSV_PATH = ""

os.environ[
    "SC_COPYRIGHT_MESSAGE"
] = "DISABLE"  # Disable system-c copyright message over stdout
Path(SUBPROCESS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

seed(SEED)


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
    expected_f_in = [2**i for i in range(LOG2_FILTER_LOWER, LOG2_FILTER_UPPER)]
    expected_c_out = [3] + [
        2**i for i in range(LOG2_CHANNEL_LOWER, LOG2_CHANNEL_UPPER)
    ]

    for r in range(count):
        op_mode = OperationMode(choices([0, 1], weights=[1, 3], k=1)[0])
        ifmap_size = ofmap_size = math.inf
        while ifmap_size > LAYER_SIZE_UB or ofmap_size > LAYER_SIZE_UB:
            if op_mode == OperationMode.linear:
                ifmap_w = 1
                ifmap_h = randint(IFMAP_LOWER, IFMAP_UPPER) ** 2
                kernel = 1
            elif op_mode == OperationMode.conv:
                ifmap_h = ifmap_w = randint(IFMAP_LOWER, IFMAP_UPPER)
                kernel = choices([1, 3], weights=[1, 3], k=1)[0]
            f_out, c_in = choice(expected_f_in), choice(expected_c_out)
            ifmap_size = ifmap_h * ifmap_w * c_in
            ofmap_size = (ifmap_w - kernel + 1) * (ifmap_h - kernel + 1) * f_out
        arch_filter_counts, arch_channel_counts = choice(
            list(ARCH_CONFIG_DICT.values())
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
        SUBPROCESS_OUTPUT_DIR, f"output_{worker_id}_{layer_name}_stderr.temp"
    )
    stdout_file_path = os.path.join(
        SUBPROCESS_OUTPUT_DIR, f"output_{worker_id}_{layer_name}_stdout.temp"
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

    return SimResult(
        valid=res.valid,
        dram=res.dram_access,
        weight=res.weight_access,
        psum=res.ifmap_access,
        ifmap=res.psum_access,
        pe_util=res.avg_util,
        latency=res.latency,
        sim_time=res.sim_time,
        macs=res.macs,
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
        logger.debug(f"worker {worker_id} spawning process with test case\n{test_case}")
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

        if (collection_counter + 1) % SAVE_EVERY == 0:
            results_dataframe = concat([results_dataframe, aggregate_dataframe])
            aggregate_dataframe = DataFrame()
            results_dataframe.to_csv(RESULTS_CSV_PATH, index=False)
            percent_complete = int(collection_counter / test_case_count * 100)
            logger.info(
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
        test_cases_list.append(
            TestCase(
                ifmap_h=ifmap_dims.height,
                ifmap_w=ifmap_dims.width,
                c_in=layer.in_channels,
                f_out=layer.out_channels,
                kernel=layer.kernel_size[0],
                groups=groups,
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
    core_count=CORE_COUNT,
):
    test_cases_queue = queue.Queue(0)
    for case in test_cases_list:
        test_cases_queue.put(case)

    queue_size = test_cases_queue.qsize()
    results_queue = queue.Queue(0)
    done_queue = queue.Queue(0)
    for worker_id in range(CORE_COUNT):
        threading.Thread(
            target=test_case_worker,
            daemon=True,
            args=[worker_id, test_cases_queue, results_queue],
        ).start()
    threading.Thread(
        target=results_collection_worker,
        daemon=True,
        args=[CORE_COUNT, queue_size, results_queue, done_queue, layer_name_tracker],
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
                ifmap_dims = pad_ifmap_dims(
                    ifmap_dims, find_minimal_fmap_padding(ifmap_dims, arch_config)
                )

                layer_dims[layer_name]["dims"] = ifmap_dims
    return layer_dims


def find_opt_arch(
    models, pe_budget: int, directly_supported_kernels=DIRECTLY_SUPPORTED_KERNELS
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

        banks_per_ifmap_single = math.ceil(
            ifmap_single_size/ ifmap_ub_per_bank
        )
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
            )  # Copy lifting lowering costs here
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


def decompose_layers_with_large_ofmaps(layer_dims, ofmap_ub: Union[int, None] = None):
    if ofmap_ub is None:
        return layer_dims

    new_layer_dims = {}
    for layer_name, layer_data in layer_dims.items():
        ifmap_dims: IfmapLayerDimensions = layer_data["dims"]
        ifmap_single_size = ifmap_dims.width * ifmap_dims.height
        if ifmap_single_size > ifmap_ub:
            raise Exception(
                "Input feature map for layer requested is too large to decompose"
            )

        total_ifmap_tensor_size = ifmap_dims.channels * ifmap_single_size
        if (
            total_ifmap_tensor_size <= ifmap_ub
        ):  # need to consider size of single ifmap bank
            new_layer_dims[layer_name] = layer_data
            continue

        sub_layers = create_sub_layers_from_layer_with_large_ifmap(layer_data, ifmap_ub)
        for layer_idx, layer in enumerate(sub_layers):
            sub_layer_name = f"{layer_name}.s{layer_idx}"
            new_layer_dims[sub_layer_name] = layer

    return new_layer_dims
    # ifmap_size = ifmap_dims.


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
        logger.warning(
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
            layer_data, ifmap_ub_per_bank, ifmap_bank_count, distribute_ifmaps_across_banks
        )
        for layer_idx, layer in enumerate(sub_layers):
            sub_layer_name = f"{layer_name}.s{layer_idx}"
            new_layer_dims[sub_layer_name] = layer

    return new_layer_dims


def convert_collected_model_layers_to_testcases(
    layer_dims, arch_config, model_name=None, layer_name_tracker=None
):
    layer_dims = get_layer_equivalents(
        layer_dims, arch_config["directly_supported_kernels"]
    )
    layer_dims = pad_layer_dims_based_on_arch_config(layer_dims, arch_config)

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


def eval_network(model, arch_config, model_name=None, processed_network=False):
    Path(SUBPROCESS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    os.environ["SC_COPYRIGHT_MESSAGE"] = "DISABLE"

    if not processed_network:
        input = load_default_input_tensor_for_model(model)
        layer_dims = ModelDimCollector.collect_layer_dims_from_model(model, input)
    else:
        layer_dims = model
    result_df = eval_collected_model_layers(layer_dims, arch_config, model_name)
    return result_df


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


if __name__ == "__main__":
    models = []
    models.append(
        (
            "mobilenetv3_rw",
            pickle.load(
                open("../data/processed_models/mobilenetv3_rw.model.pickle", "rb")
            ),
        )
    )

    arch_config = {
        "filter_count": 32,
        "channel_count": 18,
        "directly_supported_kernels": [(1, 1), (3, 3)],
        "ifmap_mem_ub": 2**17,
        "allow_ifmap_distribution": True
    }
    for model_name, model in models:
        RESULTS_CSV_PATH = f"../data/{model_name}.csv"
        eval_network(model, arch_config, model_name=model_name, processed_network=True)

    # arch_config = {
    #     "filter_count": 32,
    #     "channel_count": 18,
    #     "directly_supported_kernels": [(1, 1), (3, 3)],
    #     "ifmap_mem_ub": 2**20,
    # }

    # RESULTS_CSV_PATH = "../data/timm_0_5314.csv"

    # with open("../data/timm_lib_testcases.pickle", "rb") as file:
    #     test_case_list, layer_name_tracker = pickle.load(file)

    # test_case_list = sorted(test_case_list, reverse=True)
    # result_df = launch_workers_with_test_cases(
    #     test_case_list[2000:], layer_name_tracker
    # )

    # layer_name_tracker = {}
    # for model_name, layer_dims in tqdm(layer_dims_generator()):
    #     test_case_list, layer_name_tracker = convert_collected_model_layers_to_testcases(
    #         layer_dims, arch_config, model_name, layer_name_tracker
    #     )

    # #     total_test_cases_count += len(cases)
    # print(len(test_case_list))
