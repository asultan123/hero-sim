from argparse import ArgumentError
from ast import If
import subprocess
from enum import Enum
from random import randint, choice, seed, choices
import threading, queue
from dataclasses import dataclass, asdict
from tkinter import ARC
from typing import Dict, Tuple, List, Optional
from click import Argument
from pandas import DataFrame, concat
from pathlib import Path
import os
from timeit import default_timer as timer
import colorlog
from colorlog import ColoredFormatter
import math
import result_pb2
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from ModelAnalysis import ModelDimCollector, IfmapLayerDimensions
import urllib
from PIL import Image
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d


os.environ["SC_COPYRIGHT_MESSAGE"] = "DISABLE"

formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(log_color)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG": "green",
        "INFO": "yellow",
        "WARNING": "orange",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)

logger = colorlog.getLogger()

logger.addHandler(handler)

logger.setLevel("DEBUG")


CORE_COUNT = 32
TEST_CASE_COUNT = 100
SAVE_EVERY = 10
RESULTS_CSV_PATH = "../data/verify_results.csv"
SUBPROCESS_OUTPUT_DIR = "../data/subprocess_output"

SEED = 1234
LAYER_SIZE_UB = 2**10
IFMAP_LOWER = 10
IFMAP_UPPER = 224
LOG2_FILTER_LOWER = 0
LOG2_FILTER_UPPER = 10
LOG2_CHANNEL_LOWER = 0
LOG2_CHANNEL_UPPER = 10

DIRECTLY_SUPPORTED_KERNELS = [(1, 1), (3, 3)]

ARCH_CONFIG_DICT = {
    "small": {"filter_count": 9, "channel_count": 9},
    "medium": {"filter_count": 18, "channel_count": 18},
    "large": {"filter_count": 32, "channel_count": 18},
}


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


@dataclass
class SimResult:
    valid: str
    dram: int
    weight: int
    psum: int
    ifmap: int
    pe_util: float
    latency: int
    sim_time: int


@dataclass(frozen=True)
class TestCase:
    ifmap_h: int
    ifmap_w: int
    kernel: int
    c_in: int
    f_out: int
    arch_filter_count: int
    arch_channel_count: int
    layer_name: Optional[str] = None


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
            )
        )

    return test_cases_queue


# CACHE?
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
        res.valid,
        res.dram_access,
        res.weight_access,
        res.ifmap_access,
        res.psum_access,
        res.avg_util,
        res.latency,
        res.sim_time,
    )


def test_case_worker(
    worker_id, test_cases_queue: queue.Queue, results_queue: queue.Queue
):
    while True:
        test_case = test_cases_queue.get()
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
    
    def create_new_sim_result_rows(test_case, result, layer_name_tracker):
        rows = []
        for layer_name in layer_name_tracker[test_case]:
            test_case_with_name = TestCase(
                ifmap_h=test_case.ifmap_h,
                ifmap_w=test_case.ifmap_w,
                kernel=test_case.kernel,
                c_in=test_case.c_in,
                f_out=test_case.f_out,
                arch_filter_count=test_case.arch_filter_count,
                arch_channel_count=test_case.arch_channel_count,
                layer_name=layer_name,
            )
            combined_dict = {}
            combined_dict.update(asdict(test_case_with_name))
            combined_dict.update(asdict(result))
            rows.append(DataFrame([combined_dict]))
        return rows

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

        done_queue.put(collection_counter)
        collection_counter += 1
        results_queue.task_done()


def load_model_from_timm(model_name):
    return timm.create_model(model_name, pretrained=False)


def load_default_input_tensor_for_model(model):
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform(img).unsqueeze(0)


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
    method: LoweringMethod = LoweringMethod.balanced,
) -> Tuple[IfmapLayerDimensions, Conv2d]:

    if ifmap_dims.height != ifmap_dims.width:
        raise ArgumentError("Asymmetric input feature map cannot be lowered")

    # invalid pads due to non (1, 1) strides filtered out during lifting
    if method is LoweringMethod.balanced:
        ifmap_w = 1
        ifmap_h = ifmap_dims.height * (ifmap_dims.height - kernel_size[0] + 1)
        in_channels = int(in_channels * kernel_size[0])
        out_channels = int(in_channels * kernel_size[0])
        new_dims = IfmapLayerDimensions(ifmap_w, ifmap_h, in_channels)
        layer = Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        return (new_dims, layer)
    elif method is LoweringMethod.expensive:
        raise NotImplementedError(
            "Expensive lowering dimension calculation unavailable"
        )
    else:
        raise ArgumentError("Invalid lowering method requested")


def get_layer_equivelents(
    layer_dims: Dict[str, IfmapLayerDimensions],
    arch_config,
    directly_supported_kernels: List[int],
) -> Dict[str, Tuple[IfmapLayerDimensions, Conv2d]]:
    new_layer_dims = {}
    for layer_name, (ifmap_dims, layer) in layer_dims.items():
        if isinstance(layer, Linear):
            new_dims = pad_ifmap_dims(
                ifmap_dims, (arch_config["channel_count"], arch_config["channel_count"])
            )
            layer_out_channels = layer.out_features
            new_layer_dims[layer_name] = (
                new_dims,
                Conv2d(new_dims.channels, layer_out_channels, kernel_size=(1, 1)),
            )
        elif isinstance(layer, Conv2d):
            new_dims = pad_ifmap_dims(ifmap_dims, layer.padding)
            in_channels = int(layer.in_channels / layer.groups)
            new_dims.channels = in_channels
            out_channels = int(layer.out_channels / layer.groups)
            for group_idx in range(layer.groups):
                if (
                    layer.stride == (1, 1)
                    and layer.kernel_size in directly_supported_kernels
                ):
                    new_dims = (
                        new_dims,
                        Conv2d(
                            in_channels, out_channels, kernel_size=layer.kernel_size
                        ),
                    )
                else:
                    new_dims = lower_ifmap_and_convert_to_conv(
                        new_dims,
                        in_channels,
                        out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                    )

                new_layer_dims[f"{layer_name}.grp_{group_idx}"] = new_dims
        else:
            raise TypeError(f"Invalid layer type {type(layer)}")
    return new_layer_dims


def convert_layer_dims_to_test_cases(
    layer_dims: Dict[str, Tuple[IfmapLayerDimensions, Conv2d]],
    arch_config: Dict[str, int],
):
    test_cases_queue = queue.Queue(0)
    for layer_name, (ifmap_dim, layer) in layer_dims.items():
        test_cases_queue.put(
            TestCase(
                ifmap_h=ifmap_dim.height,
                ifmap_w=ifmap_dim.width,
                c_in=layer.in_channels,
                f_out=layer.out_channels,
                kernel=layer.kernel_size[0],
                arch_channel_count=arch_config["channel_count"],
                arch_filter_count=arch_config["filter_count"],
                layer_name=layer_name,
            )
        )
    return test_cases_queue


def launch_workers_with_test_cases(
    test_cases_queue: queue.Queue, layer_name_tracker: Dict[TestCase, str] = None
):
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
        _ = done_queue.get()

    print("Processed all test cases... exiting...")


def remove_duplicate_test_cases(test_cases_queue: queue.Queue[TestCase]):
    layer_name_tracker = {}
    for case in test_cases_queue.queue:
        layer_name = case.layer_name
        case_without_layer_name = TestCase(
            ifmap_h=case.ifmap_h,
            ifmap_w=case.ifmap_w,
            kernel=case.kernel,
            c_in=case.c_in,
            f_out=case.f_out,
            arch_filter_count=case.arch_filter_count,
            arch_channel_count=case.arch_channel_count,
        )
        try:
            layer_name_tracker[case_without_layer_name].append(layer_name)
        except KeyError:
            layer_name_tracker[case_without_layer_name] = [layer_name]
        except:
            raise
    test_cases_queue = queue.Queue()
    for case in layer_name_tracker.keys():
        test_cases_queue.put(case)
    return test_cases_queue, layer_name_tracker


def main():
    if VERIFY_MODE is VerifyMode.network:
        arch_config = ARCH_CONFIG_DICT["medium"]
        model = load_model_from_timm("resnet50")
        input = load_default_input_tensor_for_model(model)
        layer_dims = ModelDimCollector.collect_layer_dims_from_model(model, input)
        layer_dims = get_layer_equivelents(
            layer_dims, arch_config, DIRECTLY_SUPPORTED_KERNELS
        )
        test_cases_queue = convert_layer_dims_to_test_cases(layer_dims, arch_config)
        test_cases_queue, layer_name_tracker = remove_duplicate_test_cases(
            test_cases_queue
        )

    elif VERIFY_MODE is VerifyMode.random_test_cases:
        Path(SUBPROCESS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        test_cases_queue = generate_test_cases_queue(TEST_CASE_COUNT)
        layer_name_tracker = None

    launch_workers_with_test_cases(test_cases_queue, layer_name_tracker)


if __name__ == "__main__":

    start = timer()
    main()
    end = timer()

    print(f"Evaluated {TEST_CASE_COUNT} testcases in {(end - start):.2f} seconds")
