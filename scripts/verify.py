import subprocess
import regex as rx
from enum import Enum
from random import randint, choice, seed, choices
import threading, queue
from dataclasses import dataclass, asdict
from pandas import DataFrame, concat
from pathlib import Path
import os
from timeit import default_timer as timer
import colorlog
from colorlog import ColoredFormatter
import math

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
RESULTS_CSV_PATH = "./data/verify_results.csv"
SUBPROCESS_OUTPUT_DIR = "./data/subprocess_output"

SEED = 1234
LAYER_SIZE_UB = 2**15
IFMAP_LOWER = 10
IFMAP_UPPER = 224
LOG2_FILTER_LOWER = 0
LOG2_FILTER_UPPER = 10
LOG2_CHANNEL_LOWER = 0
LOG2_CHANNEL_UPPER = 10

seed(SEED)


class OperationMode(Enum):
    linear = 0
    conv = 1


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


@dataclass
class TestCase:
    ifmap_h: int
    ifmap_w: int
    kernel: int
    c_in: int
    f_out: int
    arch_filter_count: int
    arch_channel_count: int


def generate_test_cases_queue(count: int):
    test_cases_queue = queue.Queue(0)
    arch_config_dict = {
        "small": {"filter_count": 9, "channel_count": 9},
        "medium": {"filter_count": 12, "channel_count": 27},
        "large": {"filter_count": 32, "channel_count": 18},
    }
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
            list(arch_config_dict.values())
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


def spawn_simulation_process(worker_id: int, test_case: TestCase):
    args = (
        "build/hero_sim_backend",
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
    )
    output_file_path = os.path.join(SUBPROCESS_OUTPUT_DIR, f"output_{worker_id}.temp")
    with open(output_file_path, "w+") as output_file:
        popen = subprocess.Popen(args, stdout=output_file, stderr=output_file)
        popen.wait()
        output_file.seek(0)
        res_str = output_file.read()

    return res_str


def parse_simulation_process_output(output: str):
    if len(rx.findall("PASS", output)) > 0:
        valid = rx.findall("PASS", output)[0]
        dram = int(rx.findall("DRAM Access +(\w+)", output)[0], 10)
        weight = int(rx.findall("Weight Access +(\w+)", output)[0], 10)
        psum = int(rx.findall("Psum Access +(\w+)", output)[0], 10)
        ifmap = int(rx.findall("Ifmap Access +(\w+)", output)[0], 10)
        pe_util = float(rx.findall("Avg. Pe Util +([\.\w]+)", output)[0])
        latency = int(rx.findall("Latency in cycles +(\w+)", output)[0], 10)
        sim_time = int(rx.findall("Simulated in +(\w+)ms", output)[0], 10)

    elif len(rx.findall("FAIL", output)):
        valid = "FAIL"
        dram = -1
        weight = -1
        psum = -1
        ifmap = -1
        pe_util = -1
        latency = -1
        sim_time = -1

    else:
        valid = "N/A"
        dram = -1
        weight = -1
        psum = -1
        ifmap = -1
        pe_util = -1
        latency = -1
        sim_time = -1

    return SimResult(valid, dram, weight, psum, ifmap, pe_util, latency, sim_time)


def test_case_worker(
    worker_id, test_cases_queue: queue.Queue, results_queue: queue.Queue
):
    while True:
        test_case = test_cases_queue.get()
        logger.debug(f"worker {worker_id} spawning process with test case\n{test_case}")
        output = spawn_simulation_process(worker_id, test_case)
        sim_result = parse_simulation_process_output(output)
        results_queue.put((test_case, sim_result))
        test_cases_queue.task_done()


def results_collection_worker(
    worker_id: int,
    test_case_count: int,
    results_queue: queue.Queue,
    done_queue: queue.Queue,
):
    collection_counter = 0
    results_dataframe = DataFrame()
    aggregate_dataframe = DataFrame()
    while True:
        test_case, result = results_queue.get()
        combined_dict = {}
        combined_dict.update(asdict(test_case))
        combined_dict.update(asdict(result))
        new_row = DataFrame([combined_dict])
        aggregate_dataframe = concat([aggregate_dataframe, new_row])

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


def main():
    Path(SUBPROCESS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    test_cases_queue = generate_test_cases_queue(count=TEST_CASE_COUNT)
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
        args=[CORE_COUNT, TEST_CASE_COUNT, results_queue, done_queue],
    ).start()

    for _ in range(TEST_CASE_COUNT):
        _ = done_queue.get()

    print("Processed all test cases... exiting...")


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    print(
        f"Evaluated {TEST_CASE_COUNT} testcases in {(end - start):.2f} seconds"
    )  # Time in seconds, e.g. 5.38091952400282
