import subprocess
import regex as rx
from enum import Enum
from random import randint, choice, seed, choices
import threading, queue
from dataclasses import dataclass, asdict
from pandas import DataFrame, concat
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)

CORE_COUNT = 32
TEST_CASE_COUNT = 10000
SAVE_EVERY = 10
RESULTS_CSV_PATH = "./verify_results.csv"

SEED = 1234
IFMAP_LOWER = 16
IFMAP_UPPER = 32
LOG2_FILTER_LOWER = 0
LOG2_FILTER_UPPER = 9
LOG2_CHANNEL_LOWER = 0
LOG2_CHANNEL_UPPER = 9

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
        # op_mode = OperationMode.conv
        op_mode = OperationMode(choices([0, 1], weights=[1, 3], k=1)[0])
        if op_mode == OperationMode.linear:
            ifmap_w = 1
            ifmap_h = randint(IFMAP_LOWER, IFMAP_UPPER)**2
            kernel = 1
        elif op_mode == OperationMode.conv:
            ifmap_h = ifmap_w = randint(IFMAP_LOWER, IFMAP_UPPER)
            # kernel = 3
            kernel = choices([1, 3], weights=[1, 3], k=1)[0]
        arch_filter_counts, arch_channel_counts = choice(
            list(arch_config_dict.values())
        ).values()
        f_out, c_in = choice(expected_f_in), choice(expected_c_out)
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


def spawn_simulation_process(test_case: TestCase):
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
    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    res_str = ''
    while True:
        line = popen.stdout.readline()
        res_str += str(line)
        if not line:
            break

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
        logging.debug(f'worker {worker_id} spawning process with test case\n{test_case}')
        output = spawn_simulation_process(test_case)
        sim_result = parse_simulation_process_output(output)
        results_queue.put((test_case, sim_result))
        test_cases_queue.task_done()


def results_collection_worker(
    worker_id: int, test_case_count: int, results_queue: queue.Queue
):
    collection_counter = 0
    results_dataframe = DataFrame()
    while True:
        test_case, result = results_queue.get()
        combined_dict = {}
        combined_dict.update(asdict(test_case))
        combined_dict.update(asdict(result))
        new_row = DataFrame([combined_dict])
        results_dataframe = concat([results_dataframe, new_row])

        if collection_counter % SAVE_EVERY == 0:
            results_dataframe.to_csv(RESULTS_CSV_PATH, index=False)
            percent_complete = int(collection_counter / test_case_count * 100)
            logging.debug(
                f"Worker {worker_id} processed %{percent_complete} of test cases",
            )
        collection_counter += 1
        results_queue.task_done()


if __name__ == "__main__":

    test_cases_queue = generate_test_cases_queue(count=TEST_CASE_COUNT)
    results_queue = queue.Queue(0)
    for worker_id in range(CORE_COUNT):
        threading.Thread(
            target=test_case_worker,
            daemon=True,
            args=[worker_id, test_cases_queue, results_queue],
        ).start()
    threading.Thread(
        target=results_collection_worker,
        daemon=True,
        args=[CORE_COUNT, TEST_CASE_COUNT, results_queue],
    ).start()
    test_cases_queue.join()
    results_queue.join()
    print("Processed all test cases... exiting...")


# k = 1
# filter_count = 7
# channel_count = 9

# iteration_count = 0
# pass_count = 0
# fail_count = 0

# print("STARTING PARAMETER SWEEP")
# for ifmap in range(10, 310, 10):
#     f_out = 32
#     c_in = 32

#     print(
#         f"ifmap_h: {ifmap}, ifmap_w: {ifmap}, k: {k}, c_in: {c_in}, f_out: {f_out}, filter_count: {filter_count}, channel_count: {channel_count} .... ",
#         end="",
#     )
#     args = (
#         "build/tests/estimation_enviornment",
#         "--ifmap_h",
#         f"{ifmap}",
#         "--ifmap_w",
#         f"{ifmap}",
#         "--k",
#         f"{k}",
#         "--c_in",
#         f"{c_in}",
#         "--f_out",
#         f"{f_out}",
#         "--filter_count",
#         f"{filter_count}",
#         "--channel_count",
#         f"{channel_count}",
#     )
#     # Or just:
#     # args = 'bin/bar -c somefile.xml -d text.txt -r aString -f anotherString'.split()
#     popen = subprocess.Popen(
#         args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#     )
#     popen.wait()
#     output = popen.stdout.read().decode()

#     if len(rx.findall("PASS", output)) > 0:
#         valid = rx.findall("PASS", output)[0]
#         dram = int(rx.findall("DRAM Access +(\w+)", output)[0], 10)
#         weight = int(rx.findall("Weight Access +(\w+)", output)[0], 10)
#         psum = int(rx.findall("Psum Access +(\w+)", output)[0], 10)
#         ifmap = int(rx.findall("Ifmap Access +(\w+)", output)[0], 10)
#         pe_util = float(rx.findall("Avg. Pe Util +([\.\w]+)", output)[0])
#         latency = int(rx.findall("Latency in cycles +(\w+)", output)[0], 10)
#         sim_time = int(rx.findall("Simulated in +(\w+)ms", output)[0], 10)

#     elif len(rx.findall("FAIL", output)):
#         valid = "FAIL"
#         dram = -1
#         weight = -1
#         psum = -1
#         ifmap = -1
#         pe_util = -1
#         latency = -1
#         sim_time = -1

#     else:
#         raise Exception(
#             f"Neither pass nor fail found so simulation likely crashed with config: \nifmap_h = {ifmap} ifmap_w = {ifmap} k = {k} c_in = {c_in} f_out = {f_out} filter_count = {filter_count} channel_count = {channel_count}"
#         )

#     res_dict[
#         (
#             iteration_count,
#             ifmap,
#             ifmap,
#             k,
#             c_in,
#             f_out,
#             filter_count,
#             channel_count,
#         )
#     ] = (
#         valid,
#         dram,
#         weight,
#         psum,
#         ifmap,
#         pe_util,
#         latency,
#         sim_time,
#     )

#     pass_count = pass_count + 1 if (valid == "PASS") else pass_count
#     fail_count = fail_count + 1 if (valid == "FAIL") else fail_count

#     print(valid, end="")
#     print(
#         f"... PASS_COUNT: {pass_count}, FAIL_COUNT: {fail_count}, TOTAL: {pass_count + fail_count}"
#     )

#     # with open("ifmap_sweep_dict.pickle", "wb") as handle:
#     #     pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     iteration_count += 1
