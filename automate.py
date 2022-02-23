import subprocess
import regex as rx
import pickle

res_dict = {}

k = 1
filter_count = 7
channel_count = 9

iteration_count = 0
pass_count = 0
fail_count = 0

print("STARTING PARAMETER SWEEP")
for ifmap in range(10, 310, 10):
    f_out = 32
    c_in = 32

    print(
        f"ifmap_h: {ifmap}, ifmap_w: {ifmap}, k: {k}, c_in: {c_in}, f_out: {f_out}, filter_count: {filter_count}, channel_count: {channel_count} .... ",
        end="",
    )
    args = (
        "build/tests/estimation_enviornment",
        "--ifmap_h",
        f"{ifmap}",
        "--ifmap_w",
        f"{ifmap}",
        "--k",
        f"{k}",
        "--c_in",
        f"{c_in}",
        "--f_out",
        f"{f_out}",
        "--filter_count",
        f"{filter_count}",
        "--channel_count",
        f"{channel_count}",
    )
    # Or just:
    # args = 'bin/bar -c somefile.xml -d text.txt -r aString -f anotherString'.split()
    popen = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    popen.wait()
    output = popen.stdout.read().decode()

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
        raise Exception(
            f"Neither pass nor fail found so simulation likely crashed with config: \nifmap_h = {ifmap} ifmap_w = {ifmap} k = {k} c_in = {c_in} f_out = {f_out} filter_count = {filter_count} channel_count = {channel_count}"
        )

    res_dict[
        (
            iteration_count,
            ifmap,
            ifmap,
            k,
            c_in,
            f_out,
            filter_count,
            channel_count,
        )
    ] = (
        valid,
        dram,
        weight,
        psum,
        ifmap,
        pe_util,
        latency,
        sim_time,
    )

    pass_count = pass_count + 1 if (valid == "PASS") else pass_count
    fail_count = fail_count + 1 if (valid == "FAIL") else fail_count

    print(valid, end="")
    print(
        f"... PASS_COUNT: {pass_count}, FAIL_COUNT: {fail_count}, TOTAL: {pass_count + fail_count}"
    )

    # with open("ifmap_sweep_dict.pickle", "wb") as handle:
    #     pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    iteration_count += 1
