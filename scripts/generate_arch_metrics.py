
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as se
import pickle
from collections import Counter
import numpy as np
import os
import pickle
from timeit import timeit
import json
from timeit import default_timer as timer
from pathlib import Path
import shutil
import math
import matplotlib.pyplot as plt
from math import sqrt
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import os
import multiprocessing
from config import energy_model, arch_config


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



def calculate_layer_bw(
    layer_dram_load, layer_dram_store, layer_latency, clk_freq_in_ghz
):
    load_bw = (layer_dram_load / layer_latency) * (clk_freq_in_ghz * 10**9)
    store_bw = (layer_dram_store / layer_latency) * (clk_freq_in_ghz * 10**9)
    return load_bw, store_bw


def calculate_mac_energy(macs):
    return energy_model['mac'](macs)


def calculate_ifmap_access_energy(ifmap, arch_config):
    ifmap_bank_count = arch_config["channel_count"]
    ifmap_access_per_bank = ifmap / ifmap_bank_count
    ifmap_bank_size = arch_config["ifmap_mem_ub"] / ifmap_bank_count
    ifmap_precision = 8
    ifmap_sram_energy = (
        energy_model["sram"](
            ifmap_access_per_bank,
            ifmap_bank_size,
            ifmap_precision,
        )
        * ifmap_bank_count
    )
    return ifmap_sram_energy


def calculate_psum_access_energy(psum, arch_config):
    psum_bank_count = arch_config["filter_count"]
    psum_access_per_bank = psum / psum_bank_count
    psum_bank_size = arch_config["ofmap_mem_ub"] / psum_bank_count
    psum_precision = 16
    psum_sram_energy = (
        energy_model["sram"](
            psum_access_per_bank,
            psum_bank_size,
            psum_precision,
        )
        * psum_bank_count
    )
    return psum_sram_energy


def calculate_weight_access_energy(weight, arch_config):
    weight_bank_count = arch_config["filter_count"] * \
        arch_config["channel_count"]
    weight_access_per_bank = weight / weight_bank_count
    weight_bank_size = arch_config["weight_bank_size"]
    weight_precision = 8
    weight_sram_energy = (
        energy_model["sram"](weight_access_per_bank,
                             weight_bank_size, weight_precision)
        * weight_bank_count
    )
    return weight_sram_energy


def calculate_reuse_chain_access_energy(reuse_chain, arch_config):
    reuse_chain_bank_count = arch_config["channel_count"] / 9 * 2
    reuse_chain_access_per_bank = reuse_chain / reuse_chain_bank_count
    reuse_chain_bank_size = arch_config["reuse_chain_bank_size"]
    reuse_chain_precision = 8
    reuse_chain_energy = (
        energy_model["sram"](
            reuse_chain_access_per_bank,
            reuse_chain_bank_size,
            reuse_chain_precision,
        )
        * reuse_chain_bank_count
    )
    return reuse_chain_energy


def calculate_dram_access_energy(dram_load, dram_store):
    dram_precision = 8
    dram_energy = energy_model["dram"](dram_load + dram_store, dram_precision)
    return dram_energy


def calculate_layer_energy(dram_load, dram_store, ifmap, psum, reuse_chain, weight, arch_config):

    dram_energy = calculate_dram_access_energy(dram_load, dram_store)
    ifmap_sram_energy = calculate_ifmap_access_energy(ifmap, arch_config)
    psum_sram_energy = calculate_psum_access_energy(psum, arch_config)
    reuse_chain_energy = calculate_reuse_chain_access_energy(
        reuse_chain, arch_config)
    weight_sram_energy = calculate_weight_access_energy(weight, arch_config)

    return (
        dram_energy,
        ifmap_sram_energy,
        psum_sram_energy,
        reuse_chain_energy,
        weight_sram_energy,
    )


def calculate_layer_latency_in_ns(layer_latency, clk_freq_in_ghz):
    return layer_latency * 1 / clk_freq_in_ghz


def calculate_estimated_fps(layer_latency, clk_freq):
    latency_in_seconds = (
        calculate_layer_latency_in_ns(layer_latency, clk_freq) / 10**9
    )
    return 1 / latency_in_seconds


def generate_metrics():
    cpu_profiling_basepath = Path('../data/profiling')
    arch_results_basepath = Path('../data/arch_results')

    avg_model_metric = {}
    for model_name in tqdm(os.listdir(arch_results_basepath)):
        sanitized_model_name = '.'.join(model_name.split('.')[:-1])
        model_path = os.path.join(arch_results_basepath, model_name)
        profile_path = os.path.join(cpu_profiling_basepath, model_name)
        arch_metrics = pd.read_csv(model_path, index_col=0)
        cpu_profile_results = pd.read_csv(
            profile_path, index_col=False).iloc[:, 1:]
        cpu_profile_results = cpu_profile_results.set_index('Layer Name')
        layers_simulated = arch_metrics.to_dict('index')
        avg_model_metric[sanitized_model_name] = {}
        for layer in layers_simulated.keys():
            cpu_dur_ns = cpu_profile_results.loc[layer, 'Duration']*10**3
            latency = arch_metrics.loc[layer, 'latency']
            latency = calculate_layer_latency_in_ns(latency, clk_freq)
            speedup = cpu_dur_ns / latency
            avg_model_metric[sanitized_model_name][layer] = {}
            avg_model_metric[sanitized_model_name][layer]['speedup'] = speedup
            percent_of_compute = cpu_dur_ns / \
                (cpu_profile_results.loc['forward', 'Duration']*10**3)
            avg_model_metric[sanitized_model_name][layer]['percent_of_compute'] = percent_of_compute
            dram_load = arch_metrics.loc[layer, 'dram_load']
            dram_store = arch_metrics.loc[layer, 'dram_store']
            ifmap = arch_metrics.loc[layer, 'ifmap']
            weight = arch_metrics.loc[layer, 'weight']
            psum = arch_metrics.loc[layer, 'psum']
            reuse_chain = arch_metrics.loc[layer, 'reuse_chain']
            pe_util = arch_metrics.loc[layer, 'pe_util']
            dram_energy, ifmap_sram_energy, psum_sram_energy, reuse_chain_sram_energy, weight_sram_energy = calculate_layer_energy(
                dram_load, dram_store, ifmap, psum, reuse_chain, weight, arch_config)
            mac_energy = calculate_mac_energy(weight)
            load_bw, store_bw = calculate_layer_bw(
                dram_load, dram_store, latency, clk_freq)
            avg_model_metric[sanitized_model_name][layer]['latency'] = latency
            avg_model_metric[sanitized_model_name][layer]['load_bw'] = load_bw
            avg_model_metric[sanitized_model_name][layer]['store_bw'] = store_bw
            avg_model_metric[sanitized_model_name][layer]['dram_energy'] = dram_energy
            avg_model_metric[sanitized_model_name][layer]['ifmap_sram_energy'] = ifmap_sram_energy
            avg_model_metric[sanitized_model_name][layer]['psum_sram_energy'] = psum_sram_energy
            avg_model_metric[sanitized_model_name][layer]['reuse_chain_sram_energy'] = reuse_chain_sram_energy
            avg_model_metric[sanitized_model_name][layer]['weight_sram_energy'] = weight_sram_energy
            avg_model_metric[sanitized_model_name][layer]['mac_energy'] = mac_energy
            avg_model_metric[sanitized_model_name][layer]['util'] = pe_util
    return avg_model_metric

def get_per_network_arch_sim_results():
    if arch_results_basepath.exists() and arch_results_basepath.is_dir():
        shutil.rmtree(arch_results_basepath, ignore_errors=True)

    os.makedirs(arch_results_basepath)

    result_df = pd.read_csv(timm_csv_path)
    model_name_list = result_df["model_name"].unique()
    with multiprocessing.Pool(len(os.sched_getaffinity(0))) as pool:
        for _ in tqdm(
            pool.imap_unordered(collect_model_results, model_name_list),
            total=len(model_name_list),
        ):
            pass
    

if __name__ == "__main__":
    
    clk_freq = 1

    arch_results_basepath = Path("../data/arch_results_iofmap_1mb")
    timm_csv_path = Path("../data/timm_1mb.csv")

    get_per_network_arch_sim_results()
    
    arch_metrics = generate_metrics()

    multiindex_arch_dict = {}
    for model_name, layer_dict in arch_metrics.items():
        for layer_name, metric_dict in layer_dict.items():
            multiindex_arch_dict[(model_name, layer_name)] = metric_dict

    arch_metrics_df = pd.DataFrame.from_dict(multiindex_arch_dict, orient='index')
    arch_metrics_df.to_csv('../data/arch_metrics_iofmap_1mb.csv')
