import pickle
import pandas as pd

from typing import Union, Tuple
import os
from tqdm import tqdm
import pickle
from dataclasses import dataclass, asdict
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
import pandas as pd
from pathlib import Path
from frontend import lower_ifmap_and_convert_to_conv
from config import arch_config
from copy import copy
from dataclasses import asdict
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class LinearLayer:
    width: int
    height: int
    channels: int
    in_features: int
    out_features: int
    bias: bool


@dataclass(frozen=True)
class ConvLayer:
    width: int
    height: int
    channels: int
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int
    bias: bool
    padding_mode: str


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


def lowering_lifting_is_required(conv_layer: ConvLayer):
    return conv_layer.kernel_size not in arch_config[
        "directly_supported_kernels"
    ] or conv_layer.stride != (1, 1)


def calculate_ofmap_dims(
    conv_layer, ignore_stride=False, ignore_dilation=False, ignore_padding=False
):
    ifmap_w = conv_layer.width
    ifmap_h = conv_layer.height
    padding_h = 0 if ignore_padding else conv_layer.padding[0]
    padding_w = 0 if ignore_padding else conv_layer.padding[1]
    kernel_h = conv_layer.kernel_size[0]
    kernel_w = conv_layer.kernel_size[1]
    dilation_h = 1 if ignore_dilation else conv_layer.dilation[0]
    dilation_w = 1 if ignore_dilation else conv_layer.dilation[1]
    stride_h = 1 if ignore_stride else conv_layer.stride[0]
    stride_w = 1 if ignore_stride else conv_layer.stride[1]

    ofmap_h = (
        ((ifmap_h - ((kernel_h - 1) * dilation_h + 1) + 2 * padding_h)) / stride_h
    ) + 1
    ofmap_w = (
        ((ifmap_w - ((kernel_w - 1) * dilation_w + 1) + 2 * padding_w)) / stride_w
    ) + 1

    return (ofmap_h, ofmap_w)


def calculate_conv_macs(conv_layer: ConvLayer, allow_lowering=True):

    filters = conv_layer.out_channels / conv_layer.groups
    channels = conv_layer.in_channels / conv_layer.groups
    groups = conv_layer.groups

    if allow_lowering and lowering_lifting_is_required(conv_layer):
        kernel_h = conv_layer.kernel_size[0]
        kernel_w = conv_layer.kernel_size[1]
        ifmap_w = conv_layer.width
        ifmap_h = conv_layer.height
        assert kernel_h == kernel_w
        assert ifmap_w == ifmap_h

        ofmap_h, ofmap_w = calculate_ofmap_dims(conv_layer, ignore_stride=True)
        assert ofmap_h == ofmap_w
        return (
            (ifmap_h * ofmap_h) * (channels * kernel_h) * (filters * kernel_h) * groups
        )

    else:
        kernel_h = conv_layer.kernel_size[0]
        kernel_w = conv_layer.kernel_size[1]

        ofmap_h, ofmap_w = calculate_ofmap_dims(conv_layer)
        macs = ofmap_h * ofmap_w * kernel_h * kernel_w * channels * filters * groups

        return macs


def calculate_conv_weight_mem_size(conv_layer: ConvLayer):
    filters = conv_layer.out_channels / conv_layer.groups
    channels = conv_layer.in_channels / conv_layer.groups

    kernel_h = conv_layer.kernel_size[0]
    kernel_w = conv_layer.kernel_size[1]
    assert kernel_h == kernel_w
    return kernel_h * kernel_w * channels * filters * conv_layer.groups


def calculate_conv_ifmap_mem_size(conv_layer: ConvLayer, allow_lowering=True):
    channels = conv_layer.in_channels / conv_layer.groups

    if allow_lowering and lowering_lifting_is_required(conv_layer):
        kernel_h = conv_layer.kernel_size[0]
        kernel_w = conv_layer.kernel_size[1]
        ifmap_w = conv_layer.width
        ifmap_h = conv_layer.height
        assert kernel_h == kernel_w

        ofmap_h, ofmap_w = calculate_ofmap_dims(conv_layer, ignore_stride=True)
        assert ofmap_h == ofmap_w
        return (ifmap_h * ofmap_h) * channels * kernel_h * conv_layer.groups

    else:
        ifmap_w = conv_layer.width
        ifmap_h = conv_layer.height
        return ifmap_h * ifmap_w * channels * conv_layer.groups


def calculate_conv_ofmap_mem_size(conv_layer: ConvLayer, allow_lowering=True):
    filters = conv_layer.out_channels / conv_layer.groups
    channels = conv_layer.in_channels / conv_layer.groups

    if allow_lowering and lowering_lifting_is_required(conv_layer):
        ofmap_h, ofmap_w = calculate_ofmap_dims(conv_layer, ignore_stride=True)
        assert conv_layer.kernel_size[0] == conv_layer.kernel_size[1]
        lowered_ofmap_h, lowered_ofmap_w = calculate_ofmap_dims(
            ConvLayer(
                width=1,
                height=conv_layer.height * ofmap_h,
                channels=channels * conv_layer.kernel_size[0],
                in_channels=channels * conv_layer.kernel_size[0],
                out_channels=filters * conv_layer.kernel_size[0],
                kernel_size=(1, 1),
                stride=conv_layer.stride,
                padding=conv_layer.padding,
                dilation=conv_layer.dilation,
                groups=conv_layer.groups,
                padding_mode=conv_layer.padding_mode,
                bias=conv_layer.bias,
            ),
            ignore_stride=True,
            ignore_dilation=True,
            ignore_padding=False,
        )
        return (
            lowered_ofmap_h
            * lowered_ofmap_w
            * filters
            * conv_layer.kernel_size[0]
            * conv_layer.groups
        )

    else:
        ofmap_h, ofmap_w = calculate_ofmap_dims(conv_layer)

        return ofmap_h * ofmap_w * filters * conv_layer.groups


def calculate_avg_ifmap_reuse(conv_layer: ConvLayer):
    ifmap_h = conv_layer.height + conv_layer.padding[0]
    ifmap_w = conv_layer.height + conv_layer.padding[1]
    stride_h = conv_layer.stride[0]
    stride_w = conv_layer.stride[1]
    kernel_h = conv_layer.kernel_size[0]
    kernel_w = conv_layer.kernel_size[1]
    ifmap = np.arange(np.prod(ifmap_dims)).reshape(ifmap_dims)
    ifmap_reuse = np.zeros(ifmap_dims)

    views = sliding_window_view(ifmap, layer.kernel_size)[::stride_x, ::stride_y]


def calculate_linear_macs(linear_layer: LinearLayer):
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    channels = linear_layer.channels
    height = linear_layer.height
    width = linear_layer.width
    assert channels == in_features
    return (height * width) * in_features * out_features


def calculate_linear_ifmap_mem_size(linear_layer: LinearLayer):
    in_features = linear_layer.in_features
    channels = linear_layer.channels
    height = linear_layer.height
    width = linear_layer.width
    assert channels == in_features
    return (height * width) * in_features


def calculate_linear_ofmap_mem_size(linear_layer: LinearLayer):
    height = linear_layer.height
    width = linear_layer.width
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    channels = linear_layer.channels
    assert channels == in_features
    return height * width * out_features


def calculate_linear_weight_mem_size(linear_layer: LinearLayer):
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    return in_features * out_features


def aggregate_profiling_info(cpu_profiling_basepath, arch_results_basepath):
    aggregate_profiling_results = {}
    for model_name in tqdm(os.listdir(arch_results_basepath)):
        profile_path = os.path.join(cpu_profiling_basepath, model_name)
        profile_results = (
            pd.read_csv(profile_path, index_col=False)
            .iloc[:, 1:]
            .set_index("Layer Name")
            .to_dict(orient="index")
        )
        supported_layers_forward_duration = 0
        sanitized_model_name = ".".join(model_name.split(".")[:-1])
        aggregate_profiling_results[sanitized_model_name] = {}
        for layer_name, result in profile_results.items():
            if layer_name != "forward":
                supported_layers_forward_duration += result["Duration"]
            aggregate_profiling_results[sanitized_model_name][layer_name] = result
        aggregate_profiling_results[
            "supported_forward"
        ] = supported_layers_forward_duration

    return aggregate_profiling_results


def generate_unique_layer_tracker():
    model_unique_layers_tracker = {}
    for model_name, layer_dims in tqdm(layer_dims_generator()):
        layer_config_tracker = {}
        for layer_name, (dim, module) in layer_dims.items():
            if isinstance(module, Conv2d):
                hashable_layer = ConvLayer(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    **asdict(dim),
                )
            elif isinstance(module, Linear):
                hashable_layer = LinearLayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    **asdict(dim),
                )
            try:
                layer_config_tracker[hashable_layer].append(layer_name)
            except KeyError as e:
                layer_config_tracker[hashable_layer] = [layer_name]
        model_unique_layers_tracker[model_name] = layer_config_tracker

    with open("../data/model_unique_layers_tracker.pickle", "wb") as file:
        pickle.dump(model_unique_layers_tracker, file)

    return model_unique_layers_tracker


def generate_layer_properties_dict(model_unique_layers_tracker):
    model_param_dict = {}
    for model, layer_dict in model_unique_layers_tracker.items():
        layer_property_dict = {}
        for layer, layer_names_list in layer_dict.items():
            properties = asdict(layer)
            if isinstance(layer, ConvLayer):
                properties["type"] = "conv"
            elif isinstance(layer, LinearLayer):
                properties["type"] = "linear"
                properties["kernel_size"] = "(1, 1)"
            else:
                raise Exception(f"Invalid layer type {type(layer)}")
            for layer_name in layer_names_list:
                layer_property_dict[layer_name] = properties
        for layer_name, properties in layer_property_dict.items():
            model_param_dict[(model, layer_name)] = properties
    return model_param_dict


def generate_layer_metric_dict(model_unique_layers_tracker):
    model_metric_dict = {}
    for model, layer_dict in tqdm(model_unique_layers_tracker.items()):
        layer_property_dict = {}
        for layer, layer_names_list in layer_dict.items():
            if isinstance(layer, ConvLayer):
                properties = {
                    "macs": calculate_conv_macs(layer),
                    "ifmap_mem_size": calculate_conv_ifmap_mem_size(layer),
                    "ofmap_mem_size": calculate_conv_ofmap_mem_size(layer),
                    "weight_mem_size": calculate_conv_weight_mem_size(layer),
                    "original_macs": calculate_conv_macs(layer, allow_lowering=False),
                    "original_ifmap_mem_size": calculate_conv_ifmap_mem_size(
                        layer, allow_lowering=False
                    ),
                    "original_ofmap_mem_size": calculate_conv_ofmap_mem_size(
                        layer, allow_lowering=False
                    ),
                    "original_weight_mem_size": calculate_conv_weight_mem_size(layer),
                    "lowered/lifted": lowering_lifting_is_required(layer),
                }
            elif isinstance(layer, LinearLayer):
                properties = {
                    "macs": calculate_linear_macs(layer),
                    "ifmap_mem_size": calculate_linear_ifmap_mem_size(layer),
                    "weight_mem_size": calculate_linear_weight_mem_size(layer),
                    "ofmap_mem_size": calculate_linear_ofmap_mem_size(layer),
                    "original_macs": calculate_linear_macs(layer),
                    "original_ifmap_mem_size": calculate_linear_ifmap_mem_size(layer),
                    "original_ofmap_mem_size": calculate_linear_ofmap_mem_size(layer),
                    "original_weight_mem_size": calculate_linear_weight_mem_size(layer),
                    "lowered/lifted": False,
                }
            else:
                raise Exception(f"Invalid layer type {type(layer)}")

            for layer_name in layer_names_list:
                layer_property_dict[layer_name] = copy(properties)
        for layer_name, properties in layer_property_dict.items():
            properties["cpu_time"] = aggregate_profiling_results[model][layer_name][
                "Duration"
            ]
            model_metric_dict[(model, layer_name)] = properties
    return model_metric_dict


if __name__ == "__main__":

    cpu_profiling_basepath = Path("../data/profiling")
    arch_results_basepath = Path("../data/arch_results_iofmap_1mb")
    aggregate_profiling_results = aggregate_profiling_info(
        cpu_profiling_basepath, arch_results_basepath
    )

    force_fail_load_model_unique_layers = False
    try:
        if force_fail_load_model_unique_layers:
            raise Exception
        with open("../data/model_unique_layers_tracker.pickle", "rb") as file:
            model_unique_layers_tracker = pickle.load(file)
        generate_model_unique_layers_tracker = False
    except Exception as e:
        model_unique_layers_tracker = generate_unique_layer_tracker()

    model_param_dict = generate_layer_properties_dict(model_unique_layers_tracker)
    layer_df = pd.DataFrame.from_dict(model_param_dict, orient="index")
    model_metric_dict = generate_layer_metric_dict(model_unique_layers_tracker)
    layer_df = layer_df.join(pd.DataFrame.from_dict(model_metric_dict, orient="index"))
    layer_df.to_csv("../data/layer_metrics.csv")
