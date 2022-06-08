from typing import Union, Tuple
import os
from tqdm import tqdm
import pickle
from dataclasses import dataclass, asdict
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
import pandas as pd


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
                **asdict(dim)
            )
        elif isinstance(module, Linear):
            hashable_layer = LinearLayer(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                **asdict(dim)
            )
        try:
            layer_config_tracker[hashable_layer].append(layer_name)
        except KeyError as e:
            layer_config_tracker[hashable_layer] = [layer_name]
    model_unique_layers_tracker[model_name] = layer_config_tracker

with open("../data/model_unique_layers_tracker.pickle", "wb") as file:
    pickle.dump(model_unique_layers_tracker, file)
