from argparse import ArgumentError
from calendar import c
from optparse import Option
from click import option
import torch
from collections import OrderedDict, Counter
from functools import partial
from dataclasses import dataclass
from typing import Tuple, List
from copy import deepcopy
from math import prod
from enum import Enum
import torch
from PIL import Image
from torchvision import transforms, models
import timm
from tqdm import tqdm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from math import floor, ceil
import numpy as np
import urllib
import pickle
from typing import Optional


class LayerType(Enum):
    conv = 1
    linear = 2


@dataclass
class LayerDimensions:
    ifmap_w: int
    ifmap_h: int
    c_in: int
    f_out: int
    layer_type: LayerType
    kernel_size: Optional[Tuple[int, int]] = None
    stride: Optional[Tuple[int, int]] = None
    padding: Optional[Tuple[int, int]] = None
    groups: Optional[int] = None


class ModelDimCollector:
    default_targets = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]

    def __init__(self):
        self.model_stats = OrderedDict()
        self.hooks = []

    def get_next_target_layer(
        self,
        model,
        target_layers=default_targets,
    ):
        for name, module in model.named_modules():
            if type(module) in target_layers:
                yield (name, module)

    def extract_dims(self, name, module, input, output):
        if isinstance(module, torch.nn.modules.linear.Linear):
            dims = LayerDimensions(
                ifmap_h=1,
                ifmap_w=1,
                c_in=module.in_features,
                f_out=module.out_features,
                layer_type=LayerType.linear,
            )
        elif isinstance(module, torch.nn.modules.conv.Conv2d):
            dims = LayerDimensions(
                ifmap_h=input[0].size()[-2],
                ifmap_w=input[0].size()[-1],
                c_in=module.in_channels,
                f_out=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                groups=module.groups,
                padding=module.padding,
                layer_type=LayerType.conv,
            )
        else:
            raise ArgumentError(
                f"Unsupported module type {type} found during dimensions extraction"
            )
        self.model_stats[name] = dims

    def attach_collection_hooks_to_model(self, model):
        for name, layer in self.get_next_target_layer(model):
            layer_collector = partial(self.extract_dims, name)
            self.hooks.append(layer.register_forward_hook(layer_collector))

    def detach_stats_collection_hooks(self):
        for hook in self.hooks:
            hook.remove()

    @classmethod
    def collect_stats_from_model(cls, model, input_batch):
        collector = cls()
        collector.attach_collection_hooks_to_model(model)
        if torch.cuda.is_available():
            model.cuda()
            input_batch.cuda()
        model.eval()
        with torch.no_grad():
            model(input_batch)
        model.cpu()
        collector.detach_stats_collection_hooks()
        collected_stats = deepcopy(collector.model_stats)
        return collected_stats


model_list = [
    (model_name, timm.create_model(model_name, pretrained=False))
    for model_name in ["resnet50", "vgg19", "mobilenetv3_rw"]
]
model_input_image_config = {
    model_name: resolve_data_config({}, model=model) for model_name, model in model_list
}

last_config = list(model_input_image_config.values())[0]
for config in list(model_input_image_config.values())[1:]:
    if config["input_size"] != last_config["input_size"]:
        raise Exception(
            "Mismatch in required input image dimensions for desired models"
        )

url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "dog.jpg",
)
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert("RGB")

stats_dict = {}
for model_name, model in model_list:
    transform = create_transform(**model_input_image_config[model_name])
    input_batch = transform(img).unsqueeze(0)
    stats_dict[model_name] = ModelDimCollector.collect_stats_from_model(
        model, input_batch
    )
    